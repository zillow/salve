"""
Based on code found at
https://gitlab.zgtools.net/zillow/rmx/research/scripts-insights/open_platform_utils/-/raw/zind_cleanup/zfm360/visualize_zfm360_data.py

Data can be found at:
/mnt/data/zhiqiangw/ZInD_release/complete_zind_paper_final_localized_json_6_3_21

conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
"""

import collections
import glob
import json
import os
from enum import Enum
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from argoverse.utils.polyline_density import get_polyline_length
from argoverse.utils.interpolate import interp_arc, get_duplicate_indices_1d
from argoverse.utils.json_utils import read_json_file, save_json_dict
from argoverse.utils.sim2 import Sim2
from shapely.geometry import MultiPolygon, Point, Polygon

from sim3_align_dw import align_points_sim3, rotmat2d
from pano_data import FloorData, PanoData, WDO
from logger_utils import get_logger


logger = get_logger()

# The type of supported polygon/wall/point objects.
class PolygonType(Enum):
    ROOM = "room"
    WINDOW = "window"
    DOOR = "door"
    OPENING = "opening"
    PRIMARY_CAMERA = "primary_camera"
    SECONDARY_CAMERA = "secondary_camera"
    PIN_LABEL = "pin_label"


PolygonTypeMapping = {"windows": PolygonType.WINDOW, "doors": PolygonType.DOOR, "openings": PolygonType.OPENING}


# multiply all x-coordinates or y-coordinates by -1, to transfer origin from upper-left, to bottom-left
# (Reflection about either axis, would need additional rotation if reflect over x-axis)


def are_visibly_adjacent(pano1_obj: PanoData, pano2_obj: PanoData) -> bool:
    """ """
    DIST_THRESH = 0.1
    # do they share a door or window?

    from shapely.geometry import LineString

    for wdo1 in pano1_obj.windows + pano1_obj.doors + pano1_obj.openings:
        poly1 = LineString(wdo1.vertices_global_2d)

        # plt.scatter(wdo1.vertices_global_2d[:,0], wdo1.vertices_global_2d[:,1], 40, color="r", alpha=0.2)

        for wdo2 in pano2_obj.windows + pano2_obj.doors + pano2_obj.openings:

            # plt.scatter(wdo2.vertices_global_2d[:,0], wdo2.vertices_global_2d[:,1], 20, color="b", alpha=0.2)

            poly2 = LineString(wdo2.vertices_global_2d)
            if poly1.hausdorff_distance(poly2) < DIST_THRESH:
                return True

    # plt.axis("equal")
    # plt.show()

    return False


def align_by_wdo(hypotheses_save_root: str, building_id: str, pano_dir: str, json_annot_fpath: str) -> None:
    """Save candidate alignment Sim(2) transformations to disk as JSON files.

    For every pano, try to align it to another pano.

    These pairwise costs can be used in a meta-algorithm:
        while there are any unmatched rooms?
        try all possible matches at every step. compute costs, and choose the best greedily.
        or, growing consensus

    Args:
        hypotheses_save_root: base directory where alignment hypotheses will be saved
        building_id:
        pano_dir:
        json_annot_fpath:
    """
    floor_map_json = read_json_file(json_annot_fpath)

    if "merger" not in floor_map_json:
        logger.error(f"Building {building_id} does not have `merger` data, skipping...")
        return

    merger_data = floor_map_json["merger"]

    floor_dominant_rotation = {}
    for floor_id, floor_data in merger_data.items():

        logger.info("--------------------------------")
        logger.info("--------------------------------")
        logger.info("--------------------------------")
        logger.info(f"On building {building_id}, floor {floor_id}...")
        logger.info("--------------------------------")
        logger.info("--------------------------------")
        logger.info("--------------------------------")

        fd = FloorData.from_json(floor_data, floor_id)

        floor_n_valid_configurations = 0
        floor_n_invalid_configurations = 0

        pano_dict = {pano_obj.id: pano_obj for pano_obj in fd.panos}

        pano_ids = sorted(list(pano_dict.keys()))
        for i1 in pano_ids:

            for i2 in pano_ids:

                # compute only upper diagonal, since symmetric
                if i1 >= i2:
                    continue

                logger.info(f"\tOn pano pair ({i1},{i2})")
                # _ = plot_room_layout(pano_dict[i1], coord_frame="local")
                # _ = plot_room_layout(pano_dict[i2], coord_frame="local")

                visibly_adjacent = are_visibly_adjacent(pano_dict[i1], pano_dict[i2])

                try:
                    possible_alignment_info, num_invalid_configurations = align_rooms_by_wd(
                        pano_dict[i1], pano_dict[i2]
                    )
                except Exception:
                    logger.exception("Failure in `align_rooms_by_wd()`, skipping... ")
                    continue

                floor_n_valid_configurations += len(possible_alignment_info)
                floor_n_invalid_configurations += num_invalid_configurations

                # given wTi1, wTi2, then i2Ti1 = i2Tw * wTi1 = i2Ti1
                i2Ti1_gt = pano_dict[i2].global_SIM2_local.inverse().compose(pano_dict[i1].global_SIM2_local)
                gt_fname = f"{hypotheses_save_root}/{building_id}/{floor_id}/gt_alignment_exact/{i1}_{i2}.json"
                if visibly_adjacent:
                    save_Sim2(gt_fname, i2Ti1_gt)
                    expected = i2Ti1_gt.rotation.T @ i2Ti1_gt.rotation
                    # print("Identity? ", np.round(expected, 1))
                    if not np.allclose(expected, np.eye(2), atol=1e-6):
                        import pdb

                        pdb.set_trace()

                # remove redundant transformations
                pruned_possible_alignment_info = prune_to_unique_sim2_objs(possible_alignment_info)

                labels = []
                for k, (i2Ti1, alignment_object) in enumerate(pruned_possible_alignment_info):

                    if obj_almost_equal(i2Ti1, i2Ti1_gt):
                        label = "aligned"
                        save_dir = f"{hypotheses_save_root}/{building_id}/{floor_id}/gt_alignment_approx"
                    else:
                        label = "misaligned"
                        save_dir = f"{hypotheses_save_root}/{building_id}/{floor_id}/incorrect_alignment"
                    labels.append(label)

                    proposed_fname = f"{save_dir}/{i1}_{i2}_{alignment_object}___variant_{k}.json"
                    save_Sim2(proposed_fname, i2Ti1)

                    # print(f"\t GT {i2Ti1_gt.scale:.2f} ", np.round(i2Ti1_gt.translation,1))
                    # print(f"\t    {i2Ti1.scale:.2f} ", np.round(i2Ti1.translation,1), label, "visibly adjacent?", visibly_adjacent)

                    # print()
                    # print()

                if visibly_adjacent:
                    GT_valid = "aligned" in labels
                else:
                    GT_valid = "aligned" not in labels

                # such as (14,15) from building 000, floor 01, where doors are separated incorrectly in GT
                if not GT_valid:
                    logger.warning(f"\tGT invalid for Building {building_id}, Floor {floor_id}: ({i1},{i2})")

        logger.info(f"floor_n_valid_configurations: {floor_n_valid_configurations}")
        logger.info(f"floor_n_invalid_configurations: {floor_n_invalid_configurations}")


def obj_almost_equal(i2Ti1: Sim2, i2Ti1_: Sim2) -> bool:
    """ """
    angle1 = i2Ti1.theta_deg
    angle2 = i2Ti1_.theta_deg

    # print(f"\t\tTrans: {i2Ti1.translation} vs. {i2Ti1_.translation}")
    # print(f"\t\tScale: {i2Ti1.scale:.1f} vs. {i2Ti1_.scale:.1f}")
    # print(f"\t\tAngle: {angle1:.1f} vs. {angle2:.1f}")

    if not np.allclose(i2Ti1.translation, i2Ti1_.translation, atol=0.2):
        return False

    if not np.isclose(i2Ti1.scale, i2Ti1_.scale, atol=0.2):
        return False

    if not np.isclose(angle1, angle2, atol=5):
        return False

    return True


def prune_to_unique_sim2_objs(possible_alignment_info: List[Tuple[Sim2, str]]) -> List[Tuple[Sim2, str]]:
    """ """
    pruned_possible_alignment_info = []

    for j, (i2Ti1, alignment_object) in enumerate(possible_alignment_info):

        item_is_used = False
        for (i2Ti1_, _) in pruned_possible_alignment_info:

            if i2Ti1 == i2Ti1_:
                item_is_used = True

        if not item_is_used:
            pruned_possible_alignment_info += [(i2Ti1, alignment_object)]

    num_orig_objs = len(possible_alignment_info)
    num_pruned_objs = len(pruned_possible_alignment_info)

    verbose = False
    if verbose:
        logger.info(f"Pruned from {num_orig_objs} to {num_pruned_objs}")
    return pruned_possible_alignment_info


def test_prune_to_unique_sim2_objs() -> None:
    """ """
    wR1 = np.eye(2)
    wt1 = np.array([0, 1])
    ws1 = 1.5

    wR2 = np.array([[0, 1], [1, 0]])
    wt2 = np.array([1, 2])
    ws2 = 3.0

    possible_alignment_info = [
        (Sim2(wR1, wt1, ws1), "window"),
        (Sim2(wR1, wt1, ws1), "window"),
        (Sim2(wR2, wt2, ws2), "window"),
        (Sim2(wR1, wt1, ws1), "window"),
    ]
    pruned_possible_alignment_info = prune_to_unique_sim2_objs(possible_alignment_info)
    assert len(pruned_possible_alignment_info) == 2

    assert pruned_possible_alignment_info[0][0].scale == 1.5
    assert pruned_possible_alignment_info[1][0].scale == 3.0


def save_Sim2(save_fpath: str, i2Ti1: Sim2) -> None:
    """
    Args:
        save_fpath
        i2Ti1: transformation that takes points in frame1, and brings them into frame2
    """
    if not Path(save_fpath).exists():
        os.makedirs(Path(save_fpath).parent, exist_ok=True)

    dict_for_serialization = {
        "R": i2Ti1.rotation.flatten().tolist(),
        "t": i2Ti1.translation.flatten().tolist(),
        "s": i2Ti1.scale,
    }
    save_json_dict(save_fpath, dict_for_serialization)


def test_get_relative_angle() -> None:
    """ """
    vec1 = np.array([5, 0])
    vec2 = np.array([0, 9])

    angle_deg = get_relative_angle(vec1, vec2)
    assert angle_deg == 90.0


def normalize_vec(v: np.ndarray) -> np.ndarray:
    """Convert vector to equivalent unit-length version."""
    return v / np.linalg.norm(v)


def get_relative_angle(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Get angle in degrees of two vectors. We normalize them first to unit length."""

    # expect vector to be 2d or 3d
    assert vec1.size in [2, 3]
    assert vec2.size in [2, 3]

    vec1 = normalize_vec(vec1)
    vec2 = normalize_vec(vec2)

    dot_product = np.dot(vec1, vec2)
    assert dot_product >= -1.0 and dot_product <= 1.0
    angle_rad = np.arccos(dot_product)
    angle_deg = np.rad2deg(angle_rad)
    return angle_deg


def test_align_rooms_by_wd() -> None:
    """ """
    wTi5 = Sim2(
        R=np.array([[0.999897, -0.01435102], [0.01435102, 0.999897]], dtype=np.float32),
        t=np.array([0.7860708, -1.57248], dtype=np.float32),
        s=0.4042260417272217,
    )
    wTi8 = Sim2(
        R=np.array([[0.02998102, -0.99955046], [0.99955046, 0.02998102]], dtype=np.float32),
        t=np.array([0.91035557, -3.2141], dtype=np.float32),
        s=0.4042260417272217,
    )

    # fmt: off
    pano1_obj = PanoData(
        id=5,
        global_SIM2_local=wTi5,
        room_vertices_local_2d=np.array(
            [
                [ 1.46363621, -2.43808616],
                [ 1.3643741 ,  0.5424695 ],
                [ 0.73380685,  0.52146958],
                [ 0.7149462 ,  1.08780075],
                [ 0.4670652 ,  1.07954551],
                [ 0.46914653,  1.01704912],
                [-1.2252865 ,  0.96061904],
                [-1.10924507, -2.5237714 ]
            ]),
        image_path='panos/floor_01_partial_room_05_pano_5.jpg',
        label='living room',
        doors=[],
        windows=[
            WDO(
                global_SIM2_local=wTi5,
                pt1=[-1.0367953294361147, -2.5213585867749635],
                pt2=[-0.4661345615720372, -2.5023537435761822],
                bottom_z=-0.5746298535133153,
                top_z=0.38684337323286566,
                type='windows'
            ),
            WDO(
                global_SIM2_local=wTi5,
                pt1=[0.823799786466513, -2.45939477144822],
                pt2=[1.404932996095547, -2.4400411621788427],
                bottom_z=-0.5885416433689703,
                top_z=0.3591070365687572,
                type='windows'
            )
        ],
        openings=[]
    )


    pano2_obj = PanoData(
        id=8,
        global_SIM2_local=wTi8,
        room_vertices_local_2d=np.array(
            [
                [-0.7336625 , -1.3968136 ],
                [ 2.23956454, -1.16554334],
                [ 2.19063694, -0.53652654],
                [ 2.75557561, -0.4925832 ],
                [ 2.73634178, -0.2453117 ],
                [ 2.67399906, -0.25016098],
                [ 2.54252291,  1.44010577],
                [-0.93330008,  1.16974146]
            ]), 
        image_path='panos/floor_01_partial_room_05_pano_8.jpg',
        label='living room',
        doors=[],
        windows=[
            WDO(
                global_SIM2_local=wTi8,
                pt1=[-0.9276784906829552, 1.0974698581331057],
                pt2=[-0.8833992085857922, 0.5282122352406332],
                bottom_z=-0.5746298535133153,
                top_z=0.38684337323286566,
                type='windows'
            ),
            WDO(
                global_SIM2_local=wTi8,
                pt1=[-0.7833093301499523, -0.758550412558342],
                pt2=[-0.7382174598580689, -1.338254727497497],
                bottom_z=-0.5885416433689703,
                top_z=0.3591070365687572,
                type='windows'
            )
        ],
        openings=[]
    )

    # fmt: on
    possible_alignment_info, _ = align_rooms_by_wd(pano1_obj, pano2_obj)
    assert len(possible_alignment_info) == 3


def align_rooms_by_wd(
    pano1_obj: PanoData, pano2_obj: PanoData, visualize: bool = False
) -> Tuple[List[Tuple[Sim2, str]], int]:
    """
    Window-Window correspondences must be established. May have to find all possible pairwise choices, or ICP?

    Cohen16: A single match between an indoor and outdoor window determines an alignment hypothesis
    Computing the alignment boils down to finding a similarity transformation between the models, which
    can be computed from three point correspondences in the general case and from two point matches if
    the gravity direction is known.

    TODO: maybe perform alignment in 2d, instead of 3d? so we have less unknowns?

    What heuristic tells us if they should be identity or mirrored in configuration?
    Are there any new WDs that should not be visible? walls should not cross on top of each other? know same-room connections, first

    Cannot expect a match for each door or window. Find nearest neighbor -- but then choose min dist on rows or cols?
    may not be visible?
    - Rooms can be joined at windows.
    - Rooms may be joined at a door.
    - Predicted wall cannot lie behind another wall, if line of sight is clear.

    Note: when fitting Sim(3), note that if the heights are the same, but door width scale is different, perfect door-width alignment
    cannot be possible, since the height figures into the least squares problem.

    Args:
        pano1_obj
        pano2_obj
        alignment_object: "door" or "window"

    Returns:
        possible_alignment_info: list of tuples (i2Ti1, alignment_object) where i2Ti1 is an alignment transformation
        num_invalid_configurations: 
    """
    verbose = False

    pano1_id = pano1_obj.id
    pano2_id = pano2_obj.id

    num_invalid_configurations = 0
    possible_alignment_info = []

    # import pdb; pdb.set_trace()
    for alignment_object in ["door", "window", "opening"]:

        if alignment_object == "door":
            pano1_wds = pano1_obj.doors
            pano2_wds = pano2_obj.doors

        elif alignment_object == "window":
            pano1_wds = pano1_obj.windows
            pano2_wds = pano2_obj.windows

        elif alignment_object == "opening":
            pano1_wds = pano1_obj.openings
            pano2_wds = pano2_obj.openings

        # try every possible pairwise combination, for this object type
        for i, pano1_wd in enumerate(pano1_wds):
            pano1_wd_pts = pano1_wd.polygon_vertices_local_3d
            # sample_points_along_bbox_boundary(wd), # TODO: add in the 3d linear interpolation

            for j, pano2_wd in enumerate(pano2_wds):

                if alignment_object == "door":
                    plausible_configurations = ["identity", "rotated"]
                elif alignment_object == "window":
                    plausible_configurations = ["identity"]
                elif alignment_object == "opening":
                    plausible_configurations = ["identity", "rotated"]

                for configuration in plausible_configurations:

                    if verbose:
                        logger.debug(f"\t{alignment_object} {i}/{j} {configuration}")

                    if configuration == "rotated":
                        pano2_wd_ = pano2_wd.get_rotated_version()
                    else:
                        pano2_wd_ = pano2_wd

                    pano2_wd_pts = pano2_wd_.polygon_vertices_local_3d
                    # sample_points_along_bbox_boundary(wd)

                    # if visualize:
                    #     plt.close("all")

                    #     all_wd_verts_1 = get_all_pano_wd_vertices(pano1_obj)
                    #     all_wd_verts_2 = get_all_pano_wd_vertices(pano2_obj)
                    #     plt.scatter(-all_wd_verts_1[:,0], all_wd_verts_1[:,1], 10, color='k', marker='+')
                    #     plt.scatter(-all_wd_verts_2[:,0], all_wd_verts_2[:,1], 10, color='g', marker='+')

                    #     plot_room_walls(pano1_obj)
                    #     plot_room_walls(pano2_obj)

                    #     plt.plot(-pano1_wd.polygon_vertices_local_3d[:,0], pano1_wd.polygon_vertices_local_3d[:,1], color="r", linewidth=5, alpha=0.2)
                    #     plt.plot(-pano2_wd_.polygon_vertices_local_3d[:,0], pano2_wd_.polygon_vertices_local_3d[:,1], color="b", linewidth=5, alpha=0.2)

                    #     plt.axis("equal")
                    #     plt.title("Step 1: Before alignment")
                    #     #os.makedirs(f"debug_plots/{pano1_id}_{pano2_id}", exist_ok=True)
                    #     #plt.savefig(f"debug_plots/{pano1_id}_{pano2_id}/step1_{i}_{j}.jpg")
                    #     plt.show()
                    #     plt.close("all")

                    i2Ti1, aligned_pts1 = align_points_sim3(pano2_wd_pts, pano1_wd_pts)

                    # TODO: score hypotheses by reprojection error, or registration nearest neighbor-distances
                    # evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)

                    aligned_pts1 = aligned_pts1[:, :2]
                    pano2_wd_pts = pano2_wd_pts[:, :2]

                    if visualize:
                        plt.scatter(pano2_wd_pts[:, 0], pano2_wd_pts[:, 1], 200, color="r", marker=".")
                        plt.scatter(aligned_pts1[:, 0], aligned_pts1[:, 1], 50, color="b", marker=".")

                    all_pano1_pts = get_all_pano_wd_vertices(pano1_obj)
                    all_pano2_pts = get_all_pano_wd_vertices(pano2_obj)

                    all_pano1_pts = i2Ti1.transform_from(all_pano1_pts[:, :2])

                    all_pano1_pts = all_pano1_pts[:, :2]
                    all_pano2_pts = all_pano2_pts[:, :2]

                    if visualize:
                        plt.scatter(all_pano1_pts[:, 0], all_pano1_pts[:, 1], 200, color="g", marker="+")
                        plt.scatter(all_pano2_pts[:, 0], all_pano2_pts[:, 1], 50, color="m", marker="+")

                    pano1_room_vertices = pano1_obj.room_vertices_local_2d
                    pano1_room_vertices = i2Ti1.transform_from(pano1_room_vertices)
                    pano2_room_vertices = pano2_obj.room_vertices_local_2d

                    pano1_room_vertices = pano1_room_vertices[:, :2]
                    pano2_room_vertices = pano2_room_vertices[:, :2]

                    is_valid = determine_invalid_wall_overlap(
                        pano1_id, pano2_id, i, j, pano1_room_vertices, pano2_room_vertices, wall_buffer_m=0.3
                    )
                    # logger.error("Pano1 room verts: %s", str(pano1_room_vertices))
                    # logger.error("Pano2 room verts: %s", str(pano2_room_vertices))

                    if is_valid:
                        possible_alignment_info += [(i2Ti1, alignment_object)]
                        classification = "valid"
                    else:
                        num_invalid_configurations += 1
                        classification = "invalid"

                    if visualize:
                        plot_room_walls(pano1_obj, i2Ti1, linewidth=10)
                        plot_room_walls(pano2_obj, linewidth=1)

                        plt.scatter(aligned_pts1[:, 0], aligned_pts1[:, 1], 10, color="r", marker="+")
                        plt.plot(aligned_pts1[:, 0], aligned_pts1[:, 1], color="r", linewidth=10, alpha=0.2)

                        plt.scatter(pano2_wd_pts[:, 0], pano2_wd_pts[:, 1], 10, color="g", marker="+")
                        plt.plot(pano2_wd_pts[:, 0], pano2_wd_pts[:, 1], color="g", linewidth=5, alpha=0.1)

                        # plt.plot(inter_poly_verts[:,0],inter_poly_verts[:,1], color='m')

                        plt.title(
                            f"Step 3: Match: ({pano1_id},{pano2_id}): valid={is_valid}, aligned via {alignment_object}, \n  config={configuration}"
                        )
                        # window_normals_compatible={window_normals_compatible},
                        plt.axis("equal")
                        os.makedirs(f"debug_plots/{classification}", exist_ok=True)
                        plt.savefig(
                            f"debug_plots/{classification}/{pano1_id}_{pano2_id}___step3_{configuration}_{i}_{j}.jpg"
                        )

                        # plt.show()
                        plt.close("all")

    return possible_alignment_info, num_invalid_configurations


def shrink_polygon(polygon: Polygon, shrink_factor: float = 0.10) -> Polygon:
    """
    Reference: https://stackoverflow.com/questions/49558464/shrink-polygon-using-corner-coordinates
    
    Args:
        shrink_factor: shrink by 10%
    """
    xs = list(polygon.exterior.coords.xy[0])
    ys = list(polygon.exterior.coords.xy[1])
    # find the minimum volume enclosing bounding box, and treat its center as polygon centroid
    x_center = 0.5 * min(xs) + 0.5 * max(xs)
    y_center = 0.5 * min(ys) + 0.5 * max(ys)
    min_corner = Point(min(xs), min(ys))
    max_corner = Point(max(xs), max(ys))
    center = Point(x_center, y_center)
    shrink_distance = center.distance(min_corner) * shrink_factor

    polygon_shrunk = polygon.buffer(-shrink_distance)  # shrink

    # It's possible for a MultiPolygon to result as a result of the buffer operation, i.e. especially for unusual shapes
    # We choose the largest polygon inside the MultiPolygon, and return it
    if isinstance(polygon_shrunk, MultiPolygon):
        subpolygon_areas = [subpolygon.area for subpolygon in polygon_shrunk.geoms]
        largest_poly_idx = np.argmax(subpolygon_areas)
        polygon_shrunk = polygon_shrunk.geoms[largest_poly_idx]

    return polygon_shrunk


def test_shrink_polygon() -> None:
    """
    If not handled correctly, will see the following error:
        *** AttributeError: 'MultiPolygon' object has no attribute 'exterior'
    """
    def draw_polyline(polyline: np.ndarray, color: str) -> None:
        """ """
        plt.scatter(polyline[:,0], polyline[:,1], 10, color=color, marker='.')
        plt.plot(polyline[:,0], polyline[:,1], color=color)

    pano1_room_vertices = np.array(
        [
            [ 0.61807389, -1.0028074 ],
            [ 0.59331251, -0.48251453],
            [ 0.63846121, -0.38975602],
            [ 0.81566386, -0.02569123],
            [ 0.85433859,  0.05376642],
            [-1.9087475 ,  1.3986739 ],
            [-0.71553403,  3.85014409],
            [ 2.87482109,  2.10250285],
            [ 2.51753773,  1.36848825],
            [ 2.26585724,  1.49099615],
            [ 1.31355939, -0.46543567],
            [ 1.32937937, -1.00994635]
        ]
    )
    polygon1 = Polygon(pano1_room_vertices)
    shrunk_poly1 = shrink_polygon(polygon1)

    assert np.isclose(shrunk_poly1.area, 6.275, atol=1e-3)

    # draw_polyline(pano1_room_vertices, "b")

    # colors = ["r", "g"]
    # for i in range(2):
    #     print(f"poly {i}: ", shrunk_poly1.geoms[i].area)
    #     poly = np.array(list(shrunk_poly1.geoms[i].exterior.coords))
    #     draw_polyline(poly, colors[i])
    # plt.show()

    shrunk_poly1_verts = np.array(list(shrunk_poly1.exterior.coords))

    # draw_polyline(shrunk_poly1_verts, "r")
    # plt.show()

    assert isinstance(shrunk_poly1_verts, np.ndarray)


def interp_evenly_spaced_points(polyline: np.ndarray, interval_m) -> np.ndarray:
    """Nx2 polyline to Mx2 polyline, for waypoint every `interval_m` meters"""

    length_m = get_polyline_length(polyline)
    n_waypoints = int(np.ceil(length_m / interval_m))
    px, py = eliminate_duplicates_2d(polyline[:, 0], py=polyline[:, 1])
    interp_polyline = interp_arc(t=n_waypoints, px=px, py=py)

    return interp_polyline


def test_interp_evenly_spaced_points() -> None:
    """ """
    pano2_room_vertices = np.array(
        [
            [ 3.41491678,  0.82735686],
            [ 2.5812492 , -2.36060637],
            [ 0.2083626 , -1.74008522],
            [ 0.53871724, -0.47680178],
            [ 0.40395381, -0.4415605 ],
            [ 0.40395381, -0.4415605 ],
            [-0.36244272, -0.24114416],
            [-0.36244272, -0.24114416],
            [-0.56108295, -0.18919879],
            [-0.14397634,  1.40582611],
            [ 0.06767395,  1.35047855],
            [ 0.15388028,  1.68013345]
        ]
    )
    pano2_room_vertices_interp = interp_evenly_spaced_points(pano2_room_vertices, interval_m=0.1)  # meters

    assert isinstance(pano2_room_vertices_interp, np.ndarray)
    assert pano2_room_vertices_interp.shape == (104,2)


# def test_interp_arc() -> None:
#     """ """
#     polyline = np.array(
#         [
#             [ 3.41491678,  0.82735686],
#             [ 2.5812492 , -2.36060637],
#             [ 0.2083626 , -1.74008522],
#             [ 0.53871724, -0.47680178],
#             [ 0.40395381, -0.4415605 ],
#             [ 0.40395381, -0.4415605 ],
#             [-0.36244272, -0.24114416],
#             [-0.36244272, -0.24114416],
#             [-0.56108295, -0.18919879],
#             [-0.14397634,  1.40582611],
#             [ 0.06767395,  1.35047855],
#             [ 0.15388028,  1.68013345]
#         ]
#     )
#     n_waypoints = 104
#     import pdb; pdb.set_trace()
#     interp_polyline = interp_arc(t=n_waypoints, px=polyline[:, 0], py=polyline[:, 1])

#     assert isinstance(interp_polyline, np.ndarray)


def eliminate_duplicates_2d(px: np.ndarray, py: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """

    Note: Differs from the argoverse implementation.

    We compare indices to ensure that deleted values are exactly
    adjacent to each other in the polyline sequence.
    """
    num_pts = px.shape[0]
    assert px.shape[0] == py.shape[0]
    px_dup_inds = get_duplicate_indices_1d(px)
    py_dup_inds = get_duplicate_indices_1d(py)
    shared_dup_inds = np.intersect1d(px_dup_inds, py_dup_inds)

    px = np.delete(px, [shared_dup_inds])
    py = np.delete(py, [shared_dup_inds])

    return px, py



def test_eliminate_duplicates_2d() -> None:
    """Ensure two duplicated waypoints are removed.

    Duplicates are at indices 4,5 and 6,7, so rows 5 and 7 should be removed.
    """
    polyline = np.array(
        [
            [ 3.41491678,  0.82735686], # 0
            [ 2.5812492 , -2.36060637], # 1
            [ 0.2083626 , -1.74008522], # 2
            [ 0.53871724, -0.47680178], # 3
            [ 0.40395381, -0.4415605 ], # 4
            [ 0.40395381, -0.4415605 ], # 5
            [-0.36244272, -0.24114416], # 6
            [-0.36244272, -0.24114416], # 7
            [-0.56108295, -0.18919879], # 8
            [-0.14397634,  1.40582611], # 9
            [ 0.06767395,  1.35047855], # 10
            [ 0.15388028,  1.68013345] # 11
        ]
    )
    px, py = eliminate_duplicates_2d(px=polyline[:,0], py=polyline[:,1])
    polyline_no_dups = np.stack([px,py], axis=-1)

    expected_polyline_no_dups = np.array(
        [
            [ 3.41491678,  0.82735686],
            [ 2.5812492 , -2.36060637],
            [ 0.2083626 , -1.74008522],
            [ 0.53871724, -0.47680178],
            [ 0.40395381, -0.4415605 ],
            [-0.36244272, -0.24114416],
            [-0.56108295, -0.18919879],
            [-0.14397634,  1.40582611],
            [ 0.06767395,  1.35047855],
            [ 0.15388028,  1.68013345]
        ]
    )
    assert np.allclose(polyline_no_dups, expected_polyline_no_dups)


def determine_invalid_wall_overlap(
    pano1_id: int,
    pano2_id: int,
    i: int,
    j: int,
    pano1_room_vertices: np.ndarray,
    pano2_room_vertices: np.ndarray,
    wall_buffer_m: float = 0.3,
    visualize: bool = False,
) -> bool:
    """
    TODO: consider adding allowed_overlap_pct: float = 0.01
        Args:
            pano1_id: panorama id
            pano2_id: panorama id
            i: id of WDO to match from pano1
            j: id of WDO to match from pano2

        TODO: use `wall_buffer_m`

        Returns:
    """
    polygon1 = Polygon(pano1_room_vertices)
    polygon2 = Polygon(pano2_room_vertices)

    pano1_room_vertices_interp = interp_evenly_spaced_points(pano1_room_vertices, interval_m=0.1)  # meters
    pano2_room_vertices_interp = interp_evenly_spaced_points(pano2_room_vertices, interval_m=0.1)  # meters

    # # should be in the same coordinate frame, now
    # inter_poly = polygon1.intersection(polygon2)
    # inter_poly_area = inter_poly.area

    # inter_poly_verts = np.array(list(inter_poly.exterior.coords))

    shrunk_poly1 = shrink_polygon(polygon1)
    shrunk_poly1_verts = np.array(list(shrunk_poly1.exterior.coords))

    shrunk_poly2 = shrink_polygon(polygon2)
    shrunk_poly2_verts = np.array(list(shrunk_poly2.exterior.coords))


    def count_verts_inside_poly(polygon: Polygon, query_verts: np.ndarray) -> int:
        """
        Args:
        query vertices
        """
        num_violations = 0
        for vert in query_verts:
            v_pt = Point(vert)
            if polygon.contains(v_pt) or polygon.contains(v_pt):
                num_violations += 1
        return num_violations

    # TODO: interpolate evenly spaced points along edges, if this gives any benefit?
    # Cannot be the case that poly1 vertices fall within a shrinken version of polygon2
    # also, cannot be the case that poly2 vertices fall within a shrunk version of polygon1
    pano1_violations = count_verts_inside_poly(shrunk_poly1, query_verts=pano2_room_vertices_interp)
    pano2_violations = count_verts_inside_poly(shrunk_poly2, query_verts=pano1_room_vertices_interp)
    num_violations = pano1_violations + pano2_violations

    is_valid = num_violations == 0

    if visualize:
        # plot the overlap region
        plt.close("all")

        # plot the interpolated points via scatter, but keep the original lines
        plt.scatter(pano1_room_vertices_interp[:, 0], pano1_room_vertices_interp[:, 1], 10, color="m")
        plt.plot(pano1_room_vertices[:, 0], pano1_room_vertices[:, 1], color="m", linewidth=20, alpha=0.1)

        plt.scatter(pano2_room_vertices_interp[:, 0], pano2_room_vertices_interp[:, 1], 10, color="g")
        plt.plot(pano2_room_vertices[:, 0], pano2_room_vertices[:, 1], color="g", linewidth=10, alpha=0.1)

        plt.scatter(shrunk_poly1_verts[:, 0], shrunk_poly1_verts[:, 1], 10, color="r")
        plt.plot(shrunk_poly1_verts[:, 0], shrunk_poly1_verts[:, 1], color="r", linewidth=1, alpha=0.1)

        plt.scatter(shrunk_poly2_verts[:, 0], shrunk_poly2_verts[:, 1], 10, color="b")
        plt.plot(shrunk_poly2_verts[:, 0], shrunk_poly2_verts[:, 1], color="b", linewidth=1, alpha=0.1)

        plt.title(f"Step 2: Number of violations: {num_violations}")
        plt.axis("equal")
        # plt.show()

        classification = "invalid" if num_violations > 0 else "valid"

        os.makedirs(f"debug_plots/{classification}", exist_ok=True)
        plt.savefig(f"debug_plots/{classification}/{pano1_id}_{pano2_id}___step2_{i}_{j}.jpg")
        plt.close("all")

    return is_valid


def test_determine_invalid_wall_overlap1() -> None:
    """large horseshoe, and small horseshoe

    .---.---.
    |       |
    .   .xxx.
    |       x|
    .   .xxx.
    |       |
    .       .

    """
    # fmt: off
    pano1_room_vertices = np.array(
        [
            [1,2],
            [1,5],
            [3,5],
            [3,2]
        ])
    pano2_room_vertices = np.array(
        [
            [2,4],
            [3,4],
            [3,3],
            [2,3]
        ])

    # fmt: on
    wall_buffer_m = 0.2  # 20 centimeter noise buffer
    allowed_overlap_pct = 0.01  # TODO: allow 1% of violations ???

    is_valid = determine_invalid_wall_overlap(pano1_room_vertices, pano2_room_vertices, wall_buffer_m)
    assert not is_valid


def test_determine_invalid_wall_overlap2() -> None:
    """identical shape

    .---.---.
    |       |
    .       .
    |       |
    .       .
    |       |
    .       .

    """
    # fmt: off
    pano1_room_vertices = np.array(
        [
            [1,2],
            [1,5],
            [3,5],
            [3,2]
        ])
    pano2_room_vertices = np.array(
        [
            [1,2],
            [1,5],
            [3,5],
            [3,2]
        ])

    # fmt: on
    wall_buffer_m = 0.2  # 20 centimeter noise buffer
    allowed_overlap_pct = 0.01  # TODO: allow 1% of violations ???

    is_valid = determine_invalid_wall_overlap(pano1_room_vertices, pano2_room_vertices, wall_buffer_m)
    assert is_valid


def plot_room_walls(
    pano_obj: PanoData, i2Ti1: Optional[Sim2] = None, linewidth: float = 2.0, alpha: float = 0.5
) -> None:
    """ """

    room_vertices = pano_obj.room_vertices_local_2d
    if i2Ti1:
        room_vertices = i2Ti1.transform_from(room_vertices)

    color = np.random.rand(3)
    plt.scatter(room_vertices[:, 0], room_vertices[:, 1], 10, marker=".", color=color, alpha=alpha)
    plt.plot(room_vertices[:, 0], room_vertices[:, 1], color=color, alpha=alpha, linewidth=linewidth)
    # draw edge to close each polygon
    last_vert = room_vertices[-1]
    first_vert = room_vertices[0]
    plt.plot(
        [last_vert[0], first_vert[0]], [last_vert[1], first_vert[1]], color=color, alpha=alpha, linewidth=linewidth
    )


def get_all_pano_wd_vertices(pano_obj: PanoData) -> np.ndarray:
    """

    Returns:
        pts: array of shape (N,3)
    """
    pts = np.zeros((0, 3))

    for wd in pano_obj.windows + pano_obj.doors + pano_obj.openings:
        wd_pts = wd.polygon_vertices_local_3d

        pts = np.vstack([pts, wd_pts])

    return pts


# def sample_points_along_bbox_boundary(wdo: WDO) -> np.ndarray:
#     """ """

#     from argoverse.utils.interpolate import interp_arc
#     from argoverse.utils.polyline_density import get_polyline_length

#     x1,y1 = wdo.local_vertices_3d

#     num_interp_pts = int(query_l2 * NUM_PTS_PER_TRAJ / ref_l2)
#     dense_interp_polyline = interp_arc(num_interp_pts, polyline_to_interp[:, 0], polyline_to_interp[:, 1])

#     return pts


# def test_sample_points_along_bbox_boundary() -> None:
#     """ """
#     global_SIM2_local = Sim2(R=np.eye(2), t=np.zeros(2), s=1.0)

#     wdo = WDO(
#         global_SIM2_local= global_SIM2_local,
#         pt1= (x1,y1),
#         pt2= (x2,y2),
#         bottom_z=0,
#         top_z=10,
#         type="door"
#     )

#     pts = sample_points_along_bbox_boundary(wdo)


def test_reflections_1() -> None:
    """
    Compose does not work properly for chained reflection and rotation.
    """

    pts_local = np.array([[2, 1], [1, 1], [1, 2], [1, 3], [2, 3], [2, 2], [1, 2]])

    plt.scatter(pts_local[:, 0], pts_local[:, 1], 10, color="r", marker=".")
    plt.plot(pts_local[:, 0], pts_local[:, 1], color="r")

    R = rotmat2d(45)  # 45 degree rotation
    t = np.array([1, 1])
    s = 2.0
    world_Sim2_local = Sim2(R, t, s)

    pts_world = world_Sim2_local.transform_from(pts_local)

    plt.scatter(pts_world[:, 0], pts_world[:, 1], 10, color="b", marker=".")
    plt.plot(pts_world[:, 0], pts_world[:, 1], color="b")

    plt.scatter(pts_world[:, 0], pts_world[:, 1], 10, color="b", marker=".")
    plt.plot(pts_world[:, 0], pts_world[:, 1], color="b")

    R_refl = np.array([[-1.0, 0], [0, 1]])
    reflectedworld_Sim2_world = Sim2(R_refl, t=np.zeros(2), s=1.0)

    # import pdb; pdb.set_trace()
    pts_reflworld = reflectedworld_Sim2_world.transform_from(pts_world)

    plt.scatter(pts_reflworld[:, 0], pts_reflworld[:, 1], 100, color="g", marker=".")
    plt.plot(pts_reflworld[:, 0], pts_reflworld[:, 1], color="g", alpha=0.3)

    plt.scatter(-pts_world[:, 0], pts_world[:, 1], 10, color="m", marker=".")
    plt.plot(-pts_world[:, 0], pts_world[:, 1], color="m", alpha=0.3)

    plt.axis("equal")
    plt.show()


def test_reflections_2() -> None:
    """
    Try reflection -> rotation -> compare relative poses

    rotation -> reflection -> compare relative poses

    Relative pose is identical, but configuration will be different in the absolute world frame
    """
    pts_local = np.array([[2, 1], [1, 1], [1, 2], [1, 3], [2, 3], [2, 2], [1, 2]])

    R_refl = np.array([[-1.0, 0], [0, 1]])
    identity_Sim2_reflected = Sim2(R_refl, t=np.zeros(2), s=1.0)

    # pts_refl = identity_Sim2_reflected.transform_from(pts_local)
    pts_refl = pts_local

    R = rotmat2d(45)  # 45 degree rotation
    t = np.array([1, 1])
    s = 1.0
    world_Sim2_i1 = Sim2(R, t, s)

    R = rotmat2d(45)  # 45 degree rotation
    t = np.array([1, 2])
    s = 1.0
    world_Sim2_i2 = Sim2(R, t, s)

    pts_i1 = world_Sim2_i1.transform_from(pts_refl)
    pts_i2 = world_Sim2_i2.transform_from(pts_refl)

    pts_i1 = identity_Sim2_reflected.transform_from(pts_i1)
    pts_i2 = identity_Sim2_reflected.transform_from(pts_i2)

    plt.scatter(pts_i1[:, 0], pts_i1[:, 1], 10, color="b", marker=".")
    plt.plot(pts_i1[:, 0], pts_i1[:, 1], color="b")

    plt.scatter(pts_i2[:, 0], pts_i2[:, 1], 10, color="g", marker=".")
    plt.plot(pts_i2[:, 0], pts_i2[:, 1], color="g")

    plt.axis("equal")
    plt.show()


def export_alignment_hypotheses_to_json(num_processes: int, raw_dataset_dir: str, hypotheses_save_root: str) -> None:
    """
    Questions: what is tour_data_mapping.json? -> for internal people, GUIDs to production people
    Last edge of polygon (to close it) is not provided -- right??
    are all polygons closed? or just polylines?
    What is 'scale_meters_per_coordinate'?

    What is merger vs. redraw?

    Sim(2)

    s(Rp + t)  -> Sim(2)
    sRp + t -> ICP (Zillow)

    sRp + st

    Maybe compose with other Sim(2)
    """
    building_ids = [Path(fpath).stem for fpath in glob.glob(f"{raw_dataset_dir}/*") if Path(fpath).is_dir()]
    building_ids.sort()

    args = []

    for building_id in building_ids:
        json_annot_fpath = f"{raw_dataset_dir}/{building_id}/zfm_data.json"
        pano_dir = f"{raw_dataset_dir}/{building_id}/panos"
        # render_building(building_id, pano_dir, json_annot_fpath)

        args += [(hypotheses_save_root, building_id, pano_dir, json_annot_fpath)]

    if num_processes > 1:
        with Pool(num_processes) as p:
            p.starmap(align_by_wdo, args)
    else:
        for single_call_args in args:
            align_by_wdo(*single_call_args)


if __name__ == "__main__":
    """ """
    # teaser file
    raw_dataset_dir = "/Users/johnlam/Downloads/2021_05_28_Will_amazon_raw"
    #raw_dataset_dir = "/Users/johnlam/Downloads/ZInD_release/complete_zind_paper_final_localized_json_6_3_21"
    #raw_dataset_dir = "/mnt/data/johnlam/ZInD_release/complete_zind_paper_final_localized_json_6_3_21"

    # hypotheses_save_root = "/Users/johnlam/Downloads/jlambert-auto-floorplan/verifier_dataset_2021_06_21"
    #hypotheses_save_root = "/mnt/data/johnlam/ZinD_alignment_hypotheses_2021_06_25"
    # hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_alignment_hypotheses_2021_06_25"
    hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_alignment_hypotheses_2021_07_07"

    num_processes = 1

    export_alignment_hypotheses_to_json(num_processes, raw_dataset_dir, hypotheses_save_root)

    # test_shrink_polygon()

    # test_interp_evenly_spaced_points()
    #test_interp_arc()
    # test_eliminate_duplicates_2d()

    # test_reflections_2()
    # test_determine_invalid_wall_overlap1()
    # test_determine_invalid_wall_overlap2()
    # test_get_wd_normal_2d()
    # test_get_relative_angle()
    # test_align_rooms_by_wd()
    # test_prune_to_unique_sim2_objs()
