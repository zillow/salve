"""
Based on code found at
https://gitlab.zgtools.net/zillow/rmx/research/scripts-insights/open_platform_utils/-/raw/zind_cleanup/zfm360/visualize_zfm360_data.py

Data can be found at:
/mnt/data/zhiqiangw/ZInD_release/complete_zind_paper_final_localized_json_6_3_21

conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch

No shared texture between (0,75) -- yet doors align it (garage to kitchen)
"""

import glob
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import argoverse.utils.json_utils as json_utils
import matplotlib.pyplot as plt
import numpy as np
from argoverse.utils.sim2 import Sim2
from shapely.geometry import LineString

import afp.dataset.hnet_prediction_loader as hnet_prediction_loader
import afp.utils.logger_utils as logger_utils
import afp.utils.overlap_utils as overlap_utils
import afp.utils.rotation_utils as rotation_utils
import afp.utils.sim3_align_dw as sim3_align_dw  # TODO: rename module to more informative name
from afp.common.pano_data import FloorData, PanoData, WDO


# See https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal
HEADER = "\033[95m"
OKGREEN = "\033[92m"

# could increase to 10. I found 9.5 to 12 degrees let in false positives.
OPENING_ALIGNMENT_ANGLE_TOLERANCE = 9.0
DOOR_WINDOW_ALIGNMENT_ANGLE_TOLERANCE = 7.0  # set to 5.0 for GT
ALIGNMENT_TRANSLATION_TOLERANCE = 0.35  # was set to 0.2 for GT

# (smaller width) / (larger width) must be greater than 0.65 / 1.0.
MIN_ALLOWED_WDO_WIDTH_RATIO = 0.65


logger = logger_utils.get_logger()

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


class AlignmentHypothesis(NamedTuple):
    """ """

    i2Ti1: Sim2
    wdo_alignment_object: str  # either 'door', 'window', or 'opening'
    i1_wdo_idx: int  # this is the WDO index for Pano i1 (known as i)
    i2_wdo_idx: int  # this is the WDO index for Pano i2 (known as j)
    configuration: str  # either identity or rotated


@dataclass
class AlignmentGenerationReport:
    floor_alignment_infeasibility_dict = Dict[str, Tuple[int, int]]


# multiply all x-coordinates or y-coordinates by -1, to transfer origin from upper-left, to bottom-left
# (Reflection about either axis, would need additional rotation if reflect over x-axis)


def are_visibly_adjacent(pano1_obj: PanoData, pano2_obj: PanoData) -> bool:
    """ """
    DIST_THRESH = 0.1
    # do they share a door or window?

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


def obj_almost_equal(i2Ti1: Sim2, i2Ti1_: Sim2, wdo_alignment_object: str) -> bool:
    """ """
    angle1 = i2Ti1.theta_deg
    angle2 = i2Ti1_.theta_deg

    # print(f"\t\tTrans: {i2Ti1.translation} vs. {i2Ti1_.translation}")
    # print(f"\t\tScale: {i2Ti1.scale:.1f} vs. {i2Ti1_.scale:.1f}")
    # print(f"\t\tAngle: {angle1:.1f} vs. {angle2:.1f}")

    if not np.allclose(i2Ti1.translation, i2Ti1_.translation, atol=ALIGNMENT_TRANSLATION_TOLERANCE):
        return False

    if not np.isclose(i2Ti1.scale, i2Ti1_.scale, atol=0.35):
        return False

    if wdo_alignment_object in ["door", "window"]:
        alignment_angle_tolerance = DOOR_WINDOW_ALIGNMENT_ANGLE_TOLERANCE

    elif wdo_alignment_object == "opening":
        alignment_angle_tolerance = OPENING_ALIGNMENT_ANGLE_TOLERANCE

    else:
        raise RuntimeError

    if not rotation_utils.angle_is_equal(angle1, angle2, atol=alignment_angle_tolerance):
        return False

    return True


def test_obj_almost_equal() -> None:
    """ """
    # fmt: off
    i2Ti1_pred = Sim2(
        R=np.array(
            [
                [-0.99928814, 0.03772511],
                [-0.03772511, -0.99928814]
            ], dtype=np.float32
        ),
        t=np.array([-3.0711207, -0.5683456], dtype=np.float32),
        s=1.0,
    )

    i2Ti1_gt = Sim2(
        R=np.array(
            [
                [-0.9999569, -0.00928213],
                [0.00928213, -0.9999569]
            ], dtype=np.float32
        ),
        t=np.array([-3.0890038, -0.5540818], dtype=np.float32),
        s=0.9999999999999999,
    )
    # fmt: on
    assert obj_almost_equal(i2Ti1_pred, i2Ti1_gt)
    assert obj_almost_equal(i2Ti1_gt, i2Ti1_pred)


def prune_to_unique_sim2_objs(possible_alignment_info: List[AlignmentHypothesis]) -> List[AlignmentHypothesis]:
    """
    Only useful for GT objects, that might have exact equality? (confirm how GT can actually have exact values)
    """
    pruned_possible_alignment_info = []

    for j, alignment_hypothesis in enumerate(possible_alignment_info):
        is_dup = any(
            [
                alignment_hypothesis.i2Ti1 == inserted_alignment_hypothesis.i2Ti1
                for inserted_alignment_hypothesis in pruned_possible_alignment_info
            ]
        )
        # has not been used yet
        if not is_dup:
            pruned_possible_alignment_info.append(alignment_hypothesis)

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
        AlignmentHypothesis(
            i2Ti1=Sim2(wR1, wt1, ws1),
            wdo_alignment_object="window",
            i1_wdo_idx=1,
            i2_wdo_idx=5,
            configuration="identity",
        ),
        AlignmentHypothesis(
            i2Ti1=Sim2(wR1, wt1, ws1),
            wdo_alignment_object="window",
            i1_wdo_idx=2,
            i2_wdo_idx=6,
            configuration="identity",
        ),
        AlignmentHypothesis(
            i2Ti1=Sim2(wR2, wt2, ws2),
            wdo_alignment_object="window",
            i1_wdo_idx=3,
            i2_wdo_idx=7,
            configuration="identity",
        ),
        AlignmentHypothesis(
            i2Ti1=Sim2(wR1, wt1, ws1),
            wdo_alignment_object="window",
            i1_wdo_idx=4,
            i2_wdo_idx=8,
            configuration="identity",
        ),
    ]
    pruned_possible_alignment_info = prune_to_unique_sim2_objs(possible_alignment_info)
    assert len(pruned_possible_alignment_info) == 2

    assert pruned_possible_alignment_info[0].i2Ti1.scale == 1.5
    assert pruned_possible_alignment_info[1].i2Ti1.scale == 3.0


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
    json_utils.save_json_dict(save_fpath, dict_for_serialization)


# def test_align_rooms_by_wd() -> None:
#     """ """
#     wTi5 = Sim2(
#         R=np.array([[0.999897, -0.01435102], [0.01435102, 0.999897]], dtype=np.float32),
#         t=np.array([0.7860708, -1.57248], dtype=np.float32),
#         s=0.4042260417272217,
#     )
#     wTi8 = Sim2(
#         R=np.array([[0.02998102, -0.99955046], [0.99955046, 0.02998102]], dtype=np.float32),
#         t=np.array([0.91035557, -3.2141], dtype=np.float32),
#         s=0.4042260417272217,
#     )

#     # fmt: off
#     pano1_obj = PanoData(
#         id=5,
#         global_Sim2_local=wTi5,
#         room_vertices_local_2d=np.array(
#             [
#                 [ 1.46363621, -2.43808616],
#                 [ 1.3643741 ,  0.5424695 ],
#                 [ 0.73380685,  0.52146958],
#                 [ 0.7149462 ,  1.08780075],
#                 [ 0.4670652 ,  1.07954551],
#                 [ 0.46914653,  1.01704912],
#                 [-1.2252865 ,  0.96061904],
#                 [-1.10924507, -2.5237714 ]
#             ]),
#         image_path='panos/floor_01_partial_room_05_pano_5.jpg',
#         label='living room',
#         doors=[],
#         windows=[
#             WDO(
#                 global_Sim2_local=wTi5,
#                 pt1=[-1.0367953294361147, -2.5213585867749635],
#                 pt2=[-0.4661345615720372, -2.5023537435761822],
#                 bottom_z=-0.5746298535133153,
#                 top_z=0.38684337323286566,
#                 type='windows'
#             ),
#             WDO(
#                 global_Sim2_local=wTi5,
#                 pt1=[0.823799786466513, -2.45939477144822],
#                 pt2=[1.404932996095547, -2.4400411621788427],
#                 bottom_z=-0.5885416433689703,
#                 top_z=0.3591070365687572,
#                 type='windows'
#             )
#         ],
#         openings=[]
#     )


#     pano2_obj = PanoData(
#         id=8,
#         global_Sim2_local=wTi8,
#         room_vertices_local_2d=np.array(
#             [
#                 [-0.7336625 , -1.3968136 ],
#                 [ 2.23956454, -1.16554334],
#                 [ 2.19063694, -0.53652654],
#                 [ 2.75557561, -0.4925832 ],
#                 [ 2.73634178, -0.2453117 ],
#                 [ 2.67399906, -0.25016098],
#                 [ 2.54252291,  1.44010577],
#                 [-0.93330008,  1.16974146]
#             ]),
#         image_path='panos/floor_01_partial_room_05_pano_8.jpg',
#         label='living room',
#         doors=[],
#         windows=[
#             WDO(
#                 global_Sim2_local=wTi8,
#                 pt1=[-0.9276784906829552, 1.0974698581331057],
#                 pt2=[-0.8833992085857922, 0.5282122352406332],
#                 bottom_z=-0.5746298535133153,
#                 top_z=0.38684337323286566,
#                 type='windows'
#             ),
#             WDO(
#                 global_Sim2_local=wTi8,
#                 pt1=[-0.7833093301499523, -0.758550412558342],
#                 pt2=[-0.7382174598580689, -1.338254727497497],
#                 bottom_z=-0.5885416433689703,
#                 top_z=0.3591070365687572,
#                 type='windows'
#             )
#         ],
#         openings=[]
#     )

#     # fmt: on
#     possible_alignment_info, _ = align_rooms_by_wd(pano1_obj, pano2_obj)
#     assert len(possible_alignment_info) == 3


def plot_room_walls(
    pano_obj: PanoData, i2Ti1: Optional[Sim2] = None, color=None, linewidth: float = 2.0, alpha: float = 0.5
) -> None:
    """ """
    room_vertices = pano_obj.room_vertices_local_2d
    if i2Ti1:
        room_vertices = i2Ti1.transform_from(room_vertices)

    if color is None:
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


def align_rooms_by_wd(
    pano1_obj: PanoData,
    pano2_obj: PanoData,
    transform_type: str = "SE2",
    use_inferred_wdos_layout: bool = True,
    visualize: bool = False,
) -> Tuple[List[AlignmentHypothesis], int]:
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
        transform_type: transformation object to fit, e.g.  Sim(3) or SE(2), "Sim3" or "SE2"
        visualize: whether to save visualizations for each putative pair.

    Returns:
        possible_alignment_info: list of tuples (i2Ti1, alignment_object) where i2Ti1 is an alignment transformation
        num_invalid_configurations: number of alignment configurations that were rejected, because of freespace penetration by aligned walls.
    """
    verbose = False

    pano1_id = pano1_obj.id
    pano2_id = pano2_obj.id

    num_invalid_configurations = 0
    possible_alignment_info = []

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
                    # if verbose:
                    #     logger.debug(f"\t{alignment_object} {i}/{j} {configuration}")

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

                    # import pdb; pdb.set_trace()

                    if transform_type == "SE2":
                        i2Ti1, aligned_pts1 = sim3_align_dw.align_points_SE2(pano2_wd_pts[:, :2], pano1_wd_pts[:, :2])
                    elif transform_type == "Sim3":
                        i2Ti1, aligned_pts1 = sim3_align_dw.align_points_sim3(pano2_wd_pts, pano1_wd_pts)
                    else:
                        raise RuntimeError

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

                    if use_inferred_wdos_layout:
                        # overlap isn't reliable anymore?
                        # TODO: write new code that considers whether we are beyond an opening? in which case invalid?

                        # Check to see if relative width ratio is within the range [0.5, 2]
                        min_width = min(pano1_wd.width, pano2_wd_.width)
                        max_width = max(pano1_wd.width, pano2_wd_.width)
                        width_ratio = min_width / max_width

                        # pano1_uncertainty_factor = uncertainty_utils.compute_width_uncertainty(pano1_wd)
                        # pano2_uncertainty_factor = uncertainty_utils.compute_width_uncertainty(pano2_wd)

                        is_valid = width_ratio >= MIN_ALLOWED_WDO_WIDTH_RATIO  # should be in [0.5, 1.0]

                        if verbose:
                            # (i1) pink, (i2) orange
                            print(
                                f"Valid? {is_valid} -> Width: {alignment_object} {i} {j} {configuration} -> {width_ratio:.2f}"
                                + ""  # f", Uncertainty: {pano1_uncertainty_factor:.2f}, {pano2_uncertainty_factor:.2f}"
                            )

                    else:
                        is_valid = overlap_utils.determine_invalid_wall_overlap(
                            pano1_id,
                            pano2_id,
                            i,
                            j,
                            pano1_room_vertices,
                            pano2_room_vertices,
                            shrink_factor=0.1,
                            visualize=False,
                        )
                    # logger.error("Pano1 room verts: %s", str(pano1_room_vertices))
                    # logger.error("Pano2 room verts: %s", str(pano2_room_vertices))

                    if is_valid:
                        possible_alignment_info.append(
                            AlignmentHypothesis(
                                i2Ti1=i2Ti1,
                                wdo_alignment_object=alignment_object,
                                i1_wdo_idx=i,
                                i2_wdo_idx=j,
                                configuration=configuration,
                            )
                        )
                        classification = "valid"
                    else:
                        num_invalid_configurations += 1
                        classification = "invalid"

                    if visualize:
                        plot_room_walls(pano1_obj, i2Ti1, color="tab:pink", linewidth=10)
                        plot_room_walls(pano2_obj, color="tab:orange", linewidth=1)

                        plt.scatter(aligned_pts1[:, 0], aligned_pts1[:, 1], 10, color="r", marker="+")
                        plt.plot(aligned_pts1[:, 0], aligned_pts1[:, 1], color="r", linewidth=10, alpha=0.5)

                        plt.scatter(pano2_wd_pts[:, 0], pano2_wd_pts[:, 1], 10, color="b", marker="+")
                        plt.plot(pano2_wd_pts[:, 0], pano2_wd_pts[:, 1], color="g", linewidth=5, alpha=0.1)

                        # plt.plot(inter_poly_verts[:,0],inter_poly_verts[:,1], color='m')

                        plt.title(
                            f"Step 3: Match: ({pano1_id},{pano2_id}): valid={is_valid}, aligned via {alignment_object}, \n  config={configuration}"
                        )
                        # window_normals_compatible={window_normals_compatible},
                        plt.axis("equal")
                        os.makedirs(f"debug_plots/{classification}", exist_ok=True)
                        plt.savefig(
                            f"debug_plots/{classification}/{alignment_object}_{pano1_id}_{pano2_id}___step3_{configuration}_{i}_{j}.jpg"
                        )

                        # plt.show()
                        plt.close("all")

    return possible_alignment_info, num_invalid_configurations


def export_single_building_wdo_alignment_hypotheses(
    hypotheses_save_root: str,
    building_id: str,
    pano_dir: str,
    json_annot_fpath: str,
    raw_dataset_dir: str,
    use_inferred_wdos_layout: bool,
) -> None:
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
        use_inferred_wdos_layout: whether to use inferred W/D/O + inferred layout (or instead to use GT).
    """
    verbose = False

    if use_inferred_wdos_layout:
        floor_pose_graphs = hnet_prediction_loader.load_inferred_floor_pose_graphs(
            query_building_id=building_id, raw_dataset_dir=raw_dataset_dir
        )
        if floor_pose_graphs is None:
            # cannot compute putative alignments if prediction files are missing.
            return

    floor_map_json = json_utils.read_json_file(json_annot_fpath)

    if "merger" not in floor_map_json:
        logger.error(f"Building {building_id} does not have `merger` data, skipping...")
        return

    merger_data = floor_map_json["merger"]

    floor_gt_is_valid_report_dict = defaultdict(list)

    floor_dominant_rotation = {}
    for floor_id, floor_data in merger_data.items():

        logger.info("--------------------------------")
        logger.info("--------------------------------")
        logger.info("--------------------------------")
        logger.info(f"On building {building_id}, floor {floor_id}...")
        logger.info("--------------------------------")
        logger.info("--------------------------------")
        logger.info("--------------------------------")

        if use_inferred_wdos_layout:
            pano_dict_inferred = floor_pose_graphs[floor_id].nodes

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

                if (building_id == "0006") and (i1 == 7 or i2 == 7):
                    # annotation error for pano 7?
                    continue

                if i1 % 1000 == 0:
                    logger.info(f"\tOn pano pair ({i1},{i2})")
                # _ = plot_room_layout(pano_dict[i1], coord_frame="local")
                # _ = plot_room_layout(pano_dict[i2], coord_frame="local")

                # we use the GT WDOs to infer this GT label.
                visibly_adjacent = are_visibly_adjacent(pano_dict[i1], pano_dict[i2])

                try:
                    if use_inferred_wdos_layout:
                        possible_alignment_info, num_invalid_configurations = align_rooms_by_wd(
                            pano_dict_inferred[i1],
                            pano_dict_inferred[i2],
                            use_inferred_wdos_layout=use_inferred_wdos_layout,
                        )
                    else:
                        possible_alignment_info, num_invalid_configurations = align_rooms_by_wd(
                            pano_dict[i1], pano_dict[i2], use_inferred_wdos_layout=use_inferred_wdos_layout
                        )
                except Exception:
                    logger.exception("Failure in `align_rooms_by_wd()`, skipping... ")
                    continue

                floor_n_valid_configurations += len(possible_alignment_info)
                floor_n_invalid_configurations += num_invalid_configurations

                # given wTi1, wTi2, then i2Ti1 = i2Tw * wTi1 = i2Ti1
                i2Ti1_gt = pano_dict[i2].global_Sim2_local.inverse().compose(pano_dict[i1].global_Sim2_local)
                gt_fname = f"{hypotheses_save_root}/{building_id}/{floor_id}/gt_alignment_exact/{i1}_{i2}.json"
                if visibly_adjacent:
                    save_Sim2(gt_fname, i2Ti1_gt)
                    expected = i2Ti1_gt.rotation.T @ i2Ti1_gt.rotation
                    # print("Identity? ", np.round(expected, 1))
                    if not np.allclose(expected, np.eye(2), atol=1e-6):
                        import pdb

                        pdb.set_trace()

                # TODO: estimate how often an inferred opening can provide the correct relative pose.
                # TODO: make sure wide door cannot fid narrow door (e.g. 2x width not allowed)

                # remove redundant transformations
                pruned_possible_alignment_info = prune_to_unique_sim2_objs(possible_alignment_info)

                labels = []
                # loop over the alignment hypotheses
                for k, ah in enumerate(pruned_possible_alignment_info):

                    if obj_almost_equal(ah.i2Ti1, i2Ti1_gt, ah.wdo_alignment_object):
                        label = "aligned"
                        save_dir = f"{hypotheses_save_root}/{building_id}/{floor_id}/gt_alignment_approx"
                    else:
                        label = "misaligned"
                        save_dir = f"{hypotheses_save_root}/{building_id}/{floor_id}/incorrect_alignment"
                    labels.append(label)

                    fname = (
                        f"{i1}_{i2}__{ah.wdo_alignment_object}_{ah.i1_wdo_idx}_{ah.i2_wdo_idx}_{ah.configuration}.json"
                    )
                    proposed_fpath = f"{save_dir}/{fname}"
                    save_Sim2(proposed_fpath, ah.i2Ti1)

                    if verbose:
                        # allows debugging of tolerances (compare with plots)
                        print(label, fname)

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
                    # logger.warning(
                    #     f"\tGT invalid for Building {building_id}, Floor {floor_id}: ({i1},{i2}): {i2Ti1_gt} vs. {[i1Ti1 for i1Ti1 in pruned_possible_alignment_info]}"
                    # )
                    pass
                floor_gt_is_valid_report_dict[floor_id] += [GT_valid]

        logger.info(f"floor_n_valid_configurations: {floor_n_valid_configurations}")
        logger.info(f"floor_n_invalid_configurations: {floor_n_invalid_configurations}")

    print(f"{OKGREEN} Building {building_id}: ")
    for floor_id, gt_is_valid_arr in floor_gt_is_valid_report_dict.items():
        print(f"{OKGREEN} {floor_id}: {np.mean(gt_is_valid_arr):.2f} over {len(gt_is_valid_arr)} alignment pairs.")
    print(HEADER)


def export_alignment_hypotheses_to_json(
    num_processes: int, raw_dataset_dir: str, hypotheses_save_root: str, use_inferred_wdos_layout: bool
) -> None:
    """
    Questions: what is tour_data_mapping.json? -> for internal people, GUIDs to production people
    Last edge of polygon (to close it) is not provided -- right??
    are all polygons closed? or just polylines?

    Sim(2)

    s(Rp + t)  -> Sim(2)
    sRp + t -> ICP (Zillow)

    sRp + st

    Maybe compose with other Sim(2)

    Args:
        num_processes
        raw_dataset_dir
        hypotheses_save_root
        use_inferred_wdos_layout: whether to use inferred W/D/O + inferred layout (or instead to use GT).
    """
    building_ids = [Path(fpath).stem for fpath in glob.glob(f"{raw_dataset_dir}/*") if Path(fpath).is_dir()]
    building_ids.sort()

    args = []

    for building_id in building_ids:

        # if building_id in ["0003","0006","0034"]:
        #     continue

        # if building_id not in ['0767', '0712', '0711', '0706', '0613',  '0757']:#, '0654', '0560', '0544']:#, '0654', '0613', '0560', '0757', '0544']: #0246"]: #, "001", "002"]: #'1635']: #, '1584', '1583', '1578', '1530', '1490', '1442', '1626', '1427', '1394']:
        #     continue

        json_annot_fpath = f"{raw_dataset_dir}/{building_id}/zind_data.json"
        pano_dir = f"{raw_dataset_dir}/{building_id}/panos"
        # render_building(building_id, pano_dir, json_annot_fpath)

        args += [
            (hypotheses_save_root, building_id, pano_dir, json_annot_fpath, raw_dataset_dir, use_inferred_wdos_layout)
        ]

    if num_processes > 1:
        with Pool(num_processes) as p:
            p.starmap(export_single_building_wdo_alignment_hypotheses, args)
    else:
        for single_call_args in args:
            export_single_building_wdo_alignment_hypotheses(*single_call_args)


if __name__ == "__main__":
    """ """
    use_inferred_wdos_layout = False

    # teaser file
    # raw_dataset_dir = "/Users/johnlam/Downloads/2021_05_28_Will_amazon_raw"
    # raw_dataset_dir = "/Users/johnlam/Downloads/ZInD_release/complete_zind_paper_final_localized_json_6_3_21"
    # raw_dataset_dir = "/mnt/data/johnlam/ZInD_release/complete_zind_paper_final_localized_json_6_3_21"
    # raw_dataset_dir = "/mnt/data/zhiqiangw/ZInD_final_07_11/complete_07_10_new"
    # raw_dataset_dir = "/mnt/data/johnlam/complete_07_10_new"
    # raw_dataset_dir = "/Users/johnlam/Downloads/complete_07_10_new"

    raw_dataset_dir = "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05"
    ##raw_dataset_dir = "/mnt/data/johnlam/zind_bridgeapi_2021_10_05"
    # raw_dataset_dir = "/home/johnlam/zind_bridgeapi_2021_10_05"

    # hypotheses_save_root = "/Users/johnlam/Downloads/jlambert-auto-floorplan/verifier_dataset_2021_06_21"
    # hypotheses_save_root = "/mnt/data/johnlam/ZinD_alignment_hypotheses_2021_06_25"
    # hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_alignment_hypotheses_2021_06_25"
    # hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_alignment_hypotheses_2021_07_14_v3_w_wdo_idxs"
    # hypotheses_save_root = "/mnt/data/johnlam/ZinD_alignment_hypotheses_2021_07_14_v3_w_wdo_idxs"
    # hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_alignment_hypotheses_2021_07_22_find_missing_alignments"
    # hypotheses_save_root = "/mnt/data/johnlam/ZinD_07_11_alignment_hypotheses_2021_08_04_Sim3"
    # hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_07_11_alignment_hypotheses_2021_08_04_Sim3"
    # hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_07_11_alignment_hypotheses_2021_08_31_SE2"

    # hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_bridge_api_alignment_hypotheses_madori_rmx_v1_2021_10_16_SE2"
    # hypotheses_save_root = "/mnt/data/johnlam/ZinD_bridge_api_alignment_hypotheses_madori_rmx_v1_2021_10_16_SE2"

    # hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_bridge_api_alignment_hypotheses_madori_rmx_v1_2021_10_20_SE2_cosine_uncertainty"
    ##hypotheses_save_root = "/mnt/data/johnlam/ZinD_bridge_api_alignment_hypotheses_madori_rmx_v1_2021_10_17_SE2"
    # hypotheses_save_root = "/home/johnlam/ZinD_bridge_api_alignment_hypotheses_madori_rmx_v1_2021_10_20_SE2_width_thresh0.65"
    hypotheses_save_root = (
        "/Users/johnlam/Downloads/ZinD_bridge_api_alignment_hypotheses_GT_WDO_2021_11_17_SE2_width_thresh0.65"
    )

    num_processes = 1

    export_alignment_hypotheses_to_json(num_processes, raw_dataset_dir, hypotheses_save_root, use_inferred_wdos_layout)

    # test_reflections_2()
    # test_get_relative_angle()
    # test_align_rooms_by_wd()
    # test_prune_to_unique_sim2_objs()
