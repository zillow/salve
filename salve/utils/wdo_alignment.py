"""Utility to generate potential relative poses by exhaustive pairwise W/D/O alignments.

The alignment generation method here is based upon Cohen et al. 2016.

Window-Window correspondences must be established. Have to find all possible pairwise choices. ICP is not required,
since we can know correspondences between W/D/O vertices.

Cohen16: A single match between an indoor and outdoor window determines an alignment hypothesis
Computing the alignment boils down to finding a  transformation between the models.

Can perform alignment in 2d or 3d (fewer unknowns in 2d).
Note: when fitting Sim(3), note that if the heights are the same, but door width scale is different, perfect
door-width alignment cannot be possible, since the height figures into the least squares problem.

We make sure wide door cannot fit inside a narrow door (e.g. 2x width not allowed).

What heuristic tells us if they should be identity or mirrored in configuration?
Are there any new WDs that should not be visible? walls should not cross on top of each other? know same-room
connections, first.

Cannot expect a match for each door or window. Find nearest neighbor -- but then choose min dist on rows or cols?
may not be visible?
- Rooms can be joined at windows.
- Rooms may be joined at a door.
- Predicted wall cannot lie behind another wall, if line of sight is clear.
"""

import os
from enum import Enum
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import salve.utils.overlap_utils as overlap_utils
import salve.utils.rotation_utils as rotation_utils
import salve.utils.se2_estimation as se2_estimation
import salve.utils.sim3_estimation as sim3_estimation
from salve.common.alignment_hypothesis import AlignmentHypothesis
from salve.common.pano_data import WDO, PanoData
from salve.common.sim2 import Sim2

# Width ratio, defined as (smaller width) / (larger width), must be greater than 0.65 / 1.0 for inferred data.
MIN_ALLOWED_INFERRED_WDO_WIDTH_RATIO = 0.65
MIN_ALLOWED_GT_WDO_WIDTH_RATIO = 0.8


# Could increase to 10. I found 9.5 to 12 degrees let in false positives.
OPENING_ALIGNMENT_ANGLE_TOLERANCE = 9.0
DOOR_WINDOW_ALIGNMENT_ANGLE_TOLERANCE = 7.0  # set to 5.0 for GT
ALIGNMENT_TRANSLATION_TOLERANCE = 0.35  # was set to 0.2 for GT

DEFAULT_OVERLAP_CHECK_SHRINK_FACTOR = 0.1


class AlignTransformType(str, Enum):
    """Represents the type of alignment transformation used/fitted between two panoramas.

    Sim(3) represents a silarity transformation between the models, which can be computed from three point
    correspondences in the general case and from two point matches if the gravity direction is known.
    """

    SE2: str = "SE2"
    Sim3: str = "Sim3"


def get_all_pano_wd_vertices(pano_obj: PanoData) -> np.ndarray:
    """Return 3d coordinates in panorama's local frame of all W/D/O vertices.

    Args:
        pano_obj: input object containing information relevant to a single panorama.

    Returns:
        pts: array of shape (N,3)
        TODO: clarify coordinate frame used in return type.
    """
    pts = np.zeros((0, 3))

    for wd in pano_obj.windows + pano_obj.doors + pano_obj.openings:
        wd_pts = wd.polygon_vertices_local_3d

        pts = np.vstack([pts, wd_pts])

    return pts


def _plot_room_walls(
    pano_obj: PanoData, i2Ti1: Optional[Sim2] = None, color=None, linewidth: float = 2.0, alpha: float = 0.5
) -> None:
    """Plot a pano's layout (representing room boundary), and optionally transform to another frame."""
    room_vertices = pano_obj.room_vertices_local_2d
    if i2Ti1:
        room_vertices = i2Ti1.transform_from(room_vertices)

    if color is None:
        color = np.random.rand(3)
    plt.scatter(room_vertices[:, 0], room_vertices[:, 1], 10, marker=".", color=color, alpha=alpha)
    plt.plot(room_vertices[:, 0], room_vertices[:, 1], color=color, alpha=alpha, linewidth=linewidth)
    # Draw edge to close each polygon
    last_vert = room_vertices[-1]
    first_vert = room_vertices[0]
    plt.plot(
        [last_vert[0], first_vert[0]], [last_vert[1], first_vert[1]], color=color, alpha=alpha, linewidth=linewidth
    )


def align_rooms_by_wd(
    pano1_obj: PanoData,
    pano2_obj: PanoData,
    transform_type: AlignTransformType,
    use_inferred_wdos_layout: bool,
    visualize: bool = False,
    verbose: bool = False,
) -> Tuple[List[AlignmentHypothesis], int]:
    """Compute alignment between two panoramas by computing the transformation between W/D/O's of identical category.

    These categories include a window-window, door-door, or opening-opening pair of objects.

    If inferred W/D/O + inferred layout, only compare W/D/O width ratios for pruning.
    If GT W/D/O + GT layout, also compare wall overlap / freespace penetration for pruning.

    Args:
        pano1_obj: input panorama object 1.
        pano2_obj: input panorama object 2.
        transform_type: transformation object to fit, e.g.  Sim(3) or SE(2).
        use_inferred_wdos_layout: whether to use looser alignment requirements, because input pano objects include
            W/D/O's and room layout that are inferred by a noisy model (instead of using GT annotated W/D/Os + layout).
        visualize: whether to save visualizations for each putative pair.
        verbose: whether to log messages about execution progress.

    Returns:
        possible_alignment_info: list of tuples (i2Ti1, alignment_object) where i2Ti1 is an alignment transformation
        num_invalid_configurations: number of alignment configurations that were rejected, because of freespace
            penetration by aligned walls.
    """
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

        # Try every possible pairwise combination, for this object type.
        for i, pano1_wd in enumerate(pano1_wds):
            pano1_wd_pts = pano1_wd.polygon_vertices_local_3d

            for j, pano2_wd in enumerate(pano2_wds):

                if alignment_object in ["door", "opening"]:
                    plausible_configurations = ["identity", "rotated"]
                elif alignment_object == "window":
                    plausible_configurations = ["identity"]

                for configuration in plausible_configurations:
                    # if verbose:
                    #     logger.debug(f"\t{alignment_object} {i}/{j} {configuration}")

                    if configuration == "rotated":
                        pano2_wd_ = pano2_wd.get_rotated_version()
                    else:
                        pano2_wd_ = pano2_wd

                    pano2_wd_pts = pano2_wd_.polygon_vertices_local_3d

                    if visualize:
                        _visualize_wdo_before_alignment(
                            pano1_obj=pano1_obj,
                            pano2_obj=pano2_obj,
                            pano1_wd=pano1_wd,
                            pano2_wd=pano2_wd,
                        )

                    if transform_type == AlignTransformType.SE2:
                        i2Ti1, aligned_pts1 = se2_estimation.align_points_SE2(pano2_wd_pts[:, :2], pano1_wd_pts[:, :2])
                    elif transform_type == AlignTransformType.Sim3:
                        i2Ti1, aligned_pts1 = sim3_estimation.align_points_sim3(pano2_wd_pts, pano1_wd_pts)
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
                        # Use width ratio as sole criterion, as overlap isn't reliable anymore with inferred layout.
                        # One could also reason about layouts beyond openings to determine validity (we do not).
                        is_valid, width_ratio = determine_invalid_width_ratio(
                            pano1_wd=pano1_wd, pano2_wd=pano2_wd_, use_inferred_wdos_layout=use_inferred_wdos_layout
                        )
                    else:
                        width_is_valid, width_ratio = determine_invalid_width_ratio(
                            pano1_wd=pano1_wd, pano2_wd=pano2_wd_, use_inferred_wdos_layout=use_inferred_wdos_layout
                        )
                        freespace_is_valid = overlap_utils.determine_invalid_wall_overlap(
                            pano1_room_vertices=pano1_room_vertices,
                            pano2_room_vertices=pano2_room_vertices,
                            shrink_factor=DEFAULT_OVERLAP_CHECK_SHRINK_FACTOR,
                            pano1_id=pano1_id,
                            pano2_id=pano2_id,
                            i=i,
                            j=j,
                            visualize=False,
                        )
                        is_valid = freespace_is_valid and width_is_valid

                    if verbose:
                        # (i1) pink, (i2) orange
                        print(
                            f"Valid? {is_valid} -> Width: {alignment_object} {i} {j} {configuration} "
                            f"-> {width_ratio:.2f}"
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
                        _visualize_aligned_wdo(
                            pano1_obj=pano1_obj,
                            pano2_obj=pano2_obj,
                            i2Ti1=i2Ti1,
                            aligned_pts1=aligned_pts1,
                            aligned_pts2=aligned_pts2,
                            pano2_wd_pts=pano2_wd_pts,
                            pano1_id=pano1_id,
                            pano2_id=pano2_id,
                            is_valid=is_valid,
                            alignment_object=alignment_object,
                            configuration=configuration,
                            classification=classification,
                            i=i,
                            j=j
                        )

    return possible_alignment_info, num_invalid_configurations


def _visualize_wdo_before_alignment(
    pano1_obj: PanoData,
    pano2_obj: PanoData,
    pano1_wd: WDO,
    pano2_wd: WDO) -> None:
    """Save visualization of W/D/O pair and layout before alignment.
    
    Args:
        pano1_obj: data for panorama 1.
        pano2_obj: data for panorama 2.
        pano1_wd: data for W/D/O from pano 1.
        pano2_wd: data for W/D/O from pano 2.
    """
    plt.close("all")

    all_wd_verts_1 = get_all_pano_wd_vertices(pano1_obj)
    all_wd_verts_2 = get_all_pano_wd_vertices(pano2_obj)
    plt.scatter(-all_wd_verts_1[:,0], all_wd_verts_1[:,1], 10, color='k', marker='+')
    plt.scatter(-all_wd_verts_2[:,0], all_wd_verts_2[:,1], 10, color='g', marker='+')

    _plot_room_walls(pano1_obj)
    _plot_room_walls(pano2_obj)

    plt.plot(
        -pano1_wd.polygon_vertices_local_3d[:,0],
        pano1_wd.polygon_vertices_local_3d[:,1],
        color="r",
        linewidth=5,
        alpha=0.2
    )
    plt.plot(
        -pano2_wd_.polygon_vertices_local_3d[:,0],
        pano2_wd_.polygon_vertices_local_3d[:,1],
        color="b",
        linewidth=5,
        alpha=0.2
    )

    plt.axis("equal")
    plt.title("Step 1: Before alignment")
    #os.makedirs(f"debug_plots/{pano1_id}_{pano2_id}", exist_ok=True)
    #plt.savefig(f"debug_plots/{pano1_id}_{pano2_id}/step1_{i}_{j}.jpg")
    plt.show()
    plt.close("all")


def _visualize_aligned_wdo(
    pano1_obj: PanoData,
    pano2_obj: PanoData,
    i2Ti1: Sim2,
    aligned_pts1: np.ndarray,
    aligned_pts2: np.ndarray,
    pano2_wd_pts: np.ndarray,
    pano1_id: int,
    pano2_id: int,
    is_valid: bool,
    alignment_object: str,
    configuration: str,
    classification: str,
    i: int,
    j: int,
) -> None:
    """Save visualization of W/D/O and layout after alignment.

    Args:
        pano1_obj: data for panorama 1.
        pano2_obj: data for panorama 2.
        i2Ti1: Similarity(2) transformation.
        aligned_pts1: 
        aligned_pts2: 
        pano2_wd_pts: 
        pano1_id: panorama 1 index (for ZInD building).
        pano2_id: panorama 2 index (for ZInD building).
        is_valid: whether alignment hypothesis was classified as valid.
        alignment_object: 
        configuration: W/D/O surface normal configuration (identity vs. rotated).
        classification: 
        i: pano 1 W/D/O index among all of pano 1's W/D/Os.
        j: pano 2 W/D/O index among all of pano 2's W/D/Os.
    """
    _plot_room_walls(pano1_obj, i2Ti1, color="tab:pink", linewidth=10)
    _plot_room_walls(pano2_obj, color="tab:orange", linewidth=1)

    plt.scatter(aligned_pts1[:, 0], aligned_pts1[:, 1], 10, color="r", marker="+")
    plt.plot(aligned_pts1[:, 0], aligned_pts1[:, 1], color="r", linewidth=10, alpha=0.5)

    plt.scatter(pano2_wd_pts[:, 0], pano2_wd_pts[:, 1], 10, color="b", marker="+")
    plt.plot(pano2_wd_pts[:, 0], pano2_wd_pts[:, 1], color="g", linewidth=5, alpha=0.1)

    # plt.plot(inter_poly_verts[:,0],inter_poly_verts[:,1], color='m')

    plt.title(
        f"Step 3: Match: ({pano1_id},{pano2_id}): valid={is_valid},"
        f" aligned via {alignment_object}, \n  config={configuration}"
    )
    # window_normals_compatible={window_normals_compatible},
    plt.axis("equal")
    os.makedirs(f"debug_plots/{classification}", exist_ok=True)
    plt.savefig(
        f"debug_plots/{classification}/"
        f"{alignment_object}_{pano1_id}_{pano2_id}___step3_{configuration}_{i}_{j}.jpg"
    )

    # plt.show()
    plt.close("all")


def determine_invalid_width_ratio(pano1_wd: WDO, pano2_wd: WDO, use_inferred_wdos_layout: bool) -> Tuple[bool, float]:
    """Check to see if relative width ratio of W/D/Os is within some maximum allowed range, e.g. [0.65, 1]

    Args:
        pano1_wd: W/D/O object for panorama 1.
        pano2_wd: W/D/O object for panorama 2.
        use_inferred_wdos_layout: whether to use looser requirements (GT should be closer to true width.)

    Returns:
        is_valid: whether the match is plausible, given the relative W/D/O widths.
        width_ratio: w_min/w_max, where w_min=min(w1,w2) and w_max=max(w1,w2)
    """
    min_width = min(pano1_wd.width, pano2_wd.width)
    max_width = max(pano1_wd.width, pano2_wd.width)
    width_ratio = min_width / max_width

    min_allowed_wdo_width_ratio = (
        MIN_ALLOWED_INFERRED_WDO_WIDTH_RATIO if use_inferred_wdos_layout else MIN_ALLOWED_GT_WDO_WIDTH_RATIO
    )

    is_valid = width_ratio >= min_allowed_wdo_width_ratio  # should be in [0.65, 1.0] for inferred WDO
    return is_valid, width_ratio


def obj_almost_equal(i2Ti1: Sim2, i2Ti1_: Sim2, wdo_alignment_object: str) -> bool:
    """Check if two rigid body transformations are equal up to a tolerance, depending on object category.

    Args:
        i2Ti1: first relative pose.
        i2Ti1_: second relative pose.
        wdo_alignment_object: type of W/D/O (either door, window, or opening)

    Returns:
        boolean indicating whether the two input transformations are equal up to a tolerance.
    """
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
