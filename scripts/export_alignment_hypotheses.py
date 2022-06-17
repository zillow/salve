"""Script to generate all pairwise W/D/O alignment hypotheses, for all of ZInD.

No shared texture between (0,75) -- yet doors align it (garage to kitchen)
"""

import argparse
import glob
import os
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import gtsfm.utils.io as io_utils
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString

import salve.dataset.hnet_prediction_loader as hnet_prediction_loader
import salve.utils.logger_utils as logger_utils
import salve.utils.rotation_utils as rotation_utils
import salve.utils.wdo_alignment as wdo_alignment_utils
from salve.common.pano_data import FloorData, PanoData
from salve.common.sim2 import Sim2
from salve.utils.wdo_alignment import AlignTransformType, AlignmentHypothesis
from salve.common.posegraph2d import REDTEXT, ENDCOLOR

# See https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal
HEADER = "\033[95m"
OKGREEN = "\033[92m"

# could increase to 10. I found 9.5 to 12 degrees let in false positives.
OPENING_ALIGNMENT_ANGLE_TOLERANCE = 9.0
DOOR_WINDOW_ALIGNMENT_ANGLE_TOLERANCE = 7.0  # set to 5.0 for GT
ALIGNMENT_TRANSLATION_TOLERANCE = 0.35  # was set to 0.2 for GT


logger = logger_utils.get_logger()


@dataclass
class AlignmentGenerationReport:
    floor_alignment_infeasibility_dict = Dict[str, Tuple[int, int]]


# multiply all x-coordinates or y-coordinates by -1, to transfer origin from upper-left, to bottom-left
# (Reflection about either axis, would need additional rotation if reflect over x-axis)


def are_visibly_adjacent(pano1_obj: PanoData, pano2_obj: PanoData) -> bool:
    """Check if two rooms share a door, window, or opening.

    Args:
        pano1_obj: Room information for pano 1
        pano2_obj: Room information for pano 2
    """
    DIST_THRESH = 0.1

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
    """Check if two rigid body transformations are equal up to a tolerance.

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
    io_utils.save_json_file(json_fpath=save_fpath, data=dict_for_serialization)


def export_single_building_wdo_alignment_hypotheses(
    hypotheses_save_root: str,
    building_id: str,
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
        json_annot_fpath: path to GT data for this building (contained in "zind_data.json")
        raw_dataset_dir: path to ZinD dataset.
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

    floor_map_json = io_utils.read_json_file(json_annot_fpath)

    if "merger" not in floor_map_json:
        logger.error(f"Building {building_id} does not have `merger` data, skipping...")
        return

    merger_data = floor_map_json["merger"]

    floor_gt_is_valid_report_dict = defaultdict(list)

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
                    # annotation error for pano 7, in ZinD.
                    continue

                if i1 % 1000 == 0:
                    logger.info(f"\tOn pano pair ({i1},{i2})")
                # _ = plot_room_layout(pano_dict[i1], coord_frame="local")
                # _ = plot_room_layout(pano_dict[i2], coord_frame="local")

                # we use the GT WDOs to infer this GT label.
                visibly_adjacent = are_visibly_adjacent(pano_dict[i1], pano_dict[i2])

                try:
                    if use_inferred_wdos_layout:
                        possible_alignment_info, num_invalid_configurations = wdo_alignment_utils.align_rooms_by_wd(
                            pano_dict_inferred[i1],
                            pano_dict_inferred[i2],
                            use_inferred_wdos_layout=use_inferred_wdos_layout,
                            transform_type=AlignTransformType.SE2,
                        )
                    else:
                        possible_alignment_info, num_invalid_configurations = wdo_alignment_utils.align_rooms_by_wd(
                            pano_dict[i1],
                            pano_dict[i2],
                            use_inferred_wdos_layout=use_inferred_wdos_layout,
                            transform_type=AlignTransformType.SE2,
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

    print(f"{OKGREEN} Building {building_id}: " + ENDCOLOR)
    for floor_id, gt_is_valid_arr in floor_gt_is_valid_report_dict.items():
        print(f"{OKGREEN} {floor_id}: {np.mean(gt_is_valid_arr):.2f} over {len(gt_is_valid_arr)} alignment pairs." + ENDCOLOR)
    print(HEADER + ENDCOLOR)


def export_alignment_hypotheses_to_json(
    num_processes: int, raw_dataset_dir: str, hypotheses_save_root: str, use_inferred_wdos_layout: bool
) -> None:
    """Use multiprocessing to dump alignment hypotheses for all buildings to JSON.

    To confirm: Last edge of polygon (to close it) is not provided -- right??
    are all polygons closed? or just polylines?

    Args:
        num_processes: number of processes to use for parallel alignment generation.
        raw_dataset_dir: path to ZinD dataset.
        hypotheses_save_root
        use_inferred_wdos_layout: whether to use inferred W/D/O + inferred layout (or instead to use GT).
    """
    building_ids = [Path(fpath).stem for fpath in glob.glob(f"{raw_dataset_dir}/*") if Path(fpath).is_dir()]
    building_ids.sort()

    args = []

    for building_id in building_ids:

        json_annot_fpath = f"{raw_dataset_dir}/{building_id}/zind_data.json"
        # render_building(building_id, pano_dir, json_annot_fpath)

        args += [(hypotheses_save_root, building_id, json_annot_fpath, raw_dataset_dir, use_inferred_wdos_layout)]

    if num_processes > 1:
        with Pool(num_processes) as p:
            p.starmap(export_single_building_wdo_alignment_hypotheses, args)
    else:
        for single_call_args in args:
            export_single_building_wdo_alignment_hypotheses(*single_call_args)


if __name__ == "__main__":
    """
    Alignment hypotheses have been saved to the following locations:

    On se1-001:
    - from Predicted WDO + Predicted Layout:
        /home/johnlam/ZinD_bridge_api_alignment_hypotheses_madori_rmx_v1_2021_10_20_SE2_width_thresh0.65
    - from GT WDO + GT Layout:
        /home/johnlam/ZinD_bridge_api_alignment_hypotheses_GT_WDO_2021_11_20_SE2_width_thresh0.8

    Locally:
    - from GT WDO + GT Layout:
        /Users/johnlam/Downloads/ZinD_bridge_api_alignment_hypotheses_GT_WDO_2021_11_20_SE2_width_thresh0.8

    ZInD is stored in the following locations:
    - DGX: /mnt/data/johnlam/zind_bridgeapi_2021_10_05
    - se1-001: /home/johnlam/zind_bridgeapi_2021_10_05
    - locally: /Users/johnlam/Downloads/zind_bridgeapi_2021_10_05
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_processes",
        type=int,
        default=32,
        help="Number of processes to use for parallel generation of alignment hypotheses. "
        "Each worker processes one building at a time.",
    )
    parser.add_argument(
        "--raw_dataset_dir",
        type=str,
        # "/mnt/data/johnlam/zind_bridgeapi_2021_10_05"
        # "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05"
        default="/home/johnlam/zind_bridgeapi_2021_10_05",
        help="where ZInD dataset is stored on disk (after download from Bridge API)",
    )
    parser.add_argument(
        "--hypotheses_save_root",
        type=str,
        # "/home/johnlam/ZinD_bridge_api_alignment_hypotheses_GT_WDO_2021_11_20_SE2_width_thresh0.8"
        # "/Users/johnlam/Downloads/ZinD_bridge_api_alignment_hypotheses_GT_WDO_2021_11_20_SE2_width_thresh0.8"
        default="/home/johnlam/ZinD_bridge_api_alignment_hypotheses_madori_rmx_v1_2021_10_20_SE2_width_thresh0.65",
        help="Directory where JSON files with alignment hypotheses will be saved to.",
    )
    parser.add_argument(
        "--wdo_source",
        type=str,
        required=True,
        choices=["horizon_net","ground_truth"],
        help="Where to pull W/D/O and layout (either inferred from HorizonNet, or taken from annotated ground truth)"
    )
    args = parser.parse_args()

    # if the W/D/O source is not HorizonNet, then we'll fall back to using annotated GT W/D/O.
    use_inferred_wdos_layout = args.wdo_source == "horizon_net"

    export_alignment_hypotheses_to_json(
        args.num_processes, args.raw_dataset_dir, args.hypotheses_save_root, use_inferred_wdos_layout
    )

    # test_prune_to_unique_sim2_objs()
