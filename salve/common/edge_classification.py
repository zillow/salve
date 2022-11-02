"""Data structure representing a CNN-classification result on a relative pose hypothesis (from a W/D/O alignment)."""

import glob
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tqdm

import salve.utils.io as io_utils
import salve.utils.rotation_utils as rotation_utils
from salve.common.edgewdopair import EdgeWDOPair
from salve.common.posegraph2d import PoseGraph2d
from salve.common.sim2 import Sim2
from salve.common.two_view_estimation_report import TwoViewEstimationReport


@dataclass(frozen=False)
class EdgeClassification:
    """Represents a model prediction for a particular alignment hypothesis between Panorama i1 and Panorama i2.

    Attributes:
        i1: ID of panorama 1.
        i2: ID of panorama 2.
        prob: probability of ...
        y_hat: predicted category.
        y_true: true category.
        pair_idx: TODO
        wdo_pair_uuid: TODO
        configuration: surface normal configuration (identity or rotated).
        building_id: unique ID for ZinD building.
        floor_id: unique ID for floor of a ZinD building.
        i2Si1: Similarity(2) transformation, such that p_i2 = i2Si1 * p_i1.
    """

    i1: int
    i2: int
    prob: float
    y_hat: int
    y_true: int
    pair_idx: int
    wdo_pair_uuid: str
    configuration: str
    building_id: str
    floor_id: str
    i2Si1: Sim2

    def compute_measurement_relative_pose_error_from_gt(self, gt_floor_pose_graph: PoseGraph2d) -> Tuple[float, float]:
        """For a predicted relative pose from an alignment hypothesis, compare its error w.r.t. ground truth pose graph.

        Args:
            gt_floor_pg: ground truth pose graph.
            hypotheses_save_root: Directory where JSON files with alignment hypotheses have been saved to (from
                executing `export_alignment_hypotheses.py`)",
        Returns:
            rot_error_deg: rotation error (in degrees).
            trans_error: translation error.
        """
        wTi1_gt = gt_floor_pose_graph.nodes[self.i1].global_Sim2_local
        wTi2_gt = gt_floor_pose_graph.nodes[self.i2].global_Sim2_local
        i2Ti1_gt = wTi2_gt.inverse().compose(wTi1_gt)
        theta_deg_gt = i2Ti1_gt.theta_deg

        # Technically it is i2Si1, but scale will always be 1 with inferred W/D/O.
        i2Ti1 = self.i2Si1
        theta_deg_est = i2Ti1.theta_deg

        # Note: must wrap error around at 360 degrees to find smaller of possible options.
        rot_error_deg = rotation_utils.wrap_angle_deg(theta_deg_gt, theta_deg_est)
        trans_error = np.linalg.norm(i2Ti1_gt.translation - i2Ti1.translation)
        return rot_error_deg, trans_error


def get_available_floor_ids_building_ids_from_serialized_preds(serialized_preds_json_dir: str) -> List[Tuple[str, str]]:
    """Identify unique (building id, floor id) pairs for which serialized SALVe predictions are available on disk.

    Args:
        serialized_preds_json_dir: path to directory where model predictions (per edge) have been serialized as JSON.
            The serializations are stored per batch, and thus are mixed across ZInD buildings and floors.

    Returns:
        List of unique (building id, floor id) pairs.
    """
    building_id_floor_id_pairs = set()

    json_fpaths = glob.glob(f"{serialized_preds_json_dir}/batch*.json")
    for json_fpath in tqdm.tqdm(json_fpaths):
        json_data = io_utils.read_json_file(json_fpath)
        fp0_list = json_data["fp0"]
        for fp0 in fp0_list:
            building_id = Path(fp0).parent.stem
            s = Path(fp0).stem.find("floor_0")
            e = Path(fp0).stem.find("_partial")
            floor_id = Path(fp0).stem[s:e]

            building_id_floor_id_pairs.add((building_id, floor_id))

    return list(building_id_floor_id_pairs)


def get_edge_classifications_from_serialized_preds(
    query_building_id: str,
    query_floor_id: str,
    serialized_preds_json_dir: str,
    hypotheses_save_root: str,
    allowed_wdo_types: List[str] = ["door", "window", "opening"],
    confidence_threshold: Optional[float] = None,
) -> Dict[Tuple[str, str], List[EdgeClassification]]:
    """Converts serialized predictions into EdgeClassification objects.

    Given a directory of JSON files containing model predictions into predictions per ZinD building and per floor.

    Args:
        query_building_id: ZInD building ID to retrieve serialized predictions for.
        query_floor_id: unique ID of floor, from ZInD building ID specified above, to retrieve serialized predictions for.
        serialized_preds_json_dir: Path to directory where model predictions (per edge) have been serialized as JSON.
            The serializations are stored per batch, and thus are mixed across ZInD buildings and floors.
        hypotheses_save_root:  Directory where JSON files with alignment hypotheses have been saved to (from executing
            `export_alignment_hypotheses.py`).
        allowed_wdo_types: Allowed types of semantic objects (W/D/O) to use for reconstruction. Others will be ignored.
        confidence_threshold: Minimum required SALVe network confidence to accept a prediction. By thresholding
            at this stage, function execution is accelerated, as fewer glob() calls are needed for fewer hypotheses.

    Returns:
        floor_edgeclassifications_dict: a mapping from (building_id, floor_id) to corresponding EdgeClassification
            measurements.
    """
    floor_edgeclassifications_dict = defaultdict(list)

    json_fpaths = glob.glob(f"{serialized_preds_json_dir}/batch*.json")
    print("Converting serialized CNN predictions to relative poses / EdgeClassifications...")
    for json_fpath in tqdm.tqdm(json_fpaths):

        json_data = io_utils.read_json_file(json_fpath)
        y_hat_list = json_data["y_hat"]
        y_true_list = json_data["y_true"]
        y_hat_prob_list = json_data["y_hat_probs"]
        fp0_list = json_data["fp0"]
        fp1_list = json_data["fp1"]

        for y_hat, y_true, y_hat_prob, fp0, fp1 in zip(y_hat_list, y_true_list, y_hat_prob_list, fp0_list, fp1_list):
            # Note: not guaranteed that i1 < i2
            i1_ = int(Path(fp0).stem.split("_")[-1])
            i2_ = int(Path(fp1).stem.split("_")[-1])

            # TODO: this should never happen bc sorted, figure out why it occurs
            # (happens because we sort by partial room, not by i1, i2 in dataloader?)
            i1 = min(i1_, i2_)
            i2 = max(i1_, i2_)

            building_id = Path(fp0).parent.stem
            if building_id != query_building_id:
                continue

            s = Path(fp0).stem.find("floor_0")
            e = Path(fp0).stem.find("_partial")
            floor_id = Path(fp0).stem[s:e]

            if floor_id != query_floor_id:
                continue

            pair_idx = Path(fp0).stem.split("_")[1]

            is_identity = "identity" in Path(fp0).stem
            configuration = "identity" if is_identity else "rotated"

            # Rip out the WDO indices (`wdo_pair_uuid`), given a filename such as
            # `pair_3905___door_3_0_identity_floor_rgb_floor_01_partial_room_02_pano_38.jpg`
            k = Path(fp0).stem.split("___")[1].find(f"_{configuration}")
            assert k != -1
            wdo_pair_uuid = Path(fp0).stem.split("___")[1][:k]
            # split `door_3_0` to `door`
            wdo_type = wdo_pair_uuid.split("_")[0]
            if wdo_type not in allowed_wdo_types:
                continue

            if confidence_threshold is not None and y_hat_prob < confidence_threshold:
                continue

            # Retrieve the Sim(2) relative pose that corresponds to this edge's alignment hypothesis.
            # Given a classification result (measurement), find corresponding Sim(2) alignment JSON file on disk.
            # Look up the associated Sim(2) file for this prediction, by looping through the pair idxs again
            label_dirname = "gt_alignment_approx" if y_true else "incorrect_alignment"
            fpaths = glob.glob(
                f"{hypotheses_save_root}/{building_id}/{floor_id}"
                f"/{label_dirname}/{i1}_{i2}__{wdo_pair_uuid}_{configuration}.json"
            )
            if not len(fpaths) == 1:
                raise ValueError("No corresponding serialized alignment hypothesis was found on disk for measurement.")
            json_fpath = fpaths[0]
            i2Si1 = Sim2.from_json(json_fpath)

            floor_edgeclassifications_dict[(building_id, floor_id)] += [
                EdgeClassification(
                    i1=i1,
                    i2=i2,
                    prob=y_hat_prob,
                    y_hat=y_hat,
                    y_true=y_true,
                    pair_idx=pair_idx,
                    wdo_pair_uuid=wdo_pair_uuid,
                    configuration=configuration,
                    building_id=building_id,
                    floor_id=floor_id,
                    i2Si1=i2Si1,
                )
            ]
    return floor_edgeclassifications_dict


def get_conf_thresholded_edge_measurements(
    measurements: List[EdgeClassification], confidence_threshold: float
) -> List[EdgeClassification]:
    """Threshold floor edge predictions by confidence,

    Among all model predictions for a particular floor of a home, select only the positive predictions
    with sufficiently high confidence.

    Args:
        measurements: unthresholded edge predictions.
        confidence_threshold: minimum confidence to treat a model's prediction as a positive.

    Returns:
        high_conf_measurements: all measurements of sufficient confidence for one floor, even if forming a multigraph.
    """
    high_conf_measurements = []

    num_gt_negatives = 0
    num_gt_positives = 0

    for m in measurements:
        if m.y_true == 1:
            num_gt_positives += 1
        else:
            num_gt_negatives += 1

        # Find all of the predictions where predicted class is 1.
        if m.y_hat != 1:
            continue

        if m.prob < confidence_threshold:
            continue

        high_conf_measurements.append(m)

    EPS = 1e-10
    class_imbalance_ratio = num_gt_negatives / (num_gt_positives + EPS)
    print(f"\tNeg. vs. Pos. class imbalance ratio {class_imbalance_ratio:.2f} : 1")
    return high_conf_measurements


def get_most_likely_relative_pose_per_edge(
    measurements: List[EdgeClassification],
    hypotheses_save_root: str,
    building_id: str,
    floor_id: str,
    gt_floor_pose_graph: Optional[PoseGraph2d] = None,
) -> Tuple[
    Dict[Tuple[int, int], Sim2],
    Dict[Tuple[int, int], TwoViewEstimationReport],
    Dict[Tuple[int, int], EdgeWDOPair],
    Dict[Tuple[int, int], EdgeClassification],
]:
    """Obtain relative poses corresponding to the most confident measurement/prediction per edge (for a single floor).

    Args:
        measurements: list containing the model's classification prediction for each edge (likely already thresholded).
        hypotheses_save_root: path to directory where alignment hypotheses are saved as JSON files.
        building_id: unique ID for ZinD building.
        floor_id: unique ID for floor of a ZinD building.
        gt_floor_pose_graph: ground truth pose graph for this particular building floor, to allow
            computation of per-edge errors w.r.t. GT.

    Returns:
        i2Si1_dict: Similarity(2) relative pose for each (i1,i2) pano-pano edge.
        two_view_reports_dict: mapping from (i1,i2) pano pair to relative (R,t) errors w.r.t. GT.
        per_edge_wdo_dict: mapping from edge (i1,i2) to EdgeWDOPair information.
        edge_classification_dict: mapping from edge (i1,i2) to EdgeClassification information.
    """
    most_confident_edge_dict = defaultdict(list)
    for m in measurements:
        most_confident_edge_dict[(m.i1, m.i2)] += [m]

    per_edge_wdo_dict: Dict[Tuple[int, int], EdgeWDOPair] = {}
    edge_classification_dict: Dict[Tuple[int, int], EdgeClassification] = {}
    i2Si1_dict: Dict[Tuple[int, int], Sim2] = {}

    # Keep track of how often the most confident prediction per edge was the correct one.
    most_confident_was_correct = []
    for (i1, i2), measurements in most_confident_edge_dict.items():

        # For each edge, choose the most confident prediction over all W/D/O pair alignments.
        most_confident_idx = np.argmax([m.prob for m in measurements])
        m = measurements[most_confident_idx]
        if len(measurements) > 1:
            most_confident_was_correct.append(m.y_true == 1)

        per_edge_wdo_dict[(i1, i2)] = EdgeWDOPair.from_wdo_pair_uuid(i1=i1, i2=i2, wdo_pair_uuid=m.wdo_pair_uuid)
        edge_classification_dict[(i1, i2)] = m
        # Convert alignment hypothesis ID to relative pose.
        i2Si1 = m.i2Si1
        i2Si1_dict[(m.i1, m.i2)] = i2Si1

    two_view_reports_dict = create_two_view_reports_dict_from_edge_classification_dict(
        edge_classification_dict, gt_floor_pose_graph
    )
    # Analyze how often is the most confident edge the right edge, among all of the choices? (using model confidence).
    print(f"most confident was correct {np.array(most_confident_was_correct).mean():.2f}")
    return (i2Si1_dict, two_view_reports_dict, per_edge_wdo_dict, edge_classification_dict)


def create_two_view_reports_dict_from_edge_classification_dict(
    edge_classification_dict: Dict[Tuple[int, int], EdgeClassification], gt_floor_pose_graph: PoseGraph2d
) -> Dict[Tuple[int, int], TwoViewEstimationReport]:
    """Computes (R,t) errors w.r.t. ground truth for each edge, and stores them in a data structure.

    Args:
        edge_classification_dict: mapping from (i1,i2) pano pair to classification prediction.
        gt_floor_pose_graph: ground truth pose graph for a single ZInD floor.

    Returns:
        two_view_reports_dict: mapping from (i1,i2) pano pair to relative (R,t) errors w.r.t. GT.
    """
    two_view_reports_dict = {}
    for (i1, i2), m in edge_classification_dict.items():
        # TODO: modify U error to be `t` error, and not an angular error.
        R_error_deg, U_error_deg = m.compute_measurement_relative_pose_error_from_gt(
            gt_floor_pose_graph=gt_floor_pose_graph
        )
        two_view_reports_dict[(m.i1, m.i2)] = TwoViewEstimationReport(
            gt_class=m.y_true, R_error_deg=R_error_deg, U_error_deg=U_error_deg, confidence=m.prob
        )
    return two_view_reports_dict
