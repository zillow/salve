"""Load results of SALVe's SfM (global pose estimation) algorithm."""

from enum import Enum, unique
from pathlib import Path
from typing import Optional

import gtsfm.utils.io as io_utils

from salve.common.posegraph2d import PoseGraph2d
from salve.common.pano_data import PanoData


@unique
class EstimatedBoundaryType(str, Enum):
    """Enum for estimated boundary type."""

    NONE: str = "NONE"
    HNET_CORNERS: str = "HNET_CORNERS"
    HNET_DENSE: str = "HNET_DENSE"


def load_estimated_pose_graph(
    json_fpath: Path,
    boundary_type: EstimatedBoundaryType,
    raw_dataset_dir: Optional[str],
    predictions_data_root: Optional[str],
) -> PoseGraph2d:
    """Load a pose graph exported from SALVe's `run_sfm.py` script.

    We optionally merge with HNet layout predictions, for downstream layout stitching.

    Args:
        json_fpath: Path to JSON file generated by SALVe's `run_sfm.py` script.
        boundary_type:
        raw_dataset_dir: only if HNet dense layout or corners are requested to be merged in.
        predictions_data_root: HNet predictions data root, only if HNet dense layout or corners are requested to be merged in.

    Returns:
        Estimated pose graph ...TODO
    """
    if not isinstance(json_fpath, Path):
        raise ValueError("`json_fpath` arg must be a pathlib.Path object.")
    if not json_fpath.exists():
        raise FileNotFoundError(f"File not found at {json_fpath}")
    localization_data = io_utils.read_json_file(json_fpath)

    building_id = localization_data["building_id"]

    if boundary_type in [EstimatedBoundaryType.HNET_CORNERS, EstimatedBoundaryType.HNET_DENSE]:
        hnet_floor_predictions = hnet_prediction_loader.load_hnet_predictions(
            query_building_id=building_id, raw_dataset_dir=raw_dataset_dir, predictions_data_root=predictions_data_root
        )
        import pdb; pdb.set_trace()

    nodes = {}
    for pano_id, v in localization_data["wSi_dict"].items():
        global_Sim2_local = Sim2(v["R"], t=v["t"], s=v["s"])
        nodes[pano_id] = PanoData(
            id=pano_id,
            global_Sim2_local=global_Sim2_local,
            room_vertices_local_2d=np.zeros((0, 2)),
            image_path=None,
            label=None,
            doors=None,
            windows=None,
            openings=None,
            vanishing_angle_deg=None,
        )

    return PoseGraph2d(
        building_id=building_id,
        floor_id=localization_data["floor_id"],
        nodes=nodes,
        scale_meters_per_coordinate=localization_data["scale_meters_per_coordinate"],
    )
