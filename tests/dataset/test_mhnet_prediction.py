"""Unit tests for `MHNetPanoStructurePrediction` class."""

from pathlib import Path

import numpy as np

import salve.dataset.mhnet_prediction as hnet_pred_utils
from salve.dataset.mhnet_prediction import MHNetDWO, MHNetPanoStructurePrediction

_PREDICTIONS_SAMPLE_ROOT = Path(__file__).resolve().parent.parent / "test_data" / "ZInD_HorizonNet_predictions"
_ZIND_SAMPLE_ROOT = Path(__file__).resolve().parent.parent / "test_data" / "ZInD"


def test_merge_wdos_straddling_img_border_windows() -> None:
    """Ensures we can properly merge **window** objects that have been split by the panorama seam.

    Data from ZinD Building 0000, Pano 17
    """
    windows = []
    windows_merged = hnet_pred_utils.merge_wdos_straddling_img_border(wdo_instances=windows)
    assert len(windows_merged) == 0
    assert isinstance(windows_merged, list)


def test_merge_wdos_straddling_img_border_doors() -> None:
    """Ensures we can properly merge **door** objects that have been split by the panorama seam.

    Data from ZinD Building 0000, Pano 17.
    """
    doors = [
        MHNetDWO(s=0.14467253176930597, e=0.3704789833822092),
        MHNetDWO(s=0.45356793743890517, e=0.46920821114369504),
        MHNetDWO(s=0.47702834799608995, e=0.5278592375366569),
        MHNetDWO(s=0.5376344086021505, e=0.5865102639296188),
        MHNetDWO(s=0.6217008797653959, e=0.8084066471163245),
    ]
    doors_merged = hnet_pred_utils.merge_wdos_straddling_img_border(wdo_instances=doors)

    assert doors == doors_merged
    # Should be same as input -- no doors straddle the image border for this panorama.
    assert len(doors_merged) == 5


def test_merge_wdos_straddling_img_border_openings() -> None:
    """Ensure we can properly merge **opening** objects that have been split by the panorama seam.

    On ZinD Building 0000, Pano 17

    Other good examples are:
    Panos 16, 22, 33 for building 0000.
    Pano 21 for building 0001.
    """
    openings = [
        MHNetDWO(s=0.0009775171065493646, e=0.10361681329423265),
        MHNetDWO(s=0.9354838709677419, e=1.0),
    ]
    openings_merged = hnet_pred_utils.merge_wdos_straddling_img_border(wdo_instances=openings)

    assert len(openings_merged) == 1
    assert openings_merged[0] == MHNetDWO(s=0.9354838709677419, e=0.10361681329423265)


def test_pano_structure_prediction_rmx_madori_v1_from_json_fpath() -> None:
    """Verifies that a JSON file with HorizonNet predictions can be parsed into layout and W/D/O predictions."""
    raw_dataset_dir = _ZIND_SAMPLE_ROOT
    predictions_data_root = _PREDICTIONS_SAMPLE_ROOT
    building_id = "0000"
    image_fname_stem = "floor_01_partial_room_09_pano_2"

    json_fpath = predictions_data_root / "horizon_net" / building_id / f"{image_fname_stem}.json"
    image_fpath = raw_dataset_dir / building_id / "panos" / f"{image_fname_stem}.jpg"
    result = MHNetPanoStructurePrediction.from_json_fpath(json_fpath=json_fpath, image_fpath=image_fpath)

    # Verify image metadata associated with prediction.
    assert result.image_width == 1024
    assert result.image_height == 512
    assert result.image_fpath == image_fpath

    # Verify room corners.
    assert isinstance(result.corners_in_uv, np.ndarray)
    assert result.corners_in_uv.shape == (20, 2)
    # fmt: off
    expected_first_corners =  np.array(
        [[0.02813019, 0.35113618],
        [0.02813019, 0.64691073]])
    # fmt: on
    assert np.allclose(result.corners_in_uv[:2], expected_first_corners)

    # Verify floor boundary.
    assert isinstance(result.floor_boundary, np.ndarray)
    assert result.floor_boundary.shape == (1024,)
    expected_floor_boundary_start = np.array([326.23584, 325.536102, 324.849243, 324.179382, 323.147888, 322.917572])
    assert np.allclose(result.floor_boundary[:6], expected_floor_boundary_start)

    # Verify floor boundary uncertainty.
    assert isinstance(result.floor_boundary_uncertainty, np.ndarray)
    assert result.floor_boundary_uncertainty.shape == (1024,)
    expected_floor_boundary_uncertainty_start = np.array(
        [10.536544, 10.46075, 10.376159, 10.330658, 9.964458, 9.891422]
    )
    assert np.allclose(result.floor_boundary_uncertainty[:6], expected_floor_boundary_uncertainty_start)

    # Verify doors.
    assert isinstance(result.doors, list)  # List[MHNetDWO]
    expected_door_wdos = [MHNetDWO(s=0.4359726295210166, e=0.5640273704789834)]
    assert result.doors == expected_door_wdos

    # Verify windows.
    assert isinstance(result.windows, list)  # : List[MHNetDWO]
    expected_window_wdos = [
        MHNetDWO(s=0.6383186705767351, e=0.6598240469208211),
        MHNetDWO(s=0.6695992179863147, e=0.6930596285434996),
    ]
    assert result.windows == expected_window_wdos

    # Verify openings.
    assert isinstance(result.openings, list)  #: List[MHNetDWO]
    # Two openings become merged, as an opening straddles the image border.
    expected_opening_wdos = [
        MHNetDWO(s=0.8299120234604106, e=0.8690127077223851),
        MHNetDWO(s=0.9130009775171065, e=0.024437927663734114),
    ]
    assert result.openings == expected_opening_wdos
