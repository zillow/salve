"""Unit tests for HorizonNet predictions loading."""

from pathlib import Path

import salve.dataset.hnet_prediction_loader as hnet_prediction_loader
from salve.dataset.mhnet_prediction import MHNetPanoStructurePrediction


_PREDICTIONS_SAMPLE_ROOT = Path(__file__).resolve().parent.parent / "test_data" / "ZInD_HorizonNet_predictions"
_ZIND_SAMPLE_ROOT = Path(__file__).resolve().parent.parent / "test_data" / "ZInD"


def test_get_floor_id_from_img_fpath() -> None:
    """Verify we can fetch the floor ID from a panorama file path."""
    img_fpath = "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05/0109/panos/floor_01_partial_room_03_pano_13.jpg"
    floor_id = hnet_prediction_loader.get_floor_id_from_img_fpath(img_fpath)
    assert floor_id == "floor_01"

    img_fpath = "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05/1386/panos/floor_02_partial_room_18_pano_53.jpg"
    floor_id = hnet_prediction_loader.get_floor_id_from_img_fpath(img_fpath)
    assert floor_id == "floor_02"


def test_load_hnet_predictions() -> None:
    """Verifies that the HorizonNet prediction for each pano from ZInD building `0000` can be loaded correctly."""

    building_id = "0000"
    raw_dataset_dir = _ZIND_SAMPLE_ROOT
    predictions_data_root = _PREDICTIONS_SAMPLE_ROOT

    results: Dict[int, MHNetPanoStructurePrediction] = hnet_prediction_loader.load_hnet_predictions(
        building_id=building_id, raw_dataset_dir=raw_dataset_dir, predictions_data_root=predictions_data_root
    )

    assert len(results.keys()) == 1
    assert "floor_01" in results

    for v in results["floor_01"].values():
        assert isinstance(v, MHNetPanoStructurePrediction)

    expected_pano_ids = [
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        31,
        32,
        33,
        34,
    ]

    assert sorted(results["floor_01"].keys()) == expected_pano_ids
    assert len(sorted(results["floor_01"].keys())) == 32
