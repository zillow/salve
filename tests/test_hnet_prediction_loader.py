"""Unit tests for HorizonNet predictions loading."""

import salve.dataset.hnet_prediction_loader as hnet_prediction_loader


def test_get_floor_id_from_img_fpath() -> None:
    """Verify we can fetch the floor ID from a panorama file path."""
    img_fpath = "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05/0109/panos/floor_01_partial_room_03_pano_13.jpg"
    floor_id = hnet_prediction_loader.get_floor_id_from_img_fpath(img_fpath)
    assert floor_id == "floor_01"

    img_fpath = "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05/1386/panos/floor_02_partial_room_18_pano_53.jpg"
    floor_id = hnet_prediction_loader.get_floor_id_from_img_fpath(img_fpath)
    assert floor_id == "floor_02"
