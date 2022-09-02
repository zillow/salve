"""Unit tests on OpenMVG related code."""

import salve.baselines.openmvg as openmvg_utils


def test_find_seed_pair() -> None:
    """ """
    image_dirpath = "/Users/johnlam/Downloads/openmvg_demo/ZinD_000_floor_01__2021_09_21/images"
    seed_fname1, seed_fname2 = openmvg_utils.find_seed_pair(image_dirpath)
    assert seed_fname1 == "floor_01_partial_room_09_pano_2.jpg"
    assert seed_fname2 == "floor_01_partial_room_12_pano_3.jpg"
