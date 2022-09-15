"""Unit tests on OpenMVG related code."""

import tempfile

import salve.baselines.openmvg as openmvg_utils


def test_find_seed_pair() -> None:
    """Find potential seed pair (adjacent image IDs) from ZInD building `0000`.

    Could be located at a folder e.g. "openmvg_demo/ZinD_000_floor_01__2021_09_21/images".
    """
    fnames = [
        "floor_01_partial_room_08_pano_31.jpg",
        "floor_01_partial_room_01_pano_14.jpg",
        "floor_01_partial_room_01_pano_15.jpg",
        "floor_01_partial_room_02_pano_29.jpg",
        "floor_01_partial_room_03_pano_13.jpg",
        "floor_01_partial_room_04_pano_32.jpg",
        "floor_01_partial_room_05_pano_26.jpg",
        "floor_01_partial_room_06_pano_10.jpg",
        "floor_01_partial_room_06_pano_11.jpg",
        "floor_01_partial_room_06_pano_12.jpg",
        "floor_01_partial_room_07_pano_18.jpg",
        "floor_01_partial_room_07_pano_19.jpg",
        "floor_01_partial_room_09_pano_2.jpg",
        "floor_01_partial_room_09_pano_4.jpg",
        "floor_01_partial_room_09_pano_5.jpg",
        "floor_01_partial_room_09_pano_6.jpg",
        "floor_01_partial_room_10_pano_16.jpg",
        "floor_01_partial_room_10_pano_17.jpg",
        "floor_01_partial_room_10_pano_22.jpg",
        "floor_01_partial_room_11_pano_24.jpg",
        "floor_01_partial_room_11_pano_25.jpg",
        "floor_01_partial_room_12_pano_3.jpg",
        "floor_01_partial_room_13_pano_9.jpg",
        "floor_01_partial_room_14_pano_21.jpg",
        "floor_01_partial_room_15_pano_33.jpg",
        "floor_01_partial_room_15_pano_34.jpg",
        "floor_01_partial_room_16_pano_23.jpg",
        "floor_01_partial_room_17_pano_7.jpg",
        "floor_01_partial_room_17_pano_8.jpg",
        "floor_01_partial_room_18_pano_20.jpg",
        "floor_01_partial_room_19_pano_27.jpg",
        "floor_01_partial_room_19_pano_28.jpg",
    ]

    with tempfile.TemporaryDirectory() as image_dirpath:

        # Create an empty file corresponding to each image filename from this ZInD floor.
        for fname in fnames:
            image_fpath = image_dirpath + "/" + fname
            f = open(image_fpath, "w")
            f.close()

        seed_fname1, seed_fname2 = openmvg_utils.find_seed_pair(image_dirpath)
        assert seed_fname1 == "floor_01_partial_room_09_pano_2.jpg"
        assert seed_fname2 == "floor_01_partial_room_12_pano_3.jpg"
