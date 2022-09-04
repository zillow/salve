"""Unit tests for ZInD dataloader."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import salve.dataset.zind_data as zind_data_utils
from salve.dataset.zind_data import ZindData


_RENDERINGS_SAMPLE_ROOT = Path(__file__).resolve().parent.parent / "test_data" / "Renderings"

IMG_FNAME_CEILING_1 = "pair_58___door_0_0_rotated_ceiling_rgb_floor_01_partial_room_04_pano_5.jpg"
IMG_FNAME_CEILING_2 = "pair_58___door_0_0_rotated_ceiling_rgb_floor_01_partial_room_07_pano_8.jpg"
IMG_FNAME_FLOOR_1 = "pair_58___door_0_0_rotated_floor_rgb_floor_01_partial_room_04_pano_5.jpg"
IMG_FNAME_FLOOR_2 = "pair_58___door_0_0_rotated_floor_rgb_floor_01_partial_room_07_pano_8.jpg"


def test_pair_idx_from_fpath() -> None:
    """ """
    # fpath = "/Users/johnlam/Downloads/DGX-rendering-2021_06_25/ZinD_BEV_RGB_only_2021_06_25/gt_alignment_approx/000/pair_10_floor_rgb_floor_01_partial_room_01_pano_15.jpg"
    fpath = "/mnt/data/johnlam/ZinD_BEV_RGB_only_2021_07_14_v3/gt_alignment_approx/1394/pair_24___opening_0_0_identity_ceiling_rgb_floor_01_partial_room_01_pano_18.jpg"
    pair_idx = zind_data_utils.pair_idx_from_fpath(fpath)

    assert pair_idx == 24

def test_pano_id_from_fpath() -> None:
    """ """
    # fpath = "/Users/johnlam/Downloads/DGX-rendering-2021_06_25/ZinD_BEV_RGB_only_2021_06_25/gt_alignment_approx/242/pair_58_floor_rgb_floor_01_partial_room_13_pano_25.jpg"
    fpath = "/mnt/data/johnlam/ZinD_BEV_RGB_only_2021_07_14_v3/gt_alignment_approx/1394/pair_24___opening_0_0_identity_ceiling_rgb_floor_01_partial_room_01_pano_18.jpg"
    pano_id = zind_data_utils.pano_id_from_fpath(fpath)
    assert pano_id == 18

def test_ZindData_constructor() -> None:
    """Smokescreen to make sure ZindData object can be constructed successfully."""

    with tempfile.TemporaryDirectory() as tmp_data_root:
        transform = None
        args = MagicMock()
        args.modalities = ["ceiling_rgb_texture", "floor_rgb_texture"] # ["layout"]
        # Dummy path to BEV renderings.
        args.data_root = tmp_data_root

        # Copy 4 renderings into the temporary directory.
        shutil.copytree(src=_RENDERINGS_SAMPLE_ROOT / "gt_alignment_approx", dst=tmp_data_root + "/gt_alignment_approx")

        dataset = ZindData(split="train", transform=transform, args=args)
        # There should be 1 set of 4-tuples.
        assert len(dataset.data_list) == 1

        x1c_fpath, x2c_fpath, x1f_fpath, x2f_fpath, is_match = dataset.data_list[0]

        assert Path(x1c_fpath).name == IMG_FNAME_CEILING_1
        assert Path(x2c_fpath).name == IMG_FNAME_CEILING_2
        assert Path(x1f_fpath).name == IMG_FNAME_FLOOR_1
        assert Path(x2f_fpath).name == IMG_FNAME_FLOOR_2
        # Comes from the `gt_alignment`.
        assert is_match

