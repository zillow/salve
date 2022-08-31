"""Unit tests for ZInD dataloader."""

import salve.dataset.zind_data as zind_data

def test_pair_idx_from_fpath() -> None:
    """ """
    # fpath = "/Users/johnlam/Downloads/DGX-rendering-2021_06_25/ZinD_BEV_RGB_only_2021_06_25/gt_alignment_approx/000/pair_10_floor_rgb_floor_01_partial_room_01_pano_15.jpg"
    fpath = "/mnt/data/johnlam/ZinD_BEV_RGB_only_2021_07_14_v3/gt_alignment_approx/1394/pair_24___opening_0_0_identity_ceiling_rgb_floor_01_partial_room_01_pano_18.jpg"
    pair_idx = pair_idx_from_fpath(fpath)

    assert pair_idx == 24

def test_pano_id_from_fpath() -> None:
    """ """
    # fpath = "/Users/johnlam/Downloads/DGX-rendering-2021_06_25/ZinD_BEV_RGB_only_2021_06_25/gt_alignment_approx/242/pair_58_floor_rgb_floor_01_partial_room_13_pano_25.jpg"
    fpath = "/mnt/data/johnlam/ZinD_BEV_RGB_only_2021_07_14_v3/gt_alignment_approx/1394/pair_24___opening_0_0_identity_ceiling_rgb_floor_01_partial_room_01_pano_18.jpg"
    pano_id = pano_id_from_fpath(fpath)
    assert pano_id == 18

def test_ZindData_constructor() -> None:
    """ """
    transform = None

    from types import SimpleNamespace

    args = SimpleNamespace(**{"data_root": "/mnt/data/johnlam/ZinD_BEV_RGB_only_2021_07_14_v3"})
    dataset = ZindData(split="val", transform=transform, args=args)
    assert len(dataset.data_list) > 0


def test_ZindData_getitem() -> None:
    """ """
    pass


if __name__ == "__main__":
    """ """
    test_ZindData_constructor()
