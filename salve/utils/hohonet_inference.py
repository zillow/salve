"""Utilities for HoHoNet monocular depth estimation."""

import os
from pathlib import Path
from types import SimpleNamespace

# refers to the HoHoNet repo
from salve.utils.infer_depth import infer_depth


HOHONET_CONFIG_FPATH = "config/mp3d_depth/HOHO_depth_dct_efficienthc_TransEn1_hardnet.yaml"
HOHONET_CKPT_FPATH = "ckpt/mp3d_depth_HOHO_depth_dct_efficienthc_TransEn1_hardnet/ep60.pth"


def infer_depth_if_nonexistent(depth_save_root: str, building_id: str, img_fpath: str) -> None:
    """Infer depth map if it does not already exist at `depth_save_root`."""
    fname_stem = Path(img_fpath).stem
    building_depth_save_dir = f"{depth_save_root}/{building_id}"
    if Path(f"{building_depth_save_dir}/{fname_stem}.depth.png").exists():
        return

    os.makedirs(building_depth_save_dir, exist_ok=True)

    args = SimpleNamespace(
        **{
            "cfg": HOHONET_CONFIG_FPATH,
            "pth": HOHONET_CKPT_FPATH,
            "out": building_depth_save_dir,
            "inp": img_fpath,
            "opts": [],
        }
    )
    infer_depth(args)
