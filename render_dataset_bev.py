
import glob
import os
from pathlib import Path
from types import SimpleNamespace

import imageio

from infer_depth import infer_depth
from vis_depth import vis_depth


def render_dataset():
    """ """

    building_id = "000"
    img_fpaths = glob.glob(f"/Users/johnlam/Downloads/2021_05_28_Will_amazon_raw/{building_id}/panos/*.jpg")

    for img_fpath in img_fpaths:

        args = SimpleNamespace(**{
            "cfg": "config/mp3d_depth/HOHO_depth_dct_efficienthc_TransEn1_hardnet.yaml",
            "pth": "ckpt/mp3d_depth_HOHO_depth_dct_efficienthc_TransEn1_hardnet/ep60.pth",
            "out": "assets/",
            "inp": img_fpath,
            "opts": []
        })

        infer_depth(args)



        args = SimpleNamespace(**{
            "img": img_fpath,
            "depth": f"assets/{Path(img_fpath).stem}.depth.png",
            "scale": 0.001,
            "crop_ratio": 80/512,
            "crop_z_above": 0.3 # 1.2
        })
        bev_img = vis_depth(args)

        save_dir = f"/Users/johnlam/Downloads/ZinD_BEV/{building_id}"
        os.makedirs(save_dir, exist_ok=True)
        imageio.imwrite(f"{save_dir}/{Path(img_fpath).stem}.jpg", bev_img)


if __name__ == "__main__":
    render_dataset()