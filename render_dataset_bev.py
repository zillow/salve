
import glob
import os
from pathlib import Path
from types import SimpleNamespace

import imageio

from infer_depth import infer_depth
from vis_depth import vis_depth, vis_depth_and_render


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
        #infer_depth(args)

        is_semantics = False
        semantic_img_fpath = f"/Users/johnlam/Downloads/MSeg_output/{Path(img_fpath).stem}_gray.jpg"

        if is_semantics:
            crop_z_above = 2.0
        else:
            crop_z_above = -1.0

        args = SimpleNamespace(**{
            "img": semantic_img_fpath if is_semantics else img_fpath,
            "depth": f"assets/{Path(img_fpath).stem}.depth.png",
            "scale": 0.001,
            "crop_ratio": 80/512, # throw away top 80 and bottom 80 rows of pixel (too noisy of estimates)
            "crop_z_above": crop_z_above #0.3 # -1.0 # -0.5 # 0.3 # 1.2
        })
        bev_img = vis_depth_and_render(args, is_semantics=False)

        #vis_depth(args)

        save_dir = f"/Users/johnlam/Downloads/ZinD_BEV_crop_above_{args.crop_z_above}/{building_id}"
        os.makedirs(save_dir, exist_ok=True)
        if is_semantics:
            img_name = f"semantics_{Path(img_fpath).stem}.jpg"
        else:
            img_name = f"{Path(img_fpath).stem}.jpg"
        imageio.imwrite(f"{save_dir}/{img_name}", bev_img)


if __name__ == "__main__":
    render_dataset()