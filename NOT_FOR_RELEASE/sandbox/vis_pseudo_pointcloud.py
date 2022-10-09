
"""TODO..."""

import glob
import os
from pathlib import Path
from types import SimpleNamespace

import imageio

import salve.utils.hohonet_inference as hohonet_inference_utils


def render_dataset(bev_save_root: str, raw_dataset_dir: str) -> None:
    """Render inferred point clouds one by one.

    Args:
        bev_save_root:
        raw_dataset_dir:
    """
    # building_id = "000"  # "981"  #
    # building_id = "004"
    building_id = "0544"

    import pdb

    pdb.set_trace()

    img_fpaths = glob.glob(f"{raw_dataset_dir}/{building_id}/panos/*.jpg")

    for img_fpath in img_fpaths:

        if "_18.jpg" not in img_fpath:  # 13
            continue

        hohonet_inference_utils.infer_depth_if_nonexistent(depth_save_root, building_id, img_fpath)

        is_semantics = False
        semantic_img_fpath = f"/Users/johnlam/Downloads/MSeg_output/{Path(img_fpath).stem}_gray.jpg"

        if is_semantics:
            crop_z_range = [-float("inf"), 2]
        else:
            crop_z_range = [-float("inf"), float("inf")]  # -1.0]  # [0.5, float('inf')] #  # 2.0 # -1.0

        args = SimpleNamespace(
            **{
                "img": semantic_img_fpath if is_semantics else img_fpath,
                "depth": f"{depth_save_root}/{building_id}/{Path(img_fpath).stem}.depth.png",
                "scale": 0.001,
                "crop_ratio": 80 / 512,  # throw away top 80 and bottom 80 rows of pixel (too noisy of estimates)
                "crop_z_range": crop_z_range  # 0.3 # -1.0 # -0.5 # 0.3 # 1.2
                # "crop_z_above": 2,
            }
        )

        bev_img = bev_rendering_utils.vis_depth_and_render(args, is_semantics=False)

        bev_rendering_utils.vis_depth(args)

        # save_dir = f"/Users/johnlam/Downloads/ZinD_BEV_crop_above_{args.crop_z_above}/{building_id}"
        building_bev_save_dir = f"{bev_save_root}/{label_type}/{building_id}"
        fname_stem = Path(img_fpath).stem
        os.makedirs(save_dir, exist_ok=True)
        if is_semantics:
            img_name = f"semantics_{fname_stem}.jpg"
        else:
            img_name = f"{fname_stem}.jpg"
        imageio.imwrite(f"{building_bev_save_dir}/{img_name}", bev_img)


if __name__ == "__main__":

	raw_dataset_dir = "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05"
	render_dataset(bev_save_root, raw_dataset_dir)
	

