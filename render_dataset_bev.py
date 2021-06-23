
import glob
import os
from pathlib import Path
from types import SimpleNamespace

import imageio
import numpy as np
from argoverse.utils.sim2 import Sim2

from infer_depth import infer_depth
from vis_depth import vis_depth, vis_depth_and_render, render_bev_pair



def render_depth_if_nonexistent(img_fpath: str) -> None:
    """ """
    if Path(f"assets/{Path(img_fpath).stem}.depth.png").exists():
        return

    args = SimpleNamespace(**{
        "cfg": "config/mp3d_depth/HOHO_depth_dct_efficienthc_TransEn1_hardnet.yaml",
        "pth": "ckpt/mp3d_depth_HOHO_depth_dct_efficienthc_TransEn1_hardnet/ep60.pth",
        "out": "assets/",
        "inp": img_fpath,
        "opts": []
    })
    infer_depth(args)


def render_dataset():
    """ """

    building_id = "000"
    img_fpaths = glob.glob(f"/Users/johnlam/Downloads/2021_05_28_Will_amazon_raw/{building_id}/panos/*.jpg")

    for img_fpath in img_fpaths:

        render_depth_if_nonexistent(img_fpath)

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


def render_pairs() -> None:
    """ """
    is_semantics = False
    if is_semantics:
        crop_z_above = 2.0
    else:
        crop_z_above = -1.0

    # building_id = "000"
    # floor_id = "floor_02" # "floor_01"

    building_id = "001"
    floor_id = "floor_02" # "floor_01"

    def panoid_from_fpath(fpath: str) -> int:
        """ """
        return int(Path(fpath).stem.split('_')[-1])

    img_fpaths = glob.glob(f"/Users/johnlam/Downloads/2021_05_28_Will_amazon_raw/{building_id}/panos/*.jpg")
    img_fpaths_dict = { panoid_from_fpath(fpath): fpath for fpath in img_fpaths}

    #import pdb; pdb.set_trace()

    dataset_id = "verifier_dataset_2021_06_21"
    floor_labels_dirpath = f"/Users/johnlam/Downloads/jlambert-auto-floorplan/{dataset_id}/{building_id}/{floor_id}"

    gt_exact_pairs =  glob.glob(f"{floor_labels_dirpath}/gt_alignment_exact/*.json")
    gt_approx_pairs = glob.glob(f"{floor_labels_dirpath}/gt_alignment_approx/*.json")
    incorrect_pairs = glob.glob(f"{floor_labels_dirpath}/incorrect_alignment/*.json")

    for pair_fpath in gt_approx_pairs: # gt_exact_pairs:

        i2Ti1 = sim2_from_json(json_fpath=pair_fpath)

        i1, i2 = Path(pair_fpath).stem.split('_')[:2]
        i1, i2 = int(i1), int(i2)

        print(f"On {i1},{i2}")

        img1_fpath = img_fpaths_dict[i1]
        img2_fpath = img_fpaths_dict[i2]

        render_depth_if_nonexistent(img1_fpath)
        render_depth_if_nonexistent(img2_fpath)

        #import pdb; pdb.set_trace()

        args = SimpleNamespace(**{
            "img_i1": semantic_img1_fpath if is_semantics else img1_fpath,
            "img_i2": semantic_img2_fpath if is_semantics else img2_fpath,
            "depth_i1": f"assets/{Path(img1_fpath).stem}.depth.png",
            "depth_i2": f"assets/{Path(img2_fpath).stem}.depth.png",
            "scale": 0.001,
            "crop_ratio": 80/512, # throw away top 80 and bottom 80 rows of pixel (too noisy of estimates)
            "crop_z_above": crop_z_above #0.3 # -1.0 # -0.5 # 0.3 # 1.2
        })
        #bev_img = vis_depth_and_render(args, is_semantics=False)

        render_bev_pair(args, building_id, floor_id, i1, i2, i2Ti1, is_semantics=False)



def sim2_from_json(json_fpath: str) -> Sim2:
    """Generate class inst. from a JSON file containing Sim(2) parameters as flattened matrices (row-major)."""
    import json

    with open(json_fpath, "r") as f:
        json_data = json.load(f)

    R = np.array(json_data["R"]).reshape(2, 2)
    t = np.array(json_data["t"]).reshape(2)
    s = float(json_data["s"])

    print("Identity? ", np.round(R.T @ R, 1))

    return Sim2(R, t, s)


if __name__ == "__main__":
    #render_isolated_examples()


    render_pairs()

