import glob
import os
from pathlib import Path
from types import SimpleNamespace

import imageio
import numpy as np
from argoverse.utils.sim2 import Sim2
from argoverse.utils.json_utils import read_json_file

from infer_depth import infer_depth
from vis_depth import vis_depth, vis_depth_and_render, render_bev_pair


HOHONET_CONFIG_FPATH = "config/mp3d_depth/HOHO_depth_dct_efficienthc_TransEn1_hardnet.yaml"
HOHONET_CKPT_FPATH = "ckpt/mp3d_depth_HOHO_depth_dct_efficienthc_TransEn1_hardnet/ep60.pth"


def infer_depth_if_nonexistent(depth_save_root: str, building_id: str, img_fpath: str) -> None:
    """ """
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


# nasty failure cases:
DISCARD_DICT = {
    # (building, pano_ids)
    "000": [10],  # pano 10 outside
    "004": [10, 24, 28, 56, 58],  # building 004, pano 10 and pano 24, pano 28,56,58 (outdoors)
    "006": [10],
    "981": [5, 7, 14, 11, 16, 17, 28],  # 11 is a bit inaccurate
}


def render_dataset(bev_save_root: str, raw_dataset_dir: str) -> None:
    """ """
    building_id = "000" # "981"  # 
    # building_id = "004"

    img_fpaths = glob.glob(f"{raw_dataset_dir}/{building_id}/panos/*.jpg")

    for img_fpath in img_fpaths:

        if "_14.jpg" not in img_fpath:  # 13
            continue

        infer_depth_if_nonexistent(depth_save_root, building_id, img_fpath)

        is_semantics = False
        semantic_img_fpath = f"/Users/johnlam/Downloads/MSeg_output/{Path(img_fpath).stem}_gray.jpg"

        if is_semantics:
            crop_z_range = [-float("inf"), 2]
        else:
            crop_z_range = [-float("inf"), -1.0]  # [0.5, float('inf')] #  # 2.0 # -1.0

        args = SimpleNamespace(
            **{
                "img": semantic_img_fpath if is_semantics else img_fpath,
                "depth": f"{depth_save_root}/{building_id}/{Path(img_fpath).stem}.depth.png",
                "scale": 0.001,
                "crop_ratio": 80 / 512,  # throw away top 80 and bottom 80 rows of pixel (too noisy of estimates)
                # "crop_z_range": crop_z_range #0.3 # -1.0 # -0.5 # 0.3 # 1.2
                "crop_z_above": 2,
            }
        )
        import pdb

        pdb.set_trace()
        bev_img = vis_depth_and_render(args, is_semantics=False)

        vis_depth(args)

        #save_dir = f"/Users/johnlam/Downloads/ZinD_BEV_crop_above_{args.crop_z_above}/{building_id}"
        building_bev_save_dir = f"{bev_save_root}/{label_type}/{building_id}"
        fname_stem = Path(img_fpath).stem
        os.makedirs(save_dir, exist_ok=True)
        if is_semantics:
            img_name = f"semantics_{fname_stem}.jpg"
        else:
            img_name = f"{fname_stem}.jpg"
        imageio.imwrite(f"{building_bev_save_dir}/{img_name}", bev_img)


def panoid_from_fpath(fpath: str) -> int:
    """Derive panorama's id from its filename."""
    return int(Path(fpath).stem.split("_")[-1])


def render_building_floor_pairs(
    depth_save_root: str,
    bev_save_root: str,
    hypotheses_save_root: str,
    raw_dataset_dir: str,
    building_id: str,
    floor_id: str,
) -> None:
    """ """
    img_fpaths = glob.glob(f"{raw_dataset_dir}/{building_id}/panos/*.jpg")
    img_fpaths_dict = {panoid_from_fpath(fpath): fpath for fpath in img_fpaths}

    floor_labels_dirpath = f"{hypotheses_save_root}/{building_id}/{floor_id}"

    for label_type in ["gt_alignment_approx", "incorrect_alignment"]:  # "gt_alignment_exact"
        pairs = glob.glob(f"{floor_labels_dirpath}/{label_type}/*.json")

        for pair_idx, pair_fpath in enumerate(pairs):

            for surface_type in ["floor", "ceiling"]:
                is_semantics = False
                # if is_semantics:
                #     crop_z_range = [-float('inf'), 2.0]
                # else:

                if surface_type == "floor":
                    # everything 1 meter and below the camera
                    crop_z_range = [-float("inf"), -1.0]

                elif surface_type == "ceiling":
                    # everything 50 cm and above camera
                    crop_z_range = [0.5, float("inf")]

                i2Ti1 = sim2_from_json(json_fpath=pair_fpath)

                i1, i2 = Path(pair_fpath).stem.split("_")[:2]
                i1, i2 = int(i1), int(i2)

                print(f"On {i1},{i2}")

                img1_fpath = img_fpaths_dict[i1]
                img2_fpath = img_fpaths_dict[i2]

                infer_depth_if_nonexistent(
                    depth_save_root=depth_save_root, building_id=building_id, img_fpath=img1_fpath
                )
                infer_depth_if_nonexistent(
                    depth_save_root=depth_save_root, building_id=building_id, img_fpath=img2_fpath
                )

                args = SimpleNamespace(
                    **{
                        "img_i1": semantic_img1_fpath if is_semantics else img1_fpath,
                        "img_i2": semantic_img2_fpath if is_semantics else img2_fpath,
                        "depth_i1": f"{depth_save_root}/{building_id}/{Path(img1_fpath).stem}.depth.png",
                        "depth_i2": f"{depth_save_root}/{building_id}/{Path(img2_fpath).stem}.depth.png",
                        "scale": 0.001,
                        # throw away top 80 and bottom 80 rows of pixel (too noisy of estimates)
                        "crop_ratio": 80 / 512,
                        "crop_z_range": crop_z_range,  # 0.3 # -1.0 # -0.5 # 0.3 # 1.2
                    }
                )
                # bev_img = vis_depth_and_render(args, is_semantics=False)

                bev_img1, bev_img2 = render_bev_pair(args, building_id, floor_id, i1, i2, i2Ti1, is_semantics=False)

                building_bev_save_dir = f"{bev_save_root}/{label_type}/{building_id}"
                os.makedirs(building_bev_save_dir, exist_ok=True)

                for img_fpath, bev_img in zip([img1_fpath, img2_fpath], [bev_img1, bev_img2]):
                    if is_semantics:
                        img_name = f"pair_{pair_idx}_{surface_type}_semantics_{Path(img_fpath).stem}.jpg"
                    else:
                        img_name = f"pair_{pair_idx}_{surface_type}_rgb_{Path(img_fpath).stem}.jpg"
                    imageio.imwrite(f"{building_bev_save_dir}/{img_name}", bev_img)


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


def render_pairs(depth_save_root: str, bev_save_root: str, raw_dataset_dir: str, hypotheses_save_root: str) -> None:
    """ """

    # building_id = "000"
    # floor_id = "floor_02" # "floor_01"

    # building_id = "981" # "000"
    # floor_id = "floor_01" # "floor_01"

    # discover possible building ids and floors
    building_ids = [Path(fpath).stem for fpath in glob.glob(f"{raw_dataset_dir}/*") if Path(fpath).is_dir()]
    building_ids.sort()

    for building_id in building_ids:
        json_annot_fpath = f"{raw_dataset_dir}/{building_id}/zfm_data.json"
        floor_map_json = read_json_file(json_annot_fpath)
        merger_data = floor_map_json["merger"]
        for floor_id, floor_data in merger_data.items():
            render_building_floor_pairs(
                depth_save_root=depth_save_root,
                bev_save_root=bev_save_root,
                hypotheses_save_root=hypotheses_save_root,
                raw_dataset_dir=raw_dataset_dir,
                building_id=building_id,
                floor_id=floor_id,
            )


if __name__ == "__main__":
    # render_isolated_examples()

    #depth_save_root = "/Users/johnlam/Downloads/HoHoNet_Depth_Maps"
    depth_save_root = "/mnt/data/johnlam/HoHoNet_Depth_Maps"

    # hypotheses_save_root = "/Users/johnlam/Downloads/jlambert-auto-floorplan/verifier_dataset_2021_06_21"
    #hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_alignment_hypotheses_2021_06_25"
    hypotheses_save_root = "/mnt/data/johnlam/ZinD_alignment_hypotheses_2021_06_25"

    #raw_dataset_dir = "/Users/johnlam/Downloads/2021_05_28_Will_amazon_raw"
    # raw_dataset_dir = "/Users/johnlam/Downloads/ZInD_release/complete_zind_paper_final_localized_json_6_3_21"
    raw_dataset_dir = "/mnt/data/johnlam/ZInD_release/complete_zind_paper_final_localized_json_6_3_21"

    # bev_save_root = "/Users/johnlam/Downloads/ZinD_BEV_2021_06_24"
    #bev_save_root = "/Users/johnlam/Downloads/ZinD_BEV_RGB_only_2021_06_25"
    bev_save_root = "/mnt/data/johnlam/ZinD_BEV_RGB_only_2021_06_25"


    #render_dataset(bev_save_root, raw_dataset_dir)
    render_pairs(
        depth_save_root=depth_save_root,
        bev_save_root=bev_save_root,
        raw_dataset_dir=raw_dataset_dir,
        hypotheses_save_root=hypotheses_save_root,
    )
