import glob
import os
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import imageio
import numpy as np
from argoverse.utils.sim2 import Sim2
from argoverse.utils.json_utils import read_json_file

from infer_depth import infer_depth
from vis_depth import vis_depth, vis_depth_and_render, render_bev_pair, get_bev_pair_xyzrgb

from spanning_tree import greedily_construct_st_Sim2

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
    building_id = "000"  # "981"  #
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

        # save_dir = f"/Users/johnlam/Downloads/ZinD_BEV_crop_above_{args.crop_z_above}/{building_id}"
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
        pairs.sort()

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

                i2Ti1 = Sim2.from_json(json_fpath=pair_fpath)

                i1, i2 = Path(pair_fpath).stem.split("_")[:2]
                i1, i2 = int(i1), int(i2)

                # e.g. 'door_0_0_identity'
                pair_uuid = Path(pair_fpath).stem.split('__')[-1]

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

                building_bev_save_dir = f"{bev_save_root}/{label_type}/{building_id}"
                os.makedirs(building_bev_save_dir, exist_ok=True)

                def bev_fpath_from_img_fpath(pair_idx: int, pair_uuid: str, surface_type: str, img_fpath: str) -> None:
                    """ """
                    fname_stem = Path(img_fpath).stem
                    if is_semantics:
                        img_name = f"pair_{pair_idx}___{pair_uuid}_{surface_type}_semantics_{fname_stem}.jpg"
                    else:
                        img_name = f"pair_{pair_idx}___{pair_uuid}_{surface_type}_rgb_{fname_stem}.jpg"
                    return img_name

                bev_fname1 = bev_fpath_from_img_fpath(pair_idx, pair_uuid, surface_type, img1_fpath)
                bev_fname2 = bev_fpath_from_img_fpath(pair_idx, pair_uuid, surface_type, img2_fpath)

                bev_fpath1 = f"{building_bev_save_dir}/{bev_fname1}"
                bev_fpath2 = f"{building_bev_save_dir}/{bev_fname2}"

                if Path(bev_fpath1).exists() and Path(bev_fpath2).exists():
                    print("Both BEV images already exist, skipping...")
                    continue

                bev_img1, bev_img2 = render_bev_pair(args, building_id, floor_id, i1, i2, i2Ti1, is_semantics=False)

                if bev_img1 is None or bev_img2 is None:
                    continue

                imageio.imwrite(bev_fpath1, bev_img1)
                imageio.imwrite(bev_fpath2, bev_img2)


def render_floor_texture(
    depth_save_root: str,
    bev_save_root: str,
    hypotheses_save_root: str,
    raw_dataset_dir: str,
    building_id: str,
    floor_id: str,
) -> None:
    """

    Args:
        depth_save_root: 
        bev_save_root: 
        hypotheses_save_root: 
        raw_dataset_dir: 
        building_id: 
        floor_id: 
    """
    img_fpaths = glob.glob(f"{raw_dataset_dir}/{building_id}/panos/*.jpg")
    img_fpaths_dict = {panoid_from_fpath(fpath): fpath for fpath in img_fpaths}

    floor_labels_dirpath = f"{hypotheses_save_root}/{building_id}/{floor_id}"

    for label_type in ["gt_alignment_approx"]:
        pairs = glob.glob(f"{floor_labels_dirpath}/{label_type}/*.json")

        for surface_type in ["floor", "ceiling"]:

            i2Si1_dict = {}
            for sim2_json_fpath in pairs:
                i1, i2 = Path(sim2_json_fpath).stem.split("_")[:2]
                i1, i2 = int(i1), int(i2)

                i2Si1 = Sim2.from_json(json_fpath=sim2_json_fpath)
                i2Si1_dict[(i1,i2)] = i2Si1

            # find the minimal spanning tree, and the absolute poses from it
            wSi_list = greedily_construct_st_Sim2(i2Si1_dict)


            xyzrgb = np.zeros((0,6))
            for pair_idx, pair_fpath in enumerate(pairs):

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

                i2Ti1 = Sim2.from_json(json_fpath=pair_fpath)

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

                # will be in the frame of 2!
                xyzrgb1, xyzrgb2 = get_bev_pair_xyzrgb(args, building_id, floor_id, i1, i2, i2Ti1, is_semantics=False)

                #import pdb; pdb.set_trace()
                xyzrgb1[:,:2] = wSi_list[i2].transform_from(xyzrgb1[:,:2])
                xyzrgb2[:,:2] = wSi_list[i2].transform_from(xyzrgb2[:,:2])

                xyzrgb = np.vstack([xyzrgb, xyzrgb1])
                xyzrgb = np.vstack([xyzrgb, xyzrgb2])
        
                import matplotlib.pyplot as plt
                plt.scatter(xyzrgb[:, 0], xyzrgb[:, 1], 10, c=xyzrgb[:, 3:], marker=".", alpha=0.1)
                plt.title("")
                plt.axis("equal")
                save_fpath = f"texture_aggregation/{building_id}/{floor_id}/aggregated_{pair_idx}_pairs.jpg"
                os.makedirs(Path(save_fpath).parent, exist_ok=True)
                plt.savefig(save_fpath, dpi=500)
                #plt.show()
                plt.close("all")


def render_pairs(
    num_processes: int, depth_save_root: str, bev_save_root: str, raw_dataset_dir: str, hypotheses_save_root: str
) -> None:
    """ """

    # building_id = "000"
    # floor_id = "floor_02" # "floor_01"

    # building_id = "981" # "000"
    # floor_id = "floor_01" # "floor_01"

    # discover possible building ids and floors
    building_ids = [Path(fpath).stem for fpath in glob.glob(f"{raw_dataset_dir}/*") if Path(fpath).is_dir()]
    building_ids.sort()

    args = []

    for building_id in building_ids:

        # already rendered
        if building_id in ['1635', '1584', '1583', '1578', '1530', '1490', '1442', '1626', '1427', '1394']:
            continue

        json_annot_fpath = f"{raw_dataset_dir}/{building_id}/zfm_data.json"
        floor_map_json = read_json_file(json_annot_fpath)

        if "merger" not in floor_map_json:
            print(f"No merger data in {building_id}: {json_annot_fpath}")
            continue

        merger_data = floor_map_json["merger"]
        for floor_id, floor_data in merger_data.items():
            args += [(depth_save_root, bev_save_root, hypotheses_save_root, raw_dataset_dir, building_id, floor_id)]

    if num_processes > 1:
        with Pool(num_processes) as p:
            p.starmap(render_building_floor_pairs, args)
    else:
        for single_call_args in args:
            render_building_floor_pairs(*single_call_args)
            #render_floor_texture(*single_call_args)


if __name__ == "__main__":
    # render_isolated_examples()

    num_processes = 20

    #depth_save_root = "/Users/johnlam/Downloads/HoHoNet_Depth_Maps"
    depth_save_root = "/mnt/data/johnlam/HoHoNet_Depth_Maps"

    # hypotheses_save_root = "/Users/johnlam/Downloads/jlambert-auto-floorplan/verifier_dataset_2021_06_21"
    # hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_alignment_hypotheses_2021_06_25"
    # hypotheses_save_root = "/mnt/data/johnlam/ZinD_alignment_hypotheses_2021_06_25"
    # hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_alignment_hypotheses_2021_07_07"
    #hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_alignment_hypotheses_2021_07_14_w_wdo_idxs"
    #hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_alignment_hypotheses_2021_07_14_v2_w_wdo_idxs"
    #hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_alignment_hypotheses_2021_07_14_v3_w_wdo_idxs"
    hypotheses_save_root = "/mnt/data/johnlam/ZinD_alignment_hypotheses_2021_07_14_v3_w_wdo_idxs"

    # raw_dataset_dir = "/Users/johnlam/Downloads/2021_05_28_Will_amazon_raw"
    # raw_dataset_dir = "/Users/johnlam/Downloads/ZInD_release/complete_zind_paper_final_localized_json_6_3_21"
    raw_dataset_dir = "/mnt/data/johnlam/ZInD_release/complete_zind_paper_final_localized_json_6_3_21"

    # bev_save_root = "/Users/johnlam/Downloads/ZinD_BEV_2021_06_24"
    # bev_save_root = "/Users/johnlam/Downloads/ZinD_BEV_RGB_only_2021_06_25"
    #bev_save_root = "/mnt/data/johnlam/ZinD_BEV_RGB_only_2021_06_25"
    #bev_save_root = "/Users/johnlam/Downloads/ZinD_BEV_RGB_only_2021_07_14_v2"
    bev_save_root = "/mnt/data/johnlam/ZinD_BEV_RGB_only_2021_07_14_v3"


    # render_dataset(bev_save_root, raw_dataset_dir)
    render_pairs(
        num_processes=num_processes,
        depth_save_root=depth_save_root,
        bev_save_root=bev_save_root,
        raw_dataset_dir=raw_dataset_dir,
        hypotheses_save_root=hypotheses_save_root,
    )
