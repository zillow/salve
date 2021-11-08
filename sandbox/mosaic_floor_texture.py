
"""Given known camera poses, mosaic BEV textures into a coherent texture over an entire floor.

References:
- Dellaert99cvpr "Condensation Algorithm"
- Lu and Milios
"""

import glob
import os
from pathlib import Path
from types import SimpleNamespace

import argoverse.utils.json_utils as json_utils
import numpy as np
from argoverse.utils.sim2 import Sim2

import afp.algorithms.spanning_tree as spanning_tree
import afp.utils.bev_rendering_utils as bev_rendering_utils
import afp.utils.hohonet_inference as hohonet_inference_utils
from afp.dataset.zind_partition import OLD_HOME_ID_TEST_SET, NEW_HOME_ID_TEST_SET


def panoid_from_fpath(fpath: str) -> int:
    """Derive panorama's id from its filename."""
    return int(Path(fpath).stem.split("_")[-1])


def render_floor_texture(
    depth_save_root: str,
    bev_save_root: str,
    hypotheses_save_root: str,
    raw_dataset_dir: str,
    building_id: str,
    floor_id: str,
    layout_save_root: str,
) -> None:
    """On a single canvas, texture-map all panoramas onto a single space.

    We generate global poses by using a spanning tree from the approximately generated relative
    poses, using GT WDO locations.

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
                i2Si1_dict[(i1, i2)] = i2Si1

            # find the minimal spanning tree, and the absolute poses from it
            wSi_list = spanning_tree.greedily_construct_st_Sim2(i2Si1_dict)

            xyzrgb = np.zeros((0, 6))
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

                hohonet_inference_utils.infer_depth_if_nonexistent(
                    depth_save_root=depth_save_root, building_id=building_id, img_fpath=img1_fpath
                )
                hohonet_inference_utils.infer_depth_if_nonexistent(
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
                xyzrgb1, xyzrgb2 = bev_rendering_utils.get_bev_pair_xyzrgb(args, building_id, floor_id, i1, i2, i2Ti1, is_semantics=False)

                # import pdb; pdb.set_trace()
                xyzrgb1[:, :2] = wSi_list[i2].transform_from(xyzrgb1[:, :2])
                xyzrgb2[:, :2] = wSi_list[i2].transform_from(xyzrgb2[:, :2])

                xyzrgb = np.vstack([xyzrgb, xyzrgb1])
                xyzrgb = np.vstack([xyzrgb, xyzrgb2])

                import matplotlib.pyplot as plt

                plt.scatter(xyzrgb[:, 0], xyzrgb[:, 1], 10, c=xyzrgb[:, 3:], marker=".", alpha=0.1)
                plt.title("")
                plt.axis("equal")
                save_fpath = f"texture_aggregation/{building_id}/{floor_id}/aggregated_{pair_idx}_pairs.jpg"
                os.makedirs(Path(save_fpath).parent, exist_ok=True)
                plt.savefig(save_fpath, dpi=500)
                # plt.show()
                plt.close("all")



def render_floor_texture_all_buildings(
    num_processes: int,
    depth_save_root: str,
    bev_save_root: str,
    raw_dataset_dir: str,
    hypotheses_save_root: str,
    layout_save_root: str,
) -> None:
    """Render BEV texture maps for all floors of all ZinD buildings.

    We do this 
    """

    # building_id = "000"
    # floor_id = "floor_02" # "floor_01"

    # building_id = "981" # "000"
    # floor_id = "floor_01" # "floor_01"

    # discover possible building ids and floors
    building_ids = [Path(fpath).stem for fpath in glob.glob(f"{raw_dataset_dir}/*") if Path(fpath).is_dir()]
    building_ids.sort()

    args = []

    for building_id in building_ids:

        # # already rendered
        # if building_id not in ["0003"]: # NEW_HOME_ID_TEST_SET:
        #     continue

        json_annot_fpath = f"{raw_dataset_dir}/{building_id}/zind_data.json"
        if not Path(json_annot_fpath).exists():
            print(f"zind_data.json file missing for {building_id}")

        floor_map_json = json_utils.read_json_file(json_annot_fpath)

        if "merger" not in floor_map_json:
            print(f"No merger data in {building_id}: {json_annot_fpath}")
            continue

        merger_data = floor_map_json["merger"]
        for floor_id in merger_data.keys():
            render_floor_texture(
                depth_save_root,
                bev_save_root,
                hypotheses_save_root,
                raw_dataset_dir,
                building_id,
                floor_id,
                layout_save_root,
            )

if __name__ == '__main__':
    """ """

    num_processes = ""
    depth_save_root = ""
    bev_save_root = ""
    raw_dataset_dir = ""
    hypotheses_save_root = ""
    layout_save_root = ""

    render_floor_texture_all_buildings(
        num_processes,
        depth_save_root,
        bev_save_root,
        raw_dataset_dir,
        hypotheses_save_root,
        layout_save_root,
    )
