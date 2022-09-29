"""Render aligned texture maps and/or layouts in a bird's eye view, to be used for SALVe.

These texture maps can be used either for training the SALVe verifier network, or for inference
with a pre-trained model.

The rendering can be parallelized if `num_processes` is set to >1, in which case each process renders
BEV texture maps for individual homes.
"""

import glob
import os
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import click
import imageio

import salve.common.posegraph2d as posegraph2d
import salve.dataset.hnet_prediction_loader as hnet_prediction_loader
import salve.utils.bev_rendering_utils as bev_rendering_utils
import salve.utils.hohonet_inference as hohonet_inference_utils
from salve.common.sim2 import Sim2
from salve.dataset.zind_partition import DATASET_SPLITS


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
    layout_save_root: Optional[str],
    render_modalities: List[str],
) -> None:
    """Render BEV texture maps for a single floor of a single ZinD building.

    Given a set of possible alignment hypotheses for the floor, render all possible BEV floor-ceiling image pairs.

    Args:
        depth_save_root: directory where depth maps should be saved (or are already cached here).
        bev_save_root: directory where bird's eye view texture maps should be saved.
        hypotheses_save_root: directory where putative alignment hypotheses are saved.
        raw_dataset_dir: path to ZInD dataset.
        building_id: unique ID of ZInD building.
        floor_id: unique ID of floor.
        layout_save_root: if rendering rasterized layout, all images will be saved under this directory.
        render_modalities: type of BEV rendering to generate (either "rgb_texture" or "layout", or both).
    """
    if "layout" in render_modalities:
        # Load the layouts, either inferred or GT.
        use_inferred_wdos_layout = True
        if use_inferred_wdos_layout:
            floor_pose_graphs = hnet_prediction_loader.load_inferred_floor_pose_graphs(
                building_id=building_id, raw_dataset_dir=raw_dataset_dir
            )
            if floor_pose_graphs is None:
                return
            floor_pose_graph = floor_pose_graphs[floor_id]
        else:
            floor_pose_graph = posegraph2d.get_gt_pose_graph(building_id, floor_id, raw_dataset_dir)

    img_fpaths = glob.glob(f"{raw_dataset_dir}/{building_id}/panos/*.jpg")
    img_fpaths_dict = {panoid_from_fpath(fpath): fpath for fpath in img_fpaths}

    floor_labels_dirpath = f"{hypotheses_save_root}/{building_id}/{floor_id}"

    for label_type in ["gt_alignment_approx", "incorrect_alignment"]:  # "gt_alignment_exact"
        pairs = glob.glob(f"{floor_labels_dirpath}/{label_type}/*.json")
        pairs.sort()
        print(f"On Building {building_id}, {floor_id}, {label_type}")

        for pair_idx, pair_fpath in enumerate(pairs):

            for surface_type in ["floor", "ceiling"]:
                is_semantics = False
                # if is_semantics:
                #     crop_z_range = [-float('inf'), 2.0]
                # else:

                if surface_type == "floor":
                    # Keep everything 1 meter and below the camera
                    crop_z_range = [-float("inf"), -1.0]

                elif surface_type == "ceiling":
                    # Keep everything 50 cm and above the camera.
                    crop_z_range = [0.5, float("inf")]

                i2Ti1 = Sim2.from_json(json_fpath=pair_fpath)

                i1, i2 = Path(pair_fpath).stem.split("_")[:2]
                i1, i2 = int(i1), int(i2)

                # e.g. 'door_0_0_identity'
                pair_uuid = Path(pair_fpath).stem.split("__")[-1]

                img1_fpath = img_fpaths_dict[i1]
                img2_fpath = img_fpaths_dict[i2]

                building_bev_save_dir = f"{bev_save_root}/{label_type}/{building_id}"
                os.makedirs(building_bev_save_dir, exist_ok=True)

                def bev_fname_from_img_fpath(pair_idx: int, pair_uuid: str, surface_type: str, img_fpath: str) -> None:
                    """Generate file name for BEV texture map based on panorama image file path."""
                    fname_stem = Path(img_fpath).stem
                    if is_semantics:
                        img_name = f"pair_{pair_idx}___{pair_uuid}_{surface_type}_semantics_{fname_stem}.jpg"
                    else:
                        img_name = f"pair_{pair_idx}___{pair_uuid}_{surface_type}_rgb_{fname_stem}.jpg"
                    return img_name

                bev_fname1 = bev_fname_from_img_fpath(pair_idx, pair_uuid, surface_type, img1_fpath)
                bev_fname2 = bev_fname_from_img_fpath(pair_idx, pair_uuid, surface_type, img2_fpath)

                bev_fpath1 = f"{building_bev_save_dir}/{bev_fname1}"
                bev_fpath2 = f"{building_bev_save_dir}/{bev_fname2}"

                if "rgb_texture" in render_modalities:
                    print(f"On {i1},{i2}")
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
                            # Throw away top 80 and bottom 80 rows of pixel (too noisy of estimates in these regions).
                            "crop_ratio": 80 / 512,
                            "crop_z_range": crop_z_range,  # 0.3 # -1.0 # -0.5 # 0.3 # 1.2
                        }
                    )
                    # bev_img = bev_rendering_utils.vis_depth_and_render(args, is_semantics=False)

                    if Path(bev_fpath1).exists() and Path(bev_fpath2).exists():
                        print("Both BEV images already exist, skipping...")
                        continue

                    bev_img1, bev_img2 = bev_rendering_utils.render_bev_pair(
                        args, building_id, floor_id, i1, i2, i2Ti1, is_semantics=False
                    )

                    if bev_img1 is None or bev_img2 is None:
                        continue

                    imageio.imwrite(bev_fpath1, bev_img1)
                    imageio.imwrite(bev_fpath2, bev_img2)


                if "layout" not in render_modalities:
                    continue

                # Rasterized-layout rendering below (skipped if only generating texture maps).
                # We only rasterize layout for `floor', not for `ceiling`.
                # If `ceiling', we'll skip, since will be identical rendering to `floor`.
                if surface_type != "floor":
                    continue

                building_layout_save_dir = f"{layout_save_root}/{label_type}/{building_id}"
                os.makedirs(building_layout_save_dir, exist_ok=True)

                # Change to layout directory.
                layout_fpath1 = f"{building_layout_save_dir}/{bev_fname1}"
                layout_fpath2 = f"{building_layout_save_dir}/{bev_fname2}"

                if Path(layout_fpath1).exists() and Path(layout_fpath2).exists():
                    print("Both layout images already exist, skipping...")
                    continue

                # Skip for ceiling, since would be duplicate.
                layoutimg1, layoutimg2 = bev_rendering_utils.rasterize_room_layout_pair(
                    i2Ti1=i2Ti1,
                    floor_pose_graph=floor_pose_graph,
                    building_id=building_id,
                    floor_id=floor_id,
                    i1=i1,
                    i2=i2,
                )
                imageio.imwrite(layout_fpath1, layoutimg1)
                imageio.imwrite(layout_fpath2, layoutimg2)


def render_pairs(
    num_processes: int,
    depth_save_root: str,
    bev_save_root: str,
    raw_dataset_dir: str,
    hypotheses_save_root: str,
    layout_save_root: Optional[str],
    render_modalities: List[str],
    split: Optional[str],
    building_id: Optional[str],
) -> None:
    """Render BEV texture maps for all floors of all ZinD buildings.

    Args:
        num_processes: number of processes to use for parallel rendering.
        depth_save_root: directory where depth maps should be saved (or are already cached here).
        bev_save_root: directory where bird's eye view texture maps should be saved.
        raw_dataset_dir: path to ZinD dataset.
        hypotheses_save_root: directory where putative alignment hypotheses are saved.
        layout_save_root: Directory where BEV rendered layouts (instead of RGB texture maps) should be saved to.
            Only applicable when generating aligned BEV layouts intead of aligned texture maps (layout-only baseline).
        render_modalities:
        split: ZInD dataset split to generate BEV texture maps or layouts for.
        building_id: Unique ID of ZInD building to generate renderings for (`split` will be ignored in this case).
    """
    if building_id is not None and split is not None:
        raise ValueError("Either `split` or `building_id` should be provided, but not both.")

    if split is not None:
        # Generate renderings for all buildings in the split.
        building_ids = DATASET_SPLITS[split]
        building_ids.sort()
    else:
        # Generate renderings for only one specific building.
        building_ids = [building_id]

    args = []

    for building_id in building_ids:
        if building_id == "1348":
            # This ZInD building has two panoramas with identical ID, which breaks assumptions above.
            continue

        floor_ids = posegraph2d.compute_available_floors_for_building(
            building_id=building_id, raw_dataset_dir=raw_dataset_dir
        )
        for floor_id in floor_ids:
            args += [
                (
                    depth_save_root,
                    bev_save_root,
                    hypotheses_save_root,
                    raw_dataset_dir,
                    building_id,
                    floor_id,
                    layout_save_root,
                    render_modalities,
                )
            ]

    if num_processes > 1:
        with Pool(num_processes) as p:
            p.starmap(render_building_floor_pairs, args)
    else:
        for single_call_args in args:
            render_building_floor_pairs(*single_call_args)


@click.command(help="Script to render BEV texture maps for each feasible alignment hypothesis.")
@click.option(
    "--raw_dataset_dir",
    type=click.Path(exists=True),
    required=True,
    help="Path to where ZInD dataset is stored on disk (after download from Bridge API).",
)
@click.option("--num_processes", type=int, default=15, help="Number of processes to use for parallel rendering.")
@click.option(
    "--depth_save_root",
    type=str,
    required=True,
    help="Path to where depth maps are stored (or will be saved to, if not computed yet).",
)
@click.option(
    "--hypotheses_save_root",
    type=click.Path(exists=True),
    required=True,
    help="Path to where alignment hypotheses are saved on disk.",
)
@click.option(
    "--bev_save_root",
    type=str,
    required=True,
    help="Directory where BEV texture maps should be written to (directory will be created at this path).",
)
@click.option(
    "--split",
    type=click.Choice(["train", "val", "test"]),
    default=None,
    help="ZInD dataset split to generate BEV texture maps or layouts for.",
)
@click.option(
    "--layout_save_root",
    type=str,
    default=None,
    help="Directory where BEV rendered layouts (instead of RGB texture maps) should be saved to."
    "Only applicable when generating aligned, rasterized BEV layouts intead of aligned texture maps"
    "(corresponding to the layout-only baseline).",
)
@click.option(
    "--building_id",
    type=str,
    required=False,
    default=None,
    help="Unique ID of ZInD building to generate renderings for (`--split` will be ignored in this case).",
)
def run_render_dataset_bev(
    raw_dataset_dir: str,
    num_processes: int,
    depth_save_root: str,
    hypotheses_save_root: str,
    bev_save_root: str,
    split: Optional[str],
    layout_save_root: Optional[str],
    building_id: Optional[str]
) -> None:
    """Click entry point for BEV texture map or layout rendering."""
    if layout_save_root is None:
        render_modalities = ["rgb_texture"]
    else:
        render_modalities = ["layout"]

    render_pairs(
        num_processes=num_processes,
        depth_save_root=depth_save_root,
        bev_save_root=bev_save_root,
        raw_dataset_dir=raw_dataset_dir,
        hypotheses_save_root=hypotheses_save_root,
        layout_save_root=layout_save_root,
        render_modalities=render_modalities,
        split=split,
        building_id=building_id,
    )


if __name__ == "__main__":
    """ """
    run_render_dataset_bev()
