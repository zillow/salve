"""Render aligned texture maps and/or layouts in a bird's eye view, to be used for SALVe.

These texture maps can be used either for training the SALVe verifier network, or for inference
with a pre-trained model.

The rendering can be parallelized if `num_processes` is set to >1. There are two options -- either
each process renders BEV texture maps for individual homes, or each process renders BEV texture maps
for different panorama pairs for a **single home**.
"""

import glob
import os
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import click
import imageio

import salve.common.posegraph2d as posegraph2d
import salve.dataset.hnet_prediction_loader as hnet_prediction_loader
import salve.utils.bev_rendering_utils as bev_rendering_utils
from salve.common.sim2 import Sim2
from salve.dataset.zind_partition import DATASET_SPLITS
from salve.common.posegraph2d import PoseGraph2d


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
    multiprocess_building_panos: bool,
    num_processes: int,
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
        multiprocess_building_panos: Whether to apply multiprocessing within a single building (i.e. multiprocess
            the pano pairs for a single building when True), or instead across buildings (one process per building
            when False).
        num_processes: number of processes to use for rendering pairs from this building.
    """
    if "layout" in render_modalities:
        # Load the layouts that we will render (either inferred or GT layout).
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
    else:
        floor_pose_graph = None

    img_fpaths = glob.glob(f"{raw_dataset_dir}/{building_id}/panos/*.jpg")
    img_fpaths_dict = {panoid_from_fpath(fpath): fpath for fpath in img_fpaths}

    floor_labels_dirpath = f"{hypotheses_save_root}/{building_id}/{floor_id}"

    args = []

    for label_type in ["gt_alignment_approx", "incorrect_alignment"]:  # "gt_alignment_exact"
        pairs = glob.glob(f"{floor_labels_dirpath}/{label_type}/*.json")
        pairs.sort()
        print(f"On Building {building_id}, {floor_id}, {label_type}")

        for pair_idx, pair_fpath in enumerate(pairs):
            for surface_type in ["floor", "ceiling"]:

                args += [
                    (
                        img_fpaths_dict,
                        surface_type,
                        pair_fpath,
                        pair_idx,
                        label_type,
                        bev_save_root,
                        building_id,
                        floor_id,
                        depth_save_root,
                        render_modalities,
                        layout_save_root,
                        floor_pose_graph,
                    )
                ]

    if multiprocess_building_panos and num_processes > 1:
        with Pool(num_processes) as p:
            p.starmap(generate_texture_maps_for_pair, args)

    else:
        for single_call_args in args:
            generate_texture_maps_for_pair(*single_call_args)


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
    multiprocess_building_panos: bool,
) -> None:
    """Render BEV texture maps for all floors of all ZInD buildings.

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
        multiprocess_building_panos: Whether to apply multiprocessing within a single building (i.e. multiprocess the
            pano pairs for a single building when True), or instead across buildings (one process per building
            when False).
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
                    multiprocess_building_panos,
                    num_processes if multiprocess_building_panos else 1,
                )
            ]

    if not multiprocess_building_panos and num_processes > 1:
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
@click.option(
    "--multiprocess_building_panos",
    type=bool,
    default=True,
    help="Whether to apply multiprocessing within a single building (i.e. multiprocess the pano pairs"
    " for a single building when True), or instead across buildings (one process per building when False).",
)
def run_render_dataset_bev(
    raw_dataset_dir: str,
    num_processes: int,
    depth_save_root: str,
    hypotheses_save_root: str,
    bev_save_root: str,
    split: Optional[str],
    layout_save_root: Optional[str],
    building_id: Optional[str],
    multiprocess_building_panos: bool,
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
        multiprocess_building_panos=multiprocess_building_panos,
    )


if __name__ == "__main__":
    """ """
    run_render_dataset_bev()
