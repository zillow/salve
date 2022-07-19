"""Script to perform batched depth map inference with a pre-trained HoHoNet model over ZInD."""

import glob
import importlib
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace
from typing import List

import click
import imageio
import numpy as np
import torch
from tqdm import tqdm

import salve.utils.hohonet_inference as hohonet_inference_utils
from salve.utils.hohonet_inference import HOHONET_CONFIG_FPATH, HOHONET_CKPT_FPATH

from lib.config import config, update_config


def infer_depth_over_image_list(args: SimpleNamespace, image_fpaths: List[str]) -> None:
    """Infer depth maps for each input panorama provided.

    Note: Depth map only created if nonexistent.

    We load the model into memory only once, instead of doing so for every single individual input image.
    Depth map format follows the official HoHoNet inference script here:
    https://github.com/sunset1995/HoHoNet/blob/master/infer_depth.py

    Args:
        args: HoHoNet specific config. Must contain variable `building_depth_save_dir`.
    """
    update_config(config, args)
    device = "cuda"  # if config.cuda else 'cpu'

    # Init model
    model_file = importlib.import_module(config.model.file)
    model_class = getattr(model_file, config.model.modelclass)
    net = model_class(**config.model.kwargs)
    net.load_state_dict(torch.load(args.pth, map_location=device))
    net = net.eval().to(device)

    # Run inference
    with torch.no_grad():
        for path in tqdm(image_fpaths):

            fname_stem = Path(path).stem
            if Path(f"{args.out}/{fname_stem}.depth.png").exists():
                return

            rgb = imageio.imread(path)
            x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255.0
            if x.shape[2:] != config.dataset.common_kwargs.hw:
                x = torch.nn.functional.interpolate(x, config.dataset.common_kwargs.hw, mode="area")
            x = x.to(device)
            pred_depth = net.infer(x)
            if not torch.is_tensor(pred_depth):
                pred_depth = pred_depth.pop("depth")

            fname = os.path.splitext(os.path.split(path)[1])[0]
            imageio.imwrite(
                os.path.join(args.out, f"{fname}.depth.png"),
                pred_depth.mul(1000).squeeze().cpu().numpy().astype(np.uint16),
            )

            visualize = False
            if visualize:
                import matplotlib.pyplot as plt

                plt.imshow(pred_depth.mul(1000).squeeze().cpu().numpy().astype(np.uint16))
                plt.show()


def infer_depth_over_single_zind_tour(
    depth_save_root: str,
    raw_dataset_dir: str,
    building_id: str,
) -> None:
    """Launch batched depth map inference on a single ZInD tour (building).

    Args:
        depth_save_root: directory where depth maps should be saved.
        raw_dataset_dir: path to ZInD dataset.
        building_id: unique ID of ZInD building.
    """
    building_depth_save_dir = f"{depth_save_root}/{building_id}"
    os.makedirs(building_depth_save_dir, exist_ok=True)

    img_fpaths = glob.glob(f"{raw_dataset_dir}/{building_id}/panos/*.jpg")

    args = SimpleNamespace(
        **{
            "cfg": HOHONET_CONFIG_FPATH,
            "pth": HOHONET_CKPT_FPATH,
            "out": building_depth_save_dir,
            "opts": [],
        }
    )
    infer_depth_over_image_list(args, image_fpaths=img_fpaths)


def infer_depth_over_all_zind_tours(
    num_processes: int,
    depth_save_root: str,
    raw_dataset_dir: str,
) -> None:
    """Launch parallel depth map inference on all tours.

    Args:
       num_processes: path to where ZInD dataset is stored on disk (after download from Bridge API).
       depth_save_root: path to where depth maps are stored (and will be saved to, if not computed yet).
       raw_dataset_dir: number of GPU processes to use for batched inference.
    """
    # discover possible building ids and floors
    building_ids = [Path(fpath).stem for fpath in glob.glob(f"{raw_dataset_dir}/*") if Path(fpath).is_dir()]
    building_ids.sort()
    args = []

    for building_id in building_ids:
        args += [(depth_save_root, raw_dataset_dir, building_id)]

    if num_processes > 1:
        with Pool(num_processes) as p:
            p.starmap(infer_depth_over_single_zind_tour, args)
    else:
        for single_call_args in args:
            infer_depth_over_single_zind_tour(*single_call_args)


@click.command(help="Script to run batched depth map inference using a pretrained HoHoNet model.")
@click.option(
    "--raw_dataset_dir",
    type=click.Path(exists=True),
    required=True,
    help="Path to where ZInD dataset is stored on disk (after download from Bridge API).",
)
@click.option(
    "--depth_save_root",
    type=str,
    required=True,
    help="Path to where depth maps are stored (and will be saved to, if not computed yet).",
)
@click.option("--num_processes", type=int, default=2, help="Number of GPU processes to use for batched inference.")
def run_batched_depth_map_inference(raw_dataset_dir: str, depth_save_root: str, num_processes: int) -> None:
    """Click entry point for batched depth map inference."""
    # render_dataset(bev_save_root, raw_dataset_dir)
    infer_depth_over_all_zind_tours(
        num_processes=num_processes,
        depth_save_root=depth_save_root,
        raw_dataset_dir=raw_dataset_dir,
    )


if __name__ == "__main__":
    """ """
    run_batched_depth_map_inference()
