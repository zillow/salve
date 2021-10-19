
"""
Support HoHoNet batched inference over ZinD.
"""

import argparse
import glob
import importlib
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List

import afp.utils.hohonet_inference as hohonet_inference_utils
import imageio
import numpy as np
import torch
from afp.utils.hohonet_inference import HOHONET_CONFIG_FPATH, HOHONET_CKPT_FPATH
from tqdm import tqdm

from lib.config import config, update_config


def infer_depth_over_image_list(args: SimpleNamespace, image_fpaths: List[str]):
    """
    Note: Depth map only created if nonexistent.

    We load the model into memory only once, instead of doing so for every single individual input image.

    Args:
        args: must contain variable `building_depth_save_dir`
    """
    update_config(config, args)
    device = 'cuda' # if config.cuda else 'cpu'

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
            x = torch.from_numpy(rgb).permute(2,0,1)[None].float() / 255.
            if x.shape[2:] != config.dataset.common_kwargs.hw:
                x = torch.nn.functional.interpolate(x, config.dataset.common_kwargs.hw, mode='area')
            x = x.to(device)
            pred_depth = net.infer(x)
            if not torch.is_tensor(pred_depth):
                pred_depth = pred_depth.pop('depth')

            fname = os.path.splitext(os.path.split(path)[1])[0]
            imageio.imwrite(
                os.path.join(args.out, f'{fname}.depth.png'),
                pred_depth.mul(1000).squeeze().cpu().numpy().astype(np.uint16)
            )

            visualize = False
            if visualize:
                import matplotlib.pyplot as plt
                plt.imshow( pred_depth.mul(1000).squeeze().cpu().numpy().astype(np.uint16) )
                plt.show()


def infer_depth_over_single_zind_tour(
    depth_save_root: str,
    raw_dataset_dir: str,
    building_id: str,
) -> None:
    """

    Args:
        depth_save_root: directory where depth maps should be saved.
        raw_dataset_dir: path to ZinD dataset.
        building_id: unique ID of ZinD building.
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
    """ """
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


if __name__ == "__main__":
    """ """
    num_processes = 10

    #depth_save_root = "/Users/johnlam/Downloads/ZinD_Bridge_API_HoHoNet_Depth_Maps"
    ## depth_save_root = "/mnt/data/johnlam/ZinD_Bridge_API_HoHoNet_Depth_Maps"
    depth_save_root = "/data/johnlam/ZinD_Bridge_API_HoHoNet_Depth_Maps" # on se1-rmx-gpu-002

    raw_dataset_dir = "/data/johnlam/zind_bridgeapi_2021_10_05"

    # render_dataset(bev_save_root, raw_dataset_dir)
    infer_depth_over_all_zind_tours(
        num_processes=num_processes,
        depth_save_root=depth_save_root,
        raw_dataset_dir=raw_dataset_dir,
    )
