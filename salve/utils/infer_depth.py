"""Infer a depth map using a HoHoNet model.

From HoHoNet: https://github.com/sunset1995/HoHoNet/blob/master/infer_depth.py
Style is modified, but functionality unchanged.
"""

import argparse
import glob
import importlib
import os
import sys
from types import SimpleNamespace
from typing import Union

import numpy as np
import torch
from imageio import imread, imwrite
from tqdm import tqdm


try:
    from lib.config import config, update_config
except Exception as e:
    print("HoHoNet's lib could not be loaded, skipping...")
    print("Exception: ", e)


def infer_depth(args: Union[SimpleNamespace, argparse.Namespace]) -> None:
    """Infer monocular depth map and write as PNG to disk."""
    update_config(config, args)
    device = "cuda" if config.cuda and torch.cuda.is_available() else "cpu"

    # Parse input paths
    rgb_lst = glob.glob(args.inp)
    if len(rgb_lst) == 0:
        print("No images found")
        sys.exit()

    # Init model
    model_file = importlib.import_module(config.model.file)
    model_class = getattr(model_file, config.model.modelclass)
    net = model_class(**config.model.kwargs)
    net.load_state_dict(torch.load(args.pth, map_location=device))
    net = net.eval().to(device)

    # Run inference
    with torch.no_grad():
        for path in tqdm(rgb_lst):
            rgb = imread(path)
            x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255.0
            if x.shape[2:] != config.dataset.common_kwargs.hw:
                x = torch.nn.functional.interpolate(x, config.dataset.common_kwargs.hw, mode="area")
            x = x.to(device)
            pred_depth = net.infer(x)
            if not torch.is_tensor(pred_depth):
                pred_depth = pred_depth.pop("depth")

            fname = os.path.splitext(os.path.split(path)[1])[0]
            imwrite(
                os.path.join(args.out, f"{fname}.depth.png"),
                pred_depth.mul(1000).squeeze().cpu().numpy().astype(np.uint16),
            )

            visualize = False
            if visualize:
                import matplotlib.pyplot as plt

                plt.imshow(pred_depth.mul(1000).squeeze().cpu().numpy().astype(np.uint16))
                plt.show()


if __name__ == "__main__":

    # Parse args & config
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--pth", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--inp", required=True)
    parser.add_argument(
        "opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER
    )
    args = parser.parse_args()

    infer_depth(args)
