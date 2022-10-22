"""Utility to create a TANGO or red-green colormaps."""

import numpy as np
from colour import Color


def get_tango_colormap(rgb: bool = True) -> np.ndarray:
    """Create an array of visually distinctive RGB values, TANGO-image based colormap..

    Based of off mseg-api: https://github.com/mseg-dataset/mseg-api/blob/master/mseg/utils/colormap.py

    Args:
        rgb: boolean, whether to return in RGB or BGR order. BGR corresponds to OpenCV default.

    Returns:
        color_list: Numpy array (N,3) of dtype uint8 representing RGB color palette.
    """
    color_list = np.array(
        [
            [252, 233, 79],
            # [237, 212, 0],
            [196, 160, 0],
            [252, 175, 62],
            # [245, 121, 0],
            [206, 92, 0],
            [233, 185, 110],
            [193, 125, 17],
            [143, 89, 2],
            [138, 226, 52],
            # [115, 210, 22],
            [78, 154, 6],
            [114, 159, 207],
            # [52, 101, 164],
            [32, 74, 135],
            [173, 127, 168],
            # [117, 80, 123],
            [92, 53, 102],
            [239, 41, 41],
            # [204, 0, 0],
            [164, 0, 0],
            [238, 238, 236],
            # [211, 215, 207],
            # [186, 189, 182],
            [136, 138, 133],
            # [85, 87, 83],
            [46, 52, 54],
        ]
    ).astype(np.uint8)
    assert color_list.shape[1] == 3
    assert color_list.ndim == 2

    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list


def get_redgreen_colormap(N: int) -> np.ndarray:
    """Obtain an RGB colormap from red to green, with N unique colors.

    Args:
        N: number of unique colors to generate.

    Returns:
        colormap: uint8 array of shape (N,3)
    """
    colormap = np.array([[color_obj.rgb] for color_obj in Color("red").range_to(Color("green"), N)]).squeeze()
    colormap = (colormap * 255).astype(np.uint8)
    return colormap
