
"""
Utility to create a TANGO-image based colormap.
"""

import numpy as np

def colormap(rgb: bool = True) -> np.ndarray:
    """Create an array of visually distinctive RGB values.

    Based of off mseg-api: https://github.com/mseg-dataset/mseg-api/blob/master/mseg/utils/colormap.py

    Args:
        rgb: boolean, whether to return in RGB or BGR order. BGR corresponds to OpenCV default.
    
    Returns:
        color_list: Numpy array of dtype uin8 representing RGB color palette.
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
