"""Utilities for vector graphics rendering with Matplotlib."""

from typing import Optional, Tuple, Union

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MPath


def draw_polygon_mpl(
    ax: plt.Axes,
    polygon: np.ndarray,
    color: Union[Tuple[float, float, float], str],
    linewidth: Optional[float] = None,
) -> None:
    """Draw a polygon's boundary.

    The polygon's first and last point must be the same (repeated).

    From https://github.com/argoai/av2-api/blob/main/src/av2/rendering/vector.py (MIT license)

    Args:
        ax: Matplotlib axes instance to draw on
        polygon: Array of shape (N, 2) or (N, 3)
        color: Tuple of shape (3,) representing the RGB color or a single character 3-tuple, e.g. 'b'
        linewidth: Width of the lines.
    """
    if linewidth is None:
        ax.plot(polygon[:, 0], polygon[:, 1], color=color)
    else:
        ax.plot(polygon[:, 0], polygon[:, 1], color=color, linewidth=linewidth)


def plot_polygon_patch_mpl(
    polygon_pts: np.ndarray,
    ax: plt.Axes,
    color: Union[Tuple[float, float, float], str] = "y",
    alpha: float = 0.3,
    zorder: int = 1,
) -> None:
    """Plot a lane segment polyon using matplotlib's PathPatch object.

    From https://github.com/argoai/av2-api/blob/main/src/av2/rendering/vector.py (MIT license)

    Reference:
    See Descartes (https://github.com/benjimin/descartes/blob/master/descartes/patch.py)
    and Matplotlib: https://matplotlib.org/stable/gallery/shapes_and_collections/path_patch.html

    Args:
        polygon_pts: Array of shape (N, 2) representing the points of the polygon
        ax: Matplotlib axes.
        color: Tuple of shape (3,) representing the RGB color or a single character 3-tuple, e.g. 'b'.
        alpha: the opacity of the lane segment.
        zorder: Ordering of the plot overlays.
    """
    n, _ = polygon_pts.shape
    codes = np.ones(n, dtype=MPath.code_type) * MPath.LINETO
    codes[0] = MPath.MOVETO

    vertices = polygon_pts[:, :2]
    mpath = MPath(vertices, codes)
    patch = mpatches.PathPatch(mpath, facecolor=color, edgecolor=color, alpha=alpha, zorder=zorder)
    ax.add_patch(patch)


def legend_without_duplicate_labels(ax: matplotlib.axes.Axes) -> None:
    """Add a legend to Matplotlib axes, simultaneously removing duplicate labels.

    Reference: https://stackoverflow.com/a/56253636

    Args:
        ax: Current drawing canvas for Matplotlib, to which items have been plotted w/ label attributes.
    """
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
