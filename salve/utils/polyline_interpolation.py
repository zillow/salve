"""Utilities for polyline interpolation."""

from typing import Tuple

import numpy as np


def get_polyline_length(polyline: np.ndarray) -> float:
    """Calculate the length of a polyline.

    Args:
        polyline: Numpy array of shape (N,2)

    Returns:
        The length of the polyline as a scalar
    """
    assert polyline.shape[1] == 2
    return float(np.linalg.norm(np.diff(polyline, axis=0), axis=1).sum())


def interp_evenly_spaced_points(polyline: np.ndarray, interval_m: float) -> np.ndarray:
    """Nx2 polyline to Mx2 polyline, for waypoint every `interval_m` meters"""

    length_m = get_polyline_length(polyline)
    n_waypoints = int(np.ceil(length_m / interval_m))

    consecutive_dists = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    if np.any(consecutive_dists == 0):
        raise ValueError("Duplicate consecutive waypoints found in polyline.")

    #px, py = eliminate_duplicates_2d(polyline[:, 0], py=polyline[:, 1])
    interp_polyline = interp_arc(t=n_waypoints, points=polyline)

    return interp_polyline


def interp_arc(t: int, points: np.ndarray) -> np.ndarray:
    """Linearly interpolate equally-spaced points along a polyline, either in 2d or 3d.

    We use a chordal parameterization so that interpolated arc-lengths
    will approximate original polyline chord lengths.
        Ref: M. Floater and T. Surazhsky, Parameterization for curve
            interpolation. 2005.
            https://www.mathworks.com/matlabcentral/fileexchange/34874-interparc

    For the 2d case, we remove duplicate consecutive points, since these have zero
    distance and thus cause division by zero in chord length computation.

    <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
    Ref: https://github.com/argoai/argoverse-api/blob/master/argoverse/utils/interpolate.py
         https://github.com/argoai/av2-api/blob/main/src/av2/geometry/interpolate.py#L120

    Args:
        t: number of points that will be uniformly interpolated and returned.
        points: Numpy array of shape (N,2) or (N,3), representing 2d or 3d-coordinates of the arc.

    Returns:
        Numpy array of shape (N,2)

    Raises:
        ValueError: If `points` is not in R^2 or R^3.
    """
    if points.ndim != 2:
        raise ValueError("Input array must be (N,2) or (N,3) in shape.")

    # The number of points on the curve itself.
    n, _ = points.shape

    # Equally spaced in arclength -- the number of points that will be uniformly interpolated.
    eq_spaced_points = np.linspace(0, 1, t)

    # Compute the chordal arclength of each segment.
    # Compute differences between each x coord, to get the dx's
    # Do the same to get dy's. Then the hypotenuse length is computed as a norm.
    chordlen: np.ndarray = np.linalg.norm(np.diff(points, axis=0), axis=1)
    # Normalize the arclengths to a unit total
    chordlen = chordlen / np.sum(chordlen)
    # cumulative arclength

    cumarc: np.ndarray = np.zeros(len(chordlen) + 1)
    cumarc[1:] = np.cumsum(chordlen)

    # Which interval did each point fall in, in terms of eq_spaced_points? (bin index)
    tbins: NDArrayInt = np.digitize(eq_spaced_points, bins=cumarc).astype(int)

    # Catch any problems at the ends
    tbins[np.where((tbins <= 0) | (eq_spaced_points <= 0))] = 1
    tbins[np.where((tbins >= n) | (eq_spaced_points >= 1))] = n - 1

    s = np.divide((eq_spaced_points - cumarc[tbins - 1]), chordlen[tbins - 1])
    anchors = points[tbins - 1, :]
    # broadcast to scale each row of `points` by a different row of s
    offsets = (points[tbins, :] - points[tbins - 1, :]) * s.reshape(-1, 1)
    points_interp: np.ndarray = anchors + offsets

    return points_interp

