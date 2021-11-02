
"""
Tools for aligning room predictions to dominant axes.
"""
from typing import Optional, Tuple

import numpy as np

# This allows angles in the range [84.3, 95.7] to be considered close to 90 degrees.
MAX_RIGHT_ANGLE_DEVIATION = 0.1


def determine_rotation_angle(poly: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Extracts the dominant rotation angle of a room shape.

    We find which adjacent edges form close to a right-angle, and for all such edges, we take
    the median of their angle with respect to the x-axis.
    Reference: From visualize_zfm30_data.py

    Args:
        poly: Room shape polygon as a numpy array.

    Returns:
        angle: Dominant room shape rotation angle, in degrees. Returns None if no room
            polygon edges were close to orthogonal.
        angle_fraction: The fraction of polygon angles used to determine the dominant angle.
            Returns None if no room polygon edges were close to orthogonal.
    """
    POS_X_AXIS_DIR = np.array([1, 0])

    angles = []
    # Consider each subsequence of 3 consecutive vertices.
    for v_idx in range(poly.shape[0]):
        # fmt: off
        pt_indices = [
            (v_idx - 2) % len(poly),
            (v_idx - 1) % len(poly),
            (v_idx) % len(poly)
        ]
        # fmt: on
        p1, p2, p3 = [poly[idx] for idx in pt_indices]

        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p2)

        # normalize directions to unit length
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        # cos(theta) measures magnitude of deviation from orthogonal.
        orthogonality_deviation = np.abs(v1.dot(v2))
        print(f"\tClose to orthogonal? {orthogonality_deviation:.1f}")

        # check how close this is to orthogonality.
        if orthogonality_deviation < MAX_RIGHT_ANGLE_DEVIATION:
            angle_deg = compute_relative_angle_deg(v1, POS_X_AXIS_DIR)
            angles.append(angle_deg)

    print("\tAngles between edges (deg.): ", np.round(angles, 1))
    if len(angles) == 0:
        return None, None

    deviations = [ang % 90 for ang in angles]
    angle = np.median(deviations)
    if angle > 45:
        angle = 90 - angle
    else:
        angle = angle
    angle_fraction = len(angles) / len(poly)

    return angle, angle_fraction


def test_determine_rotation_angle_square() -> None:
    """For a simple square. Vertices are provided counterclockwise."""
    # fmt: off
    poly = np.array(
        [
            [0,0],
            [2,0],
            [2,2],
            [0,2]
        ]
    )
    # fmt: on
    dominant_angle, angle_frac = determine_rotation_angle(poly)
    assert dominant_angle == 0
    assert angle_frac == 1.0


def test_determine_rotation_angle_rectangle() -> None:
    """For a simple rectangle. Vertices are provided counterclockwise."""
    # fmt: off
    poly = np.array(
        [
            [0,0],
            [4,0],
            [4,2],
            [0,2]
        ]
    )
    # fmt: on
    dominant_angle, angle_frac = determine_rotation_angle(poly)
    assert dominant_angle == 0
    assert angle_frac == 1.0


def test_determine_rotation_angle_triangle() -> None:
    """For a simple equilateral triangle, with 60 degree interior angles.

    None of the triangle's adjacent edges form an angle close to 90 degrees.
    Vertices are provided counterclockwise.
    """
    # fmt: off
    poly = np.array(
        [
            [-2,0],
            [2,0],
            [0,3.4641]
        ]
    )
    # fmt: on
    dominant_angle, angle_frac = determine_rotation_angle(poly)
    assert dominant_angle is None
    assert angle_frac is None


def test_determine_rotation_angle_approx_square1() -> None:
    """ For a shape that is almost a square, but not enough.
    Vertices are provided clockwise.
    """

    # fmt: off
    poly = np.array(
        [
            [0,0],
            [0,3.5],
            [3,4],
            [3,0.5]
        ]
    )
    # fmt: on
    dominant_angle, angle_frac = determine_rotation_angle(poly)
    assert dominant_angle is None
    assert angle_frac is None

def test_determine_rotation_angle_approx_square2() -> None:
    """ For a shape that is almost a square, and just close enough.
    Vertices are provided clockwise.
    """

    # fmt: off
    poly = np.array(
        [
            [0,0],
            [0,3],
            [3,3.1],
            [3,0.1]
        ]
    )
    # fmt: on
    dominant_angle, angle_frac = determine_rotation_angle(poly)
    assert np.isclose(dominant_angle, 0.955, atol=3) # angle in degrees
    assert np.isclose(angle_frac, 1.0)


def compute_relative_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Returns the clockwise angle in degrees between 0 and 360.

    Note: Both vectors should have unit norm.
    See: https://stackoverflow.com/questions/14066933/direct-way-of-computing-clockwise-angle-between-2-vectors
    """
    if not np.isclose(np.linalg.norm(v1), 1.0) or not np.isclose(np.linalg.norm(v2), 1.0):
        raise RuntimeError("Must normalize vectors to unit length.")

    init_angle = -np.math.atan2(np.linalg.det([v1, v2]), np.dot(v1, v2))
    rotation_angle = np.mod(init_angle + 2 * np.pi, 2 * np.pi)

    return np.rad2deg(rotation_angle)


def test_compute_relative_angle_deg() -> None:
    """ """
    v1 = np.array([1,0])
    v2 = np.array([1,0])
    angle_deg = compute_relative_angle_deg(v1, v2)
    assert np.isclose(angle_deg, 0.0)

    v1 = np.array([0,1])
    v2 = np.array([1,0])
    angle_deg = compute_relative_angle_deg(v1, v2)
    # wrap around 1/4 turn clockwise
    assert np.isclose(angle_deg, 90.0)

    v1 = np.array([1,0])
    v2 = np.array([0,1])
    angle_deg = compute_relative_angle_deg(v1, v2)
    # wrap around 3/4 turn clockwise
    assert np.isclose(angle_deg, 270.0)

    v1 = np.array([1,0])
    v2 = np.array([0,-1])
    angle_deg = compute_relative_angle_deg(v1, v2)
    # wrap around 1/4 turn clockwise
    assert np.isclose(angle_deg, 90.0)
