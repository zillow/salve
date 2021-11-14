
"""
Tools for aligning room predictions to dominant axes.
"""
from typing import Optional, Tuple

import gtsfm.utils.ellipsoid as ellipsoid_utils
import matplotlib.pyplot as plt
import numpy as np
import rdp

import afp.utils.rotation_utils as rotation_utils


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
        angle: Dominant room shape rotation angle, in degrees, in the range [-45,45]. Returns None if no room
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
        angle = angle - 90
    else:
        angle = angle
    angle_fraction = len(angles) / len(poly)

    return angle, angle_fraction


def test_determine_rotation_angle_manhattanroom1() -> None:
    """ """
    vertsi1_i2fr = np.array(
        [
            [-2.2514273 , -1.19972439],
            [-2.28502837,  0.17584117],
            [-2.50067059,  0.17057366],
            [-2.52850206,  1.30994228],
            [-1.89300909,  1.32546553],
            [-1.89455772,  1.3888638 ],
            [ 0.56135492,  1.4488546 ],
            [ 0.56784876,  1.18300859],
            [ 1.77462389,  1.2124866 ],
            [ 1.83111122, -1.09999984]
        ])

    dominant_angle_deg, angle_frac = determine_rotation_angle(poly=vertsi1_i2fr)
    # draw_polygon(vertsi1_i2fr, color="g", linewidth=1)
    # plt.show()
    # import pdb; pdb.set_trace()
    expected_dominant_angle_deg = 1.399
    expected_angle_frac = 1.0

    assert np.isclose(dominant_angle_deg, expected_dominant_angle_deg, atol=1e-3)
    assert np.isclose(angle_frac, expected_angle_frac, atol=1e-3)


def test_determine_rotation_angle_manhattanroom2() -> None:
    """ """
    vertsi2 = np.array(
        [
            [-2.28579039, -1.17761538],
            [-2.23140688,  0.19728535],
            [-2.44694488,  0.20581085],
            [-2.4018995 ,  1.3446288 ],
            [-1.76671367,  1.31950434],
            [-1.76420719,  1.38287197],
            [ 0.69051847,  1.28577652],
            [ 0.68000814,  1.02005899],
            [ 1.88620002,  0.97234867],
            [ 1.79477498, -1.33902011]
        ])

    dominant_angle_deg, angle_frac = determine_rotation_angle(poly=vertsi2)
    # draw_polygon(vertsi2, color="g", linewidth=1)
    # plt.show()
    
    #dominant_angle_deg2, angle_frac2 = (2.2651251515060835, 1.0)

    expected_dominant_angle_deg = -2.265
    expected_angle_frac = 1.0

    assert np.isclose(dominant_angle_deg, expected_dominant_angle_deg, atol=1e-3)
    assert np.isclose(angle_frac, expected_angle_frac, atol=1e-3)


def draw_polygon(poly: np.ndarray, color: str, linewidth: float = 1) -> None:
    """ """
    verts = np.vstack([poly, poly[0]])  # allow connection between the last and first vertex

    plt.plot(verts[:, 0], verts[:, 1], color=color, linewidth=linewidth)
    plt.scatter(verts[:, 0], verts[:, 1], 10, color=color, marker=".")



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


def get_dominant_direction_from_point_cloud(point_cloud: np.ndarray) -> np.ndarray:
    """
    Args:
        point_cloud:

    Returns:
        dominant_angle:
    """
    N = point_cloud.shape[0]
    point_cloud_3d = np.zeros((N,3))
    point_cloud_3d[:,:2] = point_cloud

    wuprightRw = ellipsoid_utils.get_alignment_rotation_matrix_from_svd(point_cloud_3d)

    # only consider the xy plane.
    theta_deg = rotation_utils.rotmat2theta_deg(wuprightRw[:2,:2])
    theta_deg = theta_deg % 90
    if theta_deg > 45:
        theta_deg = theta_deg - 90
    else:
        theta_deg = theta_deg

    return theta_deg



def test_get_dominant_direction_from_point_cloud() -> None:
    """ """

    pts = np.array(
        [
            [0,2],
            [2,0],
            [4,2],
            [2,4]
        ])
    theta_deg = get_dominant_direction_from_point_cloud(point_cloud=pts)
    import pdb; pdb.set_trace()

    wuprightRw = rotation_utils.rotmat2d(theta_deg=theta_deg)
    pts_upright = pts @ wuprightRw.T
    draw_polygon(pts, color="g", linewidth=1)
    draw_polygon(pts_upright, color="r", linewidth=1)
    plt.show()
    import pdb; pdb.set_trace()
    assert np.isclose(theta_deg, 45.0)


def test_get_dominant_direction_from_point_cloud_noisycontour() -> None:
    """ """
    pts = np.array(
        [
            [-1.53565851e-16, -1.25396034e+00],
            [-1.26181436e-01, -1.27988242e+00],
            [-4.11747648e-01, -1.27121274e+00],
            [-4.82221888e-01, -1.12370988e+00],
            [-6.82202389e-01, -1.03339383e+00],
            [-7.58918883e-01, -9.00692972e-01],
            [-8.17266696e-01, -5.16402841e-01],
            [-7.30617455e-01,  6.33083629e-01],
            [ 7.15242613e-01,  5.97047324e-01],
            [ 7.43703725e-01,  5.61197177e-01],
            [ 6.67396098e-01, -4.81455496e-01],
            [ 5.00910746e-01, -6.65935793e-01],
            [ 1.13918679e-01, -1.23301478e+00],
            [ 1.51643861e-16, -1.23826609e+00]
        ])

    pts = rdp.rdp(pts, epsilon=0.1)
    theta_deg = get_dominant_direction_from_point_cloud(point_cloud=pts)

    wuprightRw = rotation_utils.rotmat2d(theta_deg=theta_deg)
    pts_upright = pts @ wuprightRw.T
    draw_polygon(pts, color="g", linewidth=1)
    draw_polygon(pts_upright, color="r", linewidth=1)
    plt.show()


if __name__ == "__main__":
    """ """
    # test_determine_rotation_angle_manhattanroom1()
    # test_determine_rotation_angle_manhattanroom2()

    #test_get_dominant_direction_from_point_cloud()
    test_get_dominant_direction_from_point_cloud_noisycontour()

