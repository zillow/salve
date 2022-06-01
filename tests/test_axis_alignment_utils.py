
"""Unit tests on different room axis alignment utilities."""

from types import SimpleNamespace

import argoverse.utils.geometry as geometry_utils
import matplotlib.pyplot as plt
import numpy as np
import rdp
from argoverse.utils.sim2 import Sim2
from gtsam import Pose2, Rot2

import salve.utils.axis_alignment_utils as axis_alignment_utils
import salve.utils.rotation_utils as rotation_utils


def test_determine_dominant_rotation_angle_manhattanroom1() -> None:
    """For Manhattan-style room, generally cuboidal, but with Manhattan style L-group and Manhattan alcoves.

    Should be slightly tilted (around +1.5 degrees) from the +x axis.
    """
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

    dominant_angle_deg, angle_frac = axis_alignment_utils.determine_dominant_rotation_angle(poly=vertsi1_i2fr)
    # axis_alignment_utils.draw_polygon(vertsi1_i2fr, color="g", linewidth=1)
    # plt.show()

    expected_dominant_angle_deg = 1.399
    expected_angle_frac = 1.0

    assert np.isclose(dominant_angle_deg, expected_dominant_angle_deg, atol=1e-3)
    assert np.isclose(angle_frac, expected_angle_frac, atol=1e-3)


def test_determine_dominant_rotation_angle_manhattanroom2() -> None:
    """For Manhattan-style room, generally cuboidal, but with Manhattan style L-group and Manhattan alcoves.

    Should be slightly tilted (around -2.5 degrees) from the +x axis.
    """
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

    dominant_angle_deg, angle_frac = axis_alignment_utils.determine_dominant_rotation_angle(poly=vertsi2)
    # axis_alignment_utils.draw_polygon(vertsi2, color="g", linewidth=1)
    # plt.show()
    
    expected_dominant_angle_deg = -2.265
    expected_angle_frac = 1.0

    assert np.isclose(dominant_angle_deg, expected_dominant_angle_deg, atol=1e-3)
    assert np.isclose(angle_frac, expected_angle_frac, atol=1e-3)


def test_determine_dominant_rotation_angle_square() -> None:
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
    dominant_angle, angle_frac = axis_alignment_utils.determine_dominant_rotation_angle(poly)
    assert dominant_angle == 0
    assert angle_frac == 1.0


def test_determine_dominant_rotation_angle_rectangle() -> None:
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
    dominant_angle, angle_frac = axis_alignment_utils.determine_dominant_rotation_angle(poly)
    assert dominant_angle == 0
    assert angle_frac == 1.0


def test_determine_dominant_rotation_angle_triangle() -> None:
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
    dominant_angle, angle_frac = axis_alignment_utils.determine_dominant_rotation_angle(poly)
    assert dominant_angle is None
    assert angle_frac is None


# def test_determine_dominant_rotation_angle_approx_square1() -> None:
#     """ For a shape that is almost a square, but not enough.
#     Vertices are provided clockwise.
#     """

#     # fmt: off
#     poly = np.array(
#         [
#             [0,0],
#             [0,3.5],
#             [3,4],
#             [3,0.5]
#         ]
#     )
#     # fmt: on
#     dominant_angle, angle_frac = axis_alignment_utils.determine_dominant_rotation_angle(poly)
#     assert dominant_angle is None
#     assert angle_frac is None


# def test_determine_dominant_rotation_angle_approx_square2() -> None:
#     """ For a shape that is almost a square, and just close enough.
#     Vertices are provided clockwise.
#     """

#     # fmt: off
#     poly = np.array(
#         [
#             [0,0],
#             [0,3],
#             [3,3.1],
#             [3,0.1]
#         ]
#     )
#     # fmt: on
#     dominant_angle, angle_frac = axis_alignment_utils.determine_dominant_rotation_angle(poly)
#     assert np.isclose(dominant_angle, 0.955, atol=3) # angle in degrees
#     assert np.isclose(angle_frac, 1.0)


def test_compute_relative_angle_deg() -> None:
    """Check that the clockwise angle in degrees (between 0 and 360) is computed correctly."""
    v1 = np.array([1,0])
    v2 = np.array([1,0])
    angle_deg = axis_alignment_utils.compute_relative_angle_deg(v1, v2)
    assert np.isclose(angle_deg, 0.0)

    v1 = np.array([0,1])
    v2 = np.array([1,0])
    angle_deg = axis_alignment_utils.compute_relative_angle_deg(v1, v2)
    # wrap around 1/4 turn clockwise
    assert np.isclose(angle_deg, 90.0)

    v1 = np.array([1,0])
    v2 = np.array([0,1])
    angle_deg = axis_alignment_utils.compute_relative_angle_deg(v1, v2)
    # wrap around 3/4 turn clockwise
    assert np.isclose(angle_deg, 270.0)

    v1 = np.array([1,0])
    v2 = np.array([0,-1])
    angle_deg = axis_alignment_utils.compute_relative_angle_deg(v1, v2)
    # wrap around 1/4 turn clockwise
    assert np.isclose(angle_deg, 90.0)


def test_get_dominant_direction_from_point_cloud() -> None:
    """
    TODO: add docstring.
    """
    pts = np.array(
        [
            [0,2],
            [2,0],
            [4,2],
            [2,4]
        ])
    theta_deg = axis_alignment_utils.get_dominant_direction_from_point_cloud(point_cloud=pts)
    # import pdb; pdb.set_trace()

    wuprightRw = rotation_utils.rotmat2d(theta_deg=theta_deg)
    pts_upright = pts @ wuprightRw.T
    # axis_alignment_utils.draw_polygon(pts, color="g", linewidth=1)
    # axis_alignment_utils.draw_polygon(pts_upright, color="r", linewidth=1)
    # plt.show()
    # import pdb; pdb.set_trace()
    assert np.isclose(theta_deg, 45.0)


# def test_get_dominant_direction_from_point_cloud_noisycontour() -> None:
#     """
#     Note: computing dominant direction from noisy shape is not useful/helpful.
#     """
#     pts = np.array(
#         [
#             [-1.53565851e-16, -1.25396034e+00],
#             [-1.26181436e-01, -1.27988242e+00],
#             [-4.11747648e-01, -1.27121274e+00],
#             [-4.82221888e-01, -1.12370988e+00],
#             [-6.82202389e-01, -1.03339383e+00],
#             [-7.58918883e-01, -9.00692972e-01],
#             [-8.17266696e-01, -5.16402841e-01],
#             [-7.30617455e-01,  6.33083629e-01],
#             [ 7.15242613e-01,  5.97047324e-01],
#             [ 7.43703725e-01,  5.61197177e-01],
#             [ 6.67396098e-01, -4.81455496e-01],
#             [ 5.00910746e-01, -6.65935793e-01],
#             [ 1.13918679e-01, -1.23301478e+00],
#             [ 1.51643861e-16, -1.23826609e+00]
#         ])

#     pts = rdp.rdp(pts, epsilon=0.1)
#     theta_deg = axis_alignment_utils.get_dominant_direction_from_point_cloud(point_cloud=pts)

#     wuprightRw = rotation_utils.rotmat2d(theta_deg=theta_deg)
#     pts_upright = pts @ wuprightRw.T
#     axis_alignment_utils.draw_polygon(pts, color="g", linewidth=1)
#     axis_alignment_utils.draw_polygon(pts_upright, color="r", linewidth=1)
#     plt.show()



# def test_align_pairs_by_vanishing_angle() -> None:
#     """Ensure we can use vanishing angles to make small corrections to rotations, with perfect layouts (no noise)."""

#     # cameras are at the center of each room. doorway is at (0,1.5)
#     wTi1 = Pose2(Rot2(), np.array([1.5, 1.5]))
#     wTi2 = Pose2(Rot2(), np.array([-1.5, 1.5]))

#     i2Ri1 = wTi2.between(wTi1).rotation().matrix()  # 2x2 identity matrix.
#     i2ti1 = wTi2.between(wTi1).translation()

#     i2Ri1_noisy = rotation_utils.rotmat2d(theta_deg=30)

#     # simulate noisy rotation, but correct translation
#     i2Si1_dict = {(1, 2): Sim2(R=i2Ri1, t=i2ti1, s=1.0)}

#     # We cannot specify these layouts in the world coordinate frame. They must be in the local body frame.
#     # fmt: off
#     # Note: this is provided in i1's frame.
#     layout1_w = np.array(
#         [
#             [0,0],
#             [3,0],
#             [3,3],
#             [0,3]
#         ]
#     ).astype(np.float32)
#     # TODO: figure out how to undo the effect on the translation, as well!

#     #(align at WDO only! and then recompute!)
#     # Note: this is provided in i2's frame.
#     #layout2 = (layout1 - np.array([3,0])) @ i2Ri1_noisy.T
#     # layout2 = (layout1 @ i2Ri1_noisy.T) - np.array([3,0])
#     layout2_w = geometry_utils.rotate_polygon_about_pt(layout1_w - np.array([3.,0]), rotmat=i2Ri1_noisy, center_pt=np.array([-1.5,1.5]))
#     # fmt: on

#     axis_alignment_utils.draw_polygon(layout1_w, color="r", linewidth=5)
#     axis_alignment_utils.draw_polygon(layout2_w, color="g", linewidth=1)
#     plt.axis("equal")
#     plt.show()

#     def transform_point_cloud(pts_w: np.ndarray, iTw: Pose2) -> np.ndarray:
#         """Transfer from world frame to camera i's frame."""
#         return np.vstack([iTw.transformFrom(pt_w) for pt_w in pts_w])

#     layout1_i1 = transform_point_cloud(layout1_w, iTw=wTi1.inverse())
#     layout2_i2 = transform_point_cloud(layout2_w, iTw=wTi2.inverse())

#     pano_data_1 = SimpleNamespace(**{"room_vertices_local_2d": layout1_i1})
#     pano_data_2 = SimpleNamespace(**{"room_vertices_local_2d": layout2_i2})

#     # use square and rotated square
#     gt_floor_pose_graph = SimpleNamespace(**{"nodes": {1: pano_data_1, 2: pano_data_2}})

#     import pdb

#     pdb.set_trace()
#     # ensure delta pose is accounted for properly.
#     i2Si1_dict_aligned = axis_alignment_utils.align_pairs_by_vanishing_angle(i2Si1_dict, gt_floor_pose_graph)


# def test_align_pairs_by_vanishing_angle_noisy() -> None:
#     """Ensure delta pose is accounted for properly, with dominant rotation directions computed from noisy layouts.

#     Using noisy contours, using Douglas-Peucker to capture the rough manhattanization.

#     """
#     assert False
#     i2Ri1 = None
#     i2ti1 = None

#     # simulate noisy rotation, but correct translation
#     i2Si1_dict = {(0, 1): Sim2(R=i2Ri1, t=i2ti1, s=1.0)}

#     layout0 = np.array([[]])
#     layout1 = np.array([[]])

#     pano_data_0 = SimpleNamespace(**{"room_vertices_local_2d": layout0})
#     pano_data_1 = SimpleNamespace(**{"room_vertices_local_2d": layout1})

#     # use square and rotated square
#     gt_floor_pose_graph = SimpleNamespace(**{"nodes": {0: pano_data_0, 1: pano_data_1}})

#     per_edge_wdo_dict = {}

#     i2Si1_dict_aligned = axis_alignment_utils.align_pairs_by_vanishing_angle(i2Si1_dict, gt_floor_pose_graph, per_edge_wdo_dict)


def test_compute_i2Ti1() -> None:
    """ """

    pts1_w = np.array(
        [
            [2,1],
            [1,1],
            [1,2]
        ])
    pts2_w = np.array(
        [
            [-1,1],
            [0,1],
            [0,0]
        ])
    i2Ti1 = axis_alignment_utils.compute_i2Ti1(pts1=pts1_w, pts2=pts2_w)

    for i in range(3):
        expected_pt2_w = i2Ti1.transformFrom(pts1_w[i])
        print(expected_pt2_w)
        assert np.allclose(pts2_w[i], expected_pt2_w)


def test_compute_i2Ti1_from_rotation_in_place() -> None:
    """
    Take upright line segment, rotate in place, and determine (R,t) to make it happen.
    """
    pts1_w = np.array(
        [
            [0,2],
            [0,1],
            [0,0]
        ])
    pts2_w = np.array(
        [
            [-0.5,2],
            [0,1],
            [0.5,0]
        ])
    i2Ti1 = axis_alignment_utils.compute_i2Ti1(pts1=pts1_w, pts2=pts2_w)
    

# def test_compute_vp_correction() -> None:
#     """ """
#     pass
#     # TODO: write this unit test.
#     assert False


if __name__ == "__main__":
    """ """
    # test_determine_dominant_rotation_angle_manhattanroom1()
    # test_determine_dominant_rotation_angle_manhattanroom2()

    #test_get_dominant_direction_from_point_cloud()
    test_get_dominant_direction_from_point_cloud_noisycontour()

