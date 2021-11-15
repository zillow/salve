
"""
Tools for aligning room predictions to dominant axes.

Can use vanishing points, PCA or polgon edge angles.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import argoverse.utils.geometry as geometry_utils
import gtsfm.utils.ellipsoid as ellipsoid_utils
import matplotlib.pyplot as plt
import numpy as np
import rdp
from argoverse.utils.sim2 import Sim2
from gtsam import Rot2, Point3, Point3Pairs, Pose2, Similarity3

import afp.utils.rotation_utils as rotation_utils
from afp.common.edgewdopair import EdgeWDOPair
from afp.common.posegraph2d import PoseGraph2d


# This allows angles in the range [84.3, 95.7] to be considered close to 90 degrees.
MAX_RIGHT_ANGLE_DEVIATION = 0.1
MAX_ALLOWED_CORRECTION_DEG = 15.0


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


def align_pairs_by_vanishing_angle(
    i2Si1_dict: Dict[Tuple[int, int], Sim2],
    gt_floor_pose_graph: PoseGraph2d,
    per_edge_wdo_dict: Dict[Tuple[int,int], EdgeWDOPair],
    visualize: bool = False,
) -> Dict[Tuple[int, int], Sim2]:
    """

    Note: 
    - rotating in place about the room center yields wrong results.
    - the rotation must be about a specific point (not about the origin).

    Args:
        i2Si1_dict
        gt_floor_pose_graph
        per_edge_wdo_dict
    """
    from read_prod_predictions import load_inferred_floor_pose_graphs
    raw_dataset_dir = "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05"
    floor_pose_graphs = load_inferred_floor_pose_graphs(
        query_building_id=gt_floor_pose_graph.building_id, raw_dataset_dir=raw_dataset_dir
    )
    pano_dict_inferred = floor_pose_graphs[gt_floor_pose_graph.floor_id].nodes
    # import pdb; pdb.set_trace()

    for (i1, i2), i2Si1 in i2Si1_dict.items():

        edge_wdo_pair = per_edge_wdo_dict[(i1,i2)]
        alignment_object = edge_wdo_pair.alignment_object
        i1_wdo_idx = edge_wdo_pair.i1_wdo_idx
        i1wdocenter_i1fr = getattr(pano_dict_inferred[i1], alignment_object + "s")[i1_wdo_idx].centroid
        #i1wdocenter_i1fr = getattr(gt_floor_pose_graph.nodes[i1], alignment_object + "s")[i1_wdo_idx].centroid
        i1wdocenter_i2fr = i2Si1.transform_from(i1wdocenter_i1fr.reshape(1,2)).squeeze()

        # vertsi1 = gt_floor_pose_graph.nodes[i1].room_vertices_local_2d
        # vertsi2 = gt_floor_pose_graph.nodes[i2].room_vertices_local_2d
        vertsi1 = pano_dict_inferred[i1].room_vertices_local_2d
        vertsi2 = pano_dict_inferred[i2].room_vertices_local_2d

        vertsi1_i2fr = i2Si1.transform_from(vertsi1)

        if visualize:
            plt.subplot(1, 2, 1)
            plt.title("Coordinate in i2's frame.")
            draw_polygon(vertsi1_i2fr, color="r", linewidth=5)
            draw_polygon(vertsi2, color="g", linewidth=1)

            # mark the WDO center on the plot
            plt.scatter(i1wdocenter_i2fr[0], i1wdocenter_i2fr[1], 200, color='k', marker='+', zorder=3)
            plt.scatter(i1wdocenter_i2fr[0], i1wdocenter_i2fr[1], 200, color='k', marker='.', zorder=3)
            plt.axis("equal")

        dominant_angle_method = "vp"
        if dominant_angle_method == "pca":
            dominant_angle_deg1 = get_dominant_direction_from_point_cloud(vertsi1_i2fr)
            dominant_angle_deg2 = get_dominant_direction_from_point_cloud(vertsi2)
            i2r_theta_i2 = dominant_angle_deg2 - dominant_angle_deg1

        elif dominant_angle_method == "vp":
            vp_i1 = pano_dict_inferred[i1].vanishing_angle_deg
            vp_i2 = pano_dict_inferred[i2].vanishing_angle_deg

            i2r_theta_i2 = compute_vp_correction(i2Si1=i2Si1, vp_i1=vp_i1, vp_i2=vp_i2)

            plt.title(f"i1, i2 = ({i1},{i2}) -> vps ({vp_i1:.1f}, {vp_i2:.1f})")

        elif dominant_angle_method == "polygon_edge_angles":
            # this has to happen in a common reference frame! ( in i2's frame).
            dominant_angle_deg1, angle_frac1 = determine_rotation_angle(vertsi1_i2fr)
            dominant_angle_deg2, angle_frac2 = determine_rotation_angle(vertsi2)
            i2r_theta_i2 = dominant_angle_deg2 - dominant_angle_deg1

            # Below: using the oracle.
            # wSi1 = gt_floor_pose_graph.nodes[i1].global_Sim2_local
            # wSi2 = gt_floor_pose_graph.nodes[i2].global_Sim2_local
            # wSi1 = i2Si1_dict[i1]
            # wSi2 = i2Si1_dict[i2]
            # i2Si1 = wSi2.inverse().compose(wSi1)

        if np.absolute(i2r_theta_i2) > MAX_ALLOWED_CORRECTION_DEG:
            print(f"Skipping for too large of a correction -> {i2r_theta_i2:.1f} deg.")
            
            if visualize:
                plt.show()
                plt.close("all")
            continue

        print(f"Rotate by {i2r_theta_i2:.2f} deg.", )
        i2r_R_i2 = rotation_utils.rotmat2d(theta_deg=i2r_theta_i2)
        i2r_S_i2 = Sim2(R=i2r_R_i2, t=np.zeros(2), s=1.0)
        # verts_i1_ = i2Si1_dominant.transform_from(verts_i1)

        # method = "rotate_about_origin_first"
        # method = "rotate_about_origin_last"
        # method = "rotate_about_centroid_first"
        #method = "none"
        #method = "rotate_in_place_last_about_roomcenter"
        method = "rotate_about_wdo"

        if method == "rotate_about_wdo":

            vertsi1_i2fr_r = geometry_utils.rotate_polygon_about_pt(
                vertsi1_i2fr, rotmat=i2r_R_i2, center_pt=i1wdocenter_i2fr
            )
            # note: computing i2rSi2.compose(i2Si1) as:
            # and then i2rTi2 = compute_i2Ti1(pts1=vertsi1_i2fr, pts2=vertsi1_i2fr_r)
            #   DOES NOT WORK! 
            i2rTi1 = compute_i2Ti1(pts1=vertsi1, pts2=vertsi1_i2fr_r)
            i2rSi1 = Sim2(R=i2rTi1.rotation().matrix(), t=i2rTi1.translation(), s=1.0)
            i2Si1_dict[(i1,i2)] = i2rSi1

        if visualize:
            plt.subplot(1, 2, 2)
            plt.title(f"{i1} {i2}")
            draw_polygon(vertsi1_i2fr_r, color="r", linewidth=5)
            draw_polygon(vertsi2, color="g", linewidth=1)
            plt.axis("equal")
            plt.show()
            plt.close("all")

    return i2Si1_dict


def compute_vp_correction(i2Si1: Sim2, vp_i1: float, vp_i2: float) -> float:
    """

    Args:
        i2Si1: pose of camera i1 in i2's frame.
        vp_i1: vanishing angle of i1
        vp_i2: vanishing angle of i2

    Returns:
        i2r_theta_i2: correction to relative pose
    """
    i2_theta_i1 = rotation_utils.rotmat2theta_deg(i2Si1.rotation)
    i2r_theta_i2 = -((vp_i2 - vp_i1) + i2_theta_i1)
    i2r_theta_i2 = i2r_theta_i2 % 90

    if i2r_theta_i2 > 45:
        i2r_theta_i2 = i2r_theta_i2 - 90

    return i2r_theta_i2


def test_compute_vp_correction() -> None:
    """ """
    pass
    # TODO: write this unit test.



def draw_polygon(poly: np.ndarray, color: str, linewidth: float = 1) -> None:
    """ """
    verts = np.vstack([poly, poly[0]])  # allow connection between the last and first vertex

    plt.plot(verts[:, 0], verts[:, 1], color=color, linewidth=linewidth)
    plt.scatter(verts[:, 0], verts[:, 1], 10, color=color, marker=".")


def test_align_pairs_by_vanishing_angle() -> None:
    """Ensure we can use vanishing angles to make small corrections to rotations, with perfect layouts (no noise)."""

    # cameras are at the center of each room. doorway is at (0,1.5)
    wTi1 = Pose2(Rot2(), np.array([1.5, 1.5]))
    wTi2 = Pose2(Rot2(), np.array([-1.5, 1.5]))

    i2Ri1 = wTi2.between(wTi1).rotation().matrix()  # 2x2 identity matrix.
    i2ti1 = wTi2.between(wTi1).translation()

    i2Ri1_noisy = rotation_utils.rotmat2d(theta_deg=30)

    # simulate noisy rotation, but correct translation
    i2Si1_dict = {(1, 2): Sim2(R=i2Ri1, t=i2ti1, s=1.0)}

    # We cannot specify these layouts in the world coordinate frame. They must be in the local body frame.
    # fmt: off
    # Note: this is provided in i1's frame.
    layout1_w = np.array(
        [
            [0,0],
            [3,0],
            [3,3],
            [0,3]
        ]
    ).astype(np.float32)
    # TODO: figure out how to undo the effect on the translation, as well!

    #(align at WDO only! and then recompute!)
    # Note: this is provided in i2's frame.
    #layout2 = (layout1 - np.array([3,0])) @ i2Ri1_noisy.T
    # layout2 = (layout1 @ i2Ri1_noisy.T) - np.array([3,0])
    layout2_w = geometry_utils.rotate_polygon_about_pt(layout1_w - np.array([3.,0]), rotmat=i2Ri1_noisy, center_pt=np.array([-1.5,1.5]))
    # fmt: on

    draw_polygon(layout1_w, color="r", linewidth=5)
    draw_polygon(layout2_w, color="g", linewidth=1)
    plt.axis("equal")
    plt.show()

    def transform_point_cloud(pts_w: np.ndarray, iTw: Pose2) -> np.ndarray:
        """Transfer from world frame to camera i's frame."""
        return np.vstack([iTw.transformFrom(pt_w) for pt_w in pts_w])

    layout1_i1 = transform_point_cloud(layout1_w, iTw=wTi1.inverse())
    layout2_i2 = transform_point_cloud(layout2_w, iTw=wTi2.inverse())

    pano_data_1 = SimpleNamespace(**{"room_vertices_local_2d": layout1_i1})
    pano_data_2 = SimpleNamespace(**{"room_vertices_local_2d": layout2_i2})

    # use square and rotated square
    gt_floor_pose_graph = SimpleNamespace(**{"nodes": {1: pano_data_1, 2: pano_data_2}})

    import pdb

    pdb.set_trace()
    # ensure delta pose is accounted for properly.
    i2Si1_dict_aligned = align_pairs_by_vanishing_angle(i2Si1_dict, gt_floor_pose_graph)


def test_align_pairs_by_vanishing_angle_noisy() -> None:
    """Ensure delta pose is accounted for properly, with dominant rotation directions computed from noisy layouts.

    Using noisy contours, using Douglas-Peucker to capture the rough manhattanization.

    """
    assert False
    i2Ri1 = None
    i2ti1 = None

    # simulate noisy rotation, but correct translation
    i2Si1_dict = {(0, 1): Sim2(R=i2Ri1, t=i2ti1, s=1.0)}

    layout0 = np.array([[]])
    layout1 = np.array([[]])

    pano_data_0 = SimpleNamespace(**{"room_vertices_local_2d": layout0})
    pano_data_1 = SimpleNamespace(**{"room_vertices_local_2d": layout1})

    # use square and rotated square
    gt_floor_pose_graph = SimpleNamespace(**{"nodes": {0: pano_data_0, 1: pano_data_1}})

    i2Si1_dict_aligned = align_pairs_by_vanishing_angle(i2Si1_dict, gt_floor_pose_graph)


def compute_i2Ti1(pts1: np.ndarray, pts2: np.ndarray) -> None:
    """
    pts1 and pts2 need NOT be in a common reference frame.
    """

    # lift to 3d plane
    pt_pairs_i2i1 = []
    for pt1, pt2 in zip(pts1, pts2):
        pt1_3d = np.array([pt1[0], pt1[1], 0])
        pt2_3d = np.array([pt2[0], pt2[1], 0])
        pt_pairs_i2i1 += [(Point3(pt2_3d), Point3(pt1_3d))]

    pt_pairs_i2i1 = Point3Pairs(pt_pairs_i2i1)
    i2Si1 = Similarity3.Align(abPointPairs=pt_pairs_i2i1)

    # project back to 2d
    i2Ri1 = i2Si1.rotation().matrix()[:2, :2]
    theta_deg = rotation_utils.rotmat2theta_deg(i2Ri1)
    i2Ti1 = Pose2(Rot2.fromDegrees(theta_deg), i2Si1.translation()[:2])
    return i2Ti1


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
    i2Ti1 = compute_i2Ti1(pts1=pts1_w, pts2=pts2_w)

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
    i2Ti1 = compute_i2Ti1(pts1=pts1_w, pts2=pts2_w)
    


if __name__ == "__main__":
    """ """
    # test_determine_rotation_angle_manhattanroom1()
    # test_determine_rotation_angle_manhattanroom2()

    #test_get_dominant_direction_from_point_cloud()
    test_get_dominant_direction_from_point_cloud_noisycontour()

