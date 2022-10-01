"""Tools for aligning room predictions to dominant axes.

Can use vanishing points, PCA or polgon edge angles.
"""

from typing import Dict, Optional, Tuple

import gtsfm.utils.ellipsoid as ellipsoid_utils
import matplotlib.pyplot as plt
import numpy as np
from gtsam import Rot2, Point3, Point3Pairs, Pose2, Similarity3

import salve.dataset.hnet_prediction_loader as hnet_prediction_loader
import salve.utils.rotation_utils as rotation_utils
from salve.common.edgewdopair import EdgeWDOPair
from salve.common.pano_data import PanoData
from salve.common.posegraph2d import PoseGraph2d
from salve.common.sim2 import Sim2

# This allows angles in the range [84.3, 95.7] deg. to be considered "close" to 90 degrees.
MAX_RIGHT_ANGLE_DEVIATION = 0.1
MAX_ALLOWED_CORRECTION_DEG = 15.0


def determine_dominant_rotation_angle(poly: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Extracts the dominant rotation angle of a room shape.

    Note: only works for ground truth shapes with close to zero noise.

    We find which adjacent edges form close to a right-angle (90 deg.), and for all such edges, we take
    the median of their angle with respect to the x-axis.
    Reference: From visualize_zfm30_data.py

    Args:
        poly: Room shape polygon as a numpy array (N,2).

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

        # Normalize directions to unit length.
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


def get_dominant_direction_from_point_cloud(point_cloud: np.ndarray) -> np.ndarray:
    """Fit an ellipsoid to a point-cloud (PCA / SVD), and use semi-axis direction.

    Args:
        point_cloud: array of shape (N,2) representing 2d point cloud.

    Returns:
        dominant_angle:
    """
    N = point_cloud.shape[0]
    point_cloud_3d = np.zeros((N, 3))
    point_cloud_3d[:, :2] = point_cloud

    wuprightRw = ellipsoid_utils.get_alignment_rotation_matrix_from_svd(point_cloud_3d)

    # only consider the xy plane.
    theta_deg = rotation_utils.rotmat2theta_deg(wuprightRw[:2, :2])
    theta_deg = theta_deg % 90
    if theta_deg > 45:
        theta_deg = theta_deg - 90
    else:
        theta_deg = theta_deg

    return theta_deg


def align_pairs_by_vanishing_angle(
    i2Si1_dict: Dict[Tuple[int, int], Sim2],
    inferred_floor_pose_graph: PoseGraph2d,
    per_edge_wdo_dict: Dict[Tuple[int, int], EdgeWDOPair],
    visualize: bool = False,
) -> Dict[Tuple[int, int], Sim2]:
    """Align a collection of image pairs by vanishing angle, up to a maximum allowed refinement threshold.

    Note: Rotating in-place about the room center yields wrong results. Rather, the rotation must be about
    a specific point (not about the origin). Specifically, the rotation must be about the W/D/O object
    (we choose the W/D/O midpoint here).

    Args:
        i2Si1_dict: relative pose measurements.
        inferred_floor_pose_graph: inferred pose graph for specified ZInD floor.
        per_edge_wdo_dict:
        visualize: whether to visualize intermediate results.

    Returns:
        i2Si1_dict: refined relative pose measurements.
    """
    pano_dict_inferred = inferred_floor_pose_graph.nodes

    for (i1, i2), i2Si1 in i2Si1_dict.items():
        edge_wdo_pair = per_edge_wdo_dict[(i1, i2)]
        # move to rotated i2's coordinate frame (i2r)
        # p_i2r = i2r_S_i1 * p_i1
        i2rSi1 = align_pair_measurement_by_vanishing_angle(i1, i2, i2Si1, edge_wdo_pair, pano_dict_inferred, visualize)
        if i2rSi1 is None:
            # requested correction was too large, skip proposed refinement.
            continue

        i2Si1_dict[(i1, i2)] = i2rSi1

        if visualize:
            plt.subplot(1, 2, 2)
            plt.title(f"{i1} {i2}")
            draw_polygon(vertsi1_i2fr_r, color="r", linewidth=5)
            draw_polygon(vertsi2, color="g", linewidth=1)
            plt.axis("equal")
            plt.show()
            plt.close("all")

    return i2Si1_dict


def align_pair_measurement_by_vanishing_angle(
    i1: int, i2: int, i2Si1: Sim2, edge_wdo_pair: EdgeWDOPair, pano_dict_inferred: Dict[int, PanoData], visualize: bool
) -> Optional[Sim2]:
    """Align a single image pair by their vanishing angles, up to a maximum allowed refinement threshold.

    Args:
        i1: panorama ID of camera 1
        i2: panorama ID of camera 2
        i2Si1: initial relative pose measurement, from possibly noisy W/D/O detections.
        edge_wdo_pair:
        pano_dict_inferred:
        visualize:

    Returns:
        i2rSi1: updated measurement, to an updated i2 frame. (or can think of it as updated i1 pose).
            If None, then the requested correction was too large
    """
    alignment_object = edge_wdo_pair.alignment_object
    i1_wdo_idx = edge_wdo_pair.i1_wdo_idx
    i1wdocenter_i1fr = getattr(pano_dict_inferred[i1], alignment_object + "s")[i1_wdo_idx].centroid
    # i1wdocenter_i1fr = getattr(gt_floor_pose_graph.nodes[i1], alignment_object + "s")[i1_wdo_idx].centroid
    i1wdocenter_i2fr = i2Si1.transform_from(i1wdocenter_i1fr.reshape(1, 2)).squeeze()

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
        plt.scatter(i1wdocenter_i2fr[0], i1wdocenter_i2fr[1], 200, color="k", marker="+", zorder=3)
        plt.scatter(i1wdocenter_i2fr[0], i1wdocenter_i2fr[1], 200, color="k", marker=".", zorder=3)
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
        dominant_angle_deg1, angle_frac1 = determine_dominant_rotation_angle(vertsi1_i2fr)
        dominant_angle_deg2, angle_frac2 = determine_dominant_rotation_angle(vertsi2)
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
        return None

    # print(f"Rotate by {i2r_theta_i2:.2f} deg.", )
    i2r_R_i2 = rotation_utils.rotmat2d(theta_deg=i2r_theta_i2)
    i2r_S_i2 = Sim2(R=i2r_R_i2, t=np.zeros(2), s=1.0)
    # verts_i1_ = i2Si1_dominant.transform_from(verts_i1)

    vertsi1_i2fr_r = rotation_utils.rotate_polygon_about_pt(vertsi1_i2fr, rotmat=i2r_R_i2, center_pt=i1wdocenter_i2fr)
    # note: computing i2rSi2.compose(i2Si1) as:
    # and then i2rTi2 = compute_i2Ti1(pts1=vertsi1_i2fr, pts2=vertsi1_i2fr_r)
    #   DOES NOT WORK!
    i2rTi1 = compute_i2Ti1(pts1=vertsi1, pts2=vertsi1_i2fr_r)
    i2rSi1 = Sim2(R=i2rTi1.rotation().matrix(), t=i2rTi1.translation(), s=1.0)

    return i2rSi1


def compute_vp_correction(i2Si1: Sim2, vp_i1: float, vp_i2: float) -> float:
    """Compute vanishing point correction.

    Args:
        i2Si1: pose of camera i1 in i2's frame.
        vp_i1: "vanishing angle" of i1.
        vp_i2: "vanishing angle" of i2.

    Returns:
        i2r_theta_i2: correction to relative pose.
    """
    i2_theta_i1 = rotation_utils.rotmat2theta_deg(i2Si1.rotation)
    i2r_theta_i2 = -((vp_i2 - vp_i1) + i2_theta_i1)
    i2r_theta_i2 = i2r_theta_i2 % 90

    if i2r_theta_i2 > 45:
        i2r_theta_i2 = i2r_theta_i2 - 90

    return i2r_theta_i2


def draw_polygon(poly: np.ndarray, color: str, linewidth: float = 1) -> None:
    """ """
    verts = np.vstack([poly, poly[0]])  # allow connection between the last and first vertex

    plt.plot(verts[:, 0], verts[:, 1], color=color, linewidth=linewidth)
    plt.scatter(verts[:, 0], verts[:, 1], 10, color=color, marker=".")


def compute_i2Ti1(pts1: np.ndarray, pts2: np.ndarray) -> Pose2:
    """Compute relative pose using Sim(3) alignment.

    pts1 and pts2 need NOT be in a common reference frame.

    Args:
        pts1:
        pts2:

    Returns:
        i2Ti1: relative pose between the two panoramas i1 and i2, such that p_i2 = i2Ti1 * p_i1.
    """

    # Lift 2d point cloud to 3d plane.
    pt_pairs_i2i1 = []
    for pt1, pt2 in zip(pts1, pts2):
        pt1_3d = np.array([pt1[0], pt1[1], 0])
        pt2_3d = np.array([pt2[0], pt2[1], 0])
        pt_pairs_i2i1 += [(Point3(pt2_3d), Point3(pt1_3d))]

    pt_pairs_i2i1 = Point3Pairs(pt_pairs_i2i1)
    i2Si1 = Similarity3.Align(abPointPairs=pt_pairs_i2i1)
    # TODO: we should use Pose2.Align() or Similarity2.Align()

    # Project Sim(3) transformation from 3d back to 2d.
    i2Ri1 = i2Si1.rotation().matrix()[:2, :2]
    theta_deg = rotation_utils.rotmat2theta_deg(i2Ri1)
    i2Ti1 = Pose2(Rot2.fromDegrees(theta_deg), i2Si1.translation()[:2])
    return i2Ti1
