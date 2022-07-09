"""Unit tests for ..."""

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np

import salve.common.posegraph2d as posegraph2d
import salve.dataset.hnet_prediction_loader as hnet_prediction_loader
import salve.utils.zind_pano_utils as zind_pano_utils


def test_pano_to_pano_contour_projection() -> None:
    """ """
    img_w = 1024
    img_h = 512

    hnet_pred_dir = "/Users/johnlambert/Downloads/zind_hnet_prod_predictions/zind2_john/zind2_john"
    raw_dataset_dir = "/Users/johnlambert/Downloads/ZInD"

    building_id = "0715"
    floor_id = "floor_01"

    i0 = 0
    i1 = 18

    i0_fpath = f"{raw_dataset_dir}/0715/panos/floor_01_partial_room_09_pano_0.jpg"
    i1_fpath = f"{raw_dataset_dir}/0715/panos/floor_01_partial_room_09_pano_18.jpg"

    image_i0 = cv2.resize(imageio.imread(i0_fpath), (img_w, img_h))
    image_i1 = cv2.resize(imageio.imread(i1_fpath), (img_w, img_h))

    gt_pose_graph = posegraph2d.get_gt_pose_graph(
        building_id=building_id, floor_id=floor_id, raw_dataset_dir=raw_dataset_dir
    )

    hnet_floor_predictions = hnet_prediction_loader.load_hnet_predictions(
        query_building_id=building_id, raw_dataset_dir=raw_dataset_dir, predictions_data_root=hnet_pred_dir
    )
    import pdb; pdb.set_trace()

    def get_contour(floor_boundary: np.ndarray) -> np.ndarray:
        # Get contour for pano i0.
        u, v = np.arange(img_w), np.round(floor_boundary)
        return np.hstack([u.reshape(-1, 1), v.reshape(-1, 1)])

    contour_i0 = get_contour(hnet_floor_predictions[floor_id][i0].floor_boundary)
    contour_i1 = get_contour(hnet_floor_predictions[floor_id][i1].floor_boundary)

    # Get pose for pano i0.
    wSi0 = gt_pose_graph.nodes[i0].global_Sim2_local

    # Get pose for pano i1.
    wSi1 = gt_pose_graph.nodes[i1].global_Sim2_local

    # project contour i0 from pano i0 into pano i1.
    i1Si0 = wSi1.inverse().compose(wSi0)

    layout_pts_worldmetric_i0 = zind_pano_utils.convert_points_px_to_worldmetric(
        points_px=contour_i0, image_width=img_w, camera_height_m=camera_height_m
    )
    # ignore y values, which are along the vertical axis
    layout_pts_worldmetric_i0 = layout_pts_worldmetric_i0[:, np.array([0, 2])]
    # TODO: remove this when saving (only for plotting a ready-to-go PanoData instance)
    layout_pts_worldmetric_i0[:, 0] *= -1

    layout_pts_worldmetric_i1 = i1Si0.transform_from(layout_pts_worldmetric_i0)

    # project layout into image.
    contour_i1 = zind_pano_utils.convert_points_worldmetric_to_px(
        points_worldmetric=layout_pts_worldmetric_i1, image_width=img_w, camera_height_m=camera_height_m)

    # Visualize contour i0 in image i0.
    plt.imshow(image_i0)
    plt.scatter(contour_i0[:, 0], contour_i0[:, 1], 1, color="r", marker=".")
    plt.show()

    # Visualize contour i0 in image i0.
    plt.imshow(image_i1)
    plt.scatter(contour_i1[:, 0], contour_i1[:, 1], 1, color="r", marker=".")
    plt.show()
    


def test_convert_points_px_to_worldmetric_roundtrip() -> None:
    """ """
    # sample 10k points
    N = 10000
    img_w = 1024
    camera_height_m = 1.3

    contour_px = np.random.randint(low=[0,0], high=[img_w, img_h], size=(N,))

    layout_pts_worldmetric = zind_pano_utils.convert_points_px_to_worldmetric(
        points_px=contour_px, image_width=img_w, camera_height_m=camera_height_m)

    contour_px_ = zind_pano_utils.convert_points_worldmetric_to_px(
        points_worldmetric=layout_pts_worldmetric_i1, image_width=img_w, camera_height_m=camera_height_m)

    assert np.allclose(contour_px, contour_px_)



def test_zind_sphere_to_cartesian() -> None:
    """Ensure that spherical to room-cartesian coordinate conversions are correct.
    
    """
    # provided as (theta, phi) coordinates. Imagine (u,v) provided correspond to a (H,W)=(512,1024) image.
    points_sph = np.array(
        [
            [-np.pi, np.pi/2], # corresponds to (u,v) = (0,0), pointing upwards to top of sphere
            [-np.pi, -np.pi/2], # corresponds to (u,v) = (0,511)
            [np.pi, -np.pi/2], # corresponds to (u,v) = (1023,511)
            [np.pi, np.pi/2], # corresponds to (u,v) = (1023,0)
            [0, 0], # corresponds to (u,v) = (512,256)
            [np.pi/2, 0], # 1/4 way from right edge of pano, midway up pano (u,v)=(756,0)
            [-np.pi,0]
        ])

    points_cart = zind_pano_utils.zind_sphere_to_cartesian(points_sph)

    # fmt: off
    # Points below are in the "Room Cartesian" coordinate system.
    expected_points_cart = np.array(
        [
            [0,  1, 0],
            [0, -1, 0],
            [0, -1, 0],
            [0,  1, 0],
            [0,  0, 1],
            [1, 0, 0],
            [0, 0, -1]
       ]
    )
    # fmt: on
    assert np.allclose(points_cart, expected_points_cart)


def test_zind_cartesian_to_sphere() -> None:
    """Ensure that room-cartesian to spherical coordinate conversions are correct."""

    # Note: can't use (0, X, 0) since arctan(0,0) is undefined.

    eps = 1e-5
    # fmt: off
    points_cart = np.array(
        [
            [0,  0, 1],
            [1, 0, 0],
            [0, 0, -1]
       ]
    )
    # fmt: on
    
    points_sph = zind_pano_utils.zind_cartesian_to_sphere(points_cart)

    # provided as (theta,phi,rho) coordinates. rho=1 for all.
    expected_points_sph = np.array(
        [
            [0, 0, 1], # corresponds to (u,v) = (512,256)
            [np.pi/2, 0, 1], # 1/4 way from right edge of pano, midway up pano (u,v)=(756,0)
            [-np.pi,0, 1]
        ])

    # theta is ambiguous (-pi and pi are the same), so wrap angles.
    points_sph[:, 0] = np.mod(points_sph[:, 0], 2 * np.pi)
    expected_points_sph[:, 0] = np.mod(expected_points_sph[:, 0], 2 * np.pi)
    import pdb; pdb.set_trace()
    assert np.allclose(points_sph, expected_points_sph)
    

def test_zind_pixel_to_sphere() -> None:
    """
    Assume (512,1024) for (H,W) of equirectangular projection of panorama.
    """
    # fmt: off
    # provided as (u,v) in image
    points_pix = np.array(
        [
            [0,0], # should be (-pi,pi/2)
            [0,511], # should be (-pi, -pi/2)
            [1023,511], # should be (pi, -pi/2)
            [1023,0] # should be (pi, pi/2)
        ])
    
    # provided as (theta,phi) coordinates.
    expected_points_sph = np.array(
        [
            [-np.pi, np.pi/2],
            [-np.pi, -np.pi/2],
            [np.pi, -np.pi/2],
            [np.pi, np.pi/2]
        ])
    # fmt: on
    points_sph = zind_pano_utils.zind_pixel_to_sphere(points_pix, width=1024)
    assert np.allclose(points_sph, expected_points_sph)


def test_sphere_to_pixel() -> None:
    """ """

    points_sph = np.array(
        [
            [-np.pi, np.pi/2],
            [-np.pi, -np.pi/2],
            [np.pi, -np.pi/2],
            [np.pi, np.pi/2]
        ])
    points_px = zind_pano_utils.zind_sphere_to_pixel(points_sph, width=1024)

    # provided as (u,v) in image
    expected_points_px = np.array(
        [
            [0,0], # should be (-pi,pi/2)
            [0,511], # should be (-pi, -pi/2)
            [1023,511], # should be (pi, -pi/2)
            [1023,0] # should be (pi, pi/2)
        ])

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    #test_pano_to_pano_contour_projection()
    #test_zind_sphere_to_cartesian()
    #test_zind_cartesian_to_sphere()
    #test_sphere_to_pixel()
    test_convert_points_px_to_worldmetric_roundtrip()

