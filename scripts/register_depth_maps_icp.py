"""Run simple baseline over all image pairs that uses ICP to fit a relative pose transform.

The ICP is quite slow. We generate point clouds by backprojecting depth maps.

Aligns partial room scans in the spirit of
"Floorplan-jigsaw: Jointly estimating scene layout and aligning partial scans"
(https://arxiv.org/abs/1812.06677).
"""

import glob
from pathlib import Path
from types import SimpleNamespace

import imageio
import matplotlib.pyplot as plt
import open3d

import salve.baselines.open3d_icp as open3d_icp
import salve.common.posegraph2d as posegraph2d
import salve.utils.bev_rendering_utils as bev_rendering_utils


# Parameters for depth map conversion to point cloud.
DEPTH_MAP_ARGS = SimpleNamespace(
    **{
        "scale": 0.001,
        "crop_ratio": 80 / 512,  # throw away top 80 and bottom 80 rows of pixel (too noisy of estimates)
        "crop_z_range": [-10, 10],  # crop_z_range #0.3 # -1.0 # -0.5 # 0.3 # 1.2
        # "crop_z_above": 2
    }
)


def get_pano_fname_from_depthmap_fpath(depthmap_fpath: str) -> str:
    """ """
    return Path(depthmap_fpath).stem.replace(".depth", "") + ".jpg"


def get_pano_id_from_pano_fpath(pano_fpath: str) -> int:
    """Derive integer panorama ID from panoram file path."""
    return int(Path(pano_fpath).stem.split("_")[-1])


def register_pano_pair_by_depthmaps(depthmap_fpath1: str, depthmap_fpath2: str) -> None:
    """Compute relative pose by point cloud (backprojected depth map) registration."""
    depthmap_img1 = imageio.imread(depthmap_fpath1)
    depthmap_img2 = imageio.imread(depthmap_fpath2)

    pano_fpath1 = pano_dirpath + "/" + get_pano_fname_from_depthmap_fpath(depthmap_fpath1)
    pano_fpath2 = pano_dirpath + "/" + get_pano_fname_from_depthmap_fpath(depthmap_fpath2)

    pano_id1 = get_pano_id_from_pano_fpath(pano_fpath1)
    pano_id2 = get_pano_id_from_pano_fpath(pano_fpath2)

    if pano_id1 != 14:
        return

    if pano_id2 != 15:
        return

    if not Path(pano_fpath1).exists() or not Path(pano_fpath2).exists():
        return

    pano_img1 = imageio.imread(pano_fpath1)
    pano_img2 = imageio.imread(pano_fpath2)

    show_2d_viz = False
    if show_2d_viz:
        # Visualize image pair and depthmap pair with Matplotlib.
        plt.subplot(2, 2, 1)
        plt.imshow(pano_img1)
        plt.subplot(2, 2, 2)
        plt.imshow(depthmap_img1)

        plt.subplot(2, 2, 3)
        plt.imshow(pano_img2)
        plt.subplot(2, 2, 4)
        plt.imshow(depthmap_img2)

        plt.show()

    xyzrgb1 = bev_rendering_utils.get_xyzrgb_from_depth(
        args=DEPTH_MAP_ARGS, depth_fpath=depthmap_fpath1, rgb_fpath=pano_fpath1, is_semantics=False
    )
    xyzrgb2 = bev_rendering_utils.get_xyzrgb_from_depth(
        args=DEPTH_MAP_ARGS, depth_fpath=depthmap_fpath2, rgb_fpath=pano_fpath2, is_semantics=False
    )

    pcd1 = open3d_icp.xyzrgb_to_open3d_point_cloud(xyzrgb1)
    pcd2 = open3d_icp.xyzrgb_to_open3d_point_cloud(xyzrgb2)

    # pcd1.estimate_normals(
    #     search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    # )

    show_3d_viz = False
    if show_3d_viz:
        # Visualize each point cloud using Open3d.
        open3d.visualization.draw_geometries([pcd1])
        open3d.visualization.draw_geometries([pcd2])

    # pcd2.estimate_normals(
    #     search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    # )

    print(f"Register {pano_id1} {pano_id2}")
    import pdb; pdb.set_trace()
    use_color = True
    if use_color:
        i2Ti1 = open3d_icp.register_colored_point_clouds(source=pcd1, target=pcd2)
    else:
        i2Ti1 = register_point_clouds(source=pcd1, target=pcd2)

    # Visualize registered point clouds in the same coordinate frame.
    open3d.visualization.draw_geometries([pcd2, pcd1.transform(i2Ti1)])


def register_all_scenes(building_id: str, raw_dataset_dir: str, depthmap_dirpath: str, pano_dirpath: str) -> None:
    """Fit a relative pose transform with colored ICP from every possible panorama pair from specified ZInD building.
    Look for pairs, e.g.
    floor_01_partial_room_01_pano_3.depth.png
    floor_01_partial_room_01_pano_3.jpg

    Args:
        building_id: unique ZInD building id.
        raw_dataset_dir
        depthmap_dirpath:
        pano_dirpath:
    """
    floor_ids = posegraph2d.compute_available_floors_for_building(
        building_id=building_id, raw_dataset_dir=raw_dataset_dir
    )
    for floor_id in floor_ids:

        depthmap_fpaths = glob.glob(f"{depthmap_dirpath}/*{floor_id}*.png")
        depthmap_fpaths.sort()

        # Generate potential panorama pairs.
        pairs = []
        for i1, depthmap_fpath1 in enumerate(depthmap_fpaths):
            for i2, depthmap_fpath2 in enumerate(depthmap_fpaths):
                if i2 <= i1:
                    continue
                pairs.append((depthmap_fpath1, depthmap_fpath2))

        # Attempt ICP on each backprojected depth maps from each panorama pair.
        for (depthmap_fpath1, depthmap_fpath2) in pairs:
            register_pano_pair_by_depthmaps(depthmap_fpath1, depthmap_fpath2)


if __name__ == "__main__":

    building_id = "0000"
    raw_dataset_dir = "/Users/johnlambert/Downloads/zind_bridgeapi_2021_10_05"
    depthmap_dirpath = f"/Users/johnlambert/Downloads/ZinD_Bridge_API_HoHoNet_Depth_Maps/{building_id}"
    pano_dirpath = f"/Users/johnlambert/Downloads/zind_bridgeapi_2021_10_05/{building_id}/panos"

    register_all_scenes(
        building_id=building_id,
        raw_dataset_dir=raw_dataset_dir,
        depthmap_dirpath=depthmap_dirpath,
        pano_dirpath=pano_dirpath,
    )
