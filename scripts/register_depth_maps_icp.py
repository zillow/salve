"""Run simple baseline over all image pairs that uses ICP to fit a relative pose transform.

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

import salve.algorithms.open3d_icp as open3d_icp


def register_all_scenes(depthmap_dirpath: str, pano_dirpath: str) -> None:
    """
    Look for pairs, e.g.
        floor_01_partial_room_01_pano_3.depth.png
        floor_01_partial_room_01_pano_3.jpg

    Args:
        depthmap_dirpath:
        pano_dirpath:
    """
    args = SimpleNamespace(
        **{
            "scale": 0.001,
            "crop_ratio": 80 / 512,  # throw away top 80 and bottom 80 rows of pixel (too noisy of estimates)
            "crop_z_range": [-10, 10],  # crop_z_range #0.3 # -1.0 # -0.5 # 0.3 # 1.2
            # "crop_z_above": 2
        }
    )

    for floor_id in ["floor_00"]:  # , "floor_01", "floor_02", "floor_03"]:

        depthmap_fpaths = glob.glob(f"{depthmap_dirpath}/*{floor_id}*.png")
        depthmap_fpaths.sort()
        
        # Generate potential panorama pairs.
        pairs = []
        for i1, depthmap_fpath1 in enumerate(depthmap_fpaths):
        	for i2, depthmap_fpath2 in enumerate(depthmap_fpaths):
                if i2 <= i1:
                    continue
                pairs.append(depthmap_fpath1, depthmap_fpath2)

        for (depthmap_fpath1, depthmap_fpath2) in pairs:

            depthmap_img1 = imageio.imread(depthmap_fpath1)
            pano_fname1 = Path(depthmap_fpath1).stem.replace(".depth", "") + ".jpg"
            pano_fpath1 = f"{pano_dirpath}/{pano_fname1}"

            if int(Path(pano_fpath1).stem.split("_")[-1]) != 42:
                continue

            if not Path(pano_fpath1).exists():
                continue

            pano_img1 = imageio.imread(pano_fpath1)
            xyzrgb1 = bev_rendering_utils.get_xyzrgb_from_depth(
                args=args, depth_fpath=depthmap_fpath1, rgb_fpath=pano_fpath1, is_semantics=False
            )

            pcd1 = open3d_icp.xyzrgb_to_open3d_point_cloud(xyzrgb1)
            # pcd1.estimate_normals(
            #     search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            # )
            open3d.visualization.draw_geometries([pcd1])

            depthmap_img2 = imageio.imread(depthmap_fpath2)
            pano_fname2 = Path(depthmap_fpath2).stem.replace(".depth", "") + ".jpg"
            pano_fpath2 = f"{pano_dirpath}/{pano_fname2}"

            if int(Path(pano_fpath2).stem.split("_")[-1]) != 43:
                continue

            if not Path(pano_fpath2).exists():
                continue

            pano_img2 = imageio.imread(pano_fpath2)

            plt.subplot(2, 2, 1)
            plt.imshow(pano_img1)
            plt.subplot(2, 2, 2)
            plt.imshow(depthmap_img1)

            plt.subplot(2, 2, 3)
            plt.imshow(pano_img2)
            plt.subplot(2, 2, 4)
            plt.imshow(depthmap_img2)

            plt.show()

            xyzrgb2 = bev_rendering_utils.get_xyzrgb_from_depth(
                args=args, depth_fpath=depthmap_fpath2, rgb_fpath=pano_fpath2, is_semantics=False
            )

            pcd2 = open3d_icp.xyzrgb_to_open3d_point_cloud(xyzrgb2)
            # pcd2.estimate_normals(
            #     search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            # )
            open3d.visualization.draw_geometries([pcd2])

            print(f"Register {i1} {i2}")
            import pdb

            pdb.set_trace()
            i2Ti1 = register_colored_point_clouds(source=pcd1, target=pcd2)
            # i2Ti1 = register_point_clouds(source=pcd1, target=pcd2)

            open3d.visualization.draw_geometries([pcd2, pcd1.transform(i2Ti1)])


if __name__ == "__main__":

    depthmap_dirpath = "/Users/johnlambert/Downloads/ZinD_Bridge_API_HoHoNet_Depth_Maps/0000"
    pano_dirpath = "/Users/johnlambert/Downloads/zind_bridgeapi_2021_10_05/0000/panos"

    register_all_scenes(depthmap_dirpath=depthmap_dirpath, pano_dirpath=pano_dirpath)
