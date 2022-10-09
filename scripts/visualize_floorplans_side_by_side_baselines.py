"""Script to visualize floorplan results from different methods, side-by-side."""

from pathlib import Path

import imageio
import matplotlib.pyplot as plt


def visualize_side_by_side(openmvg_dir: str, opensfm_dir: str, salve_dir: str) -> None:
    """Visualize SALVe results side-by-side with OpenSfM and OpenMVG results.

    Args:
        openmvg_dir: path to rendered floorplans, generated using OpenMVG camera localization.
        opensfm_dir: path to rendered floorplans, generated using OpenSfM camera localization.
        salve_dir: path to rendered floorplans, generated using SALVe camera localization.
	"""
    for openmvg_fpath in glob.glob(f"{openmvg_dir}/*.jpg"):

        building_floor_id = Path(openmvg_fpath).stem
        k = building_floor_id.find("_floor")
        building_id = building_floor_id[:k]
        floor_id = building_floor_id[k + 1 :]

        if building_id not in DATASET_SPLITS["test"]:
            continue

        print(f"On Test ID {building_id}")
        opensfm_fpath = f"{opensfm_dir}/{building_id}_{floor_id}.jpg"
        salve_fpath = f"{salve_dir}/{building_id}_{floor_id}.jpg"

        if not Path(opensfm_fpath).exists():
            print("\tOpenSfM result missing.")
            continue

        if not Path(salve_fpath).exists():
            print("\tSALVe result missing.", salve_fpath)
            continue

        openmvg_img = imageio.imread(openmvg_fpath)
        opensfm_img = imageio.imread(opensfm_fpath)
        salve_img = imageio.imread(salve_fpath)

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 3, 1)
        plt.axis("off")
        plt.imshow(openmvg_img)
        plt.subplot(1, 3, 2)
        plt.axis("off")
        plt.imshow(opensfm_img)
        plt.subplot(1, 3, 3)
        plt.axis("off")
        plt.imshow(salve_img)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    openmvg_dir = "/Users/johnlam/Downloads/jlambert-auto-floorplan/openmvg_zind_viz_2021_11_09_largest"
    opensfm_dir = "/Users/johnlam/Downloads/jlambert-auto-floorplan/opensfm_zind_viz_2021_11_09_largest"
    salve_dir = "/Users/johnlam/Downloads/jlambert-auto-floorplan/2021_10_26__ResNet152__435tours_serialized_edge_classifications_test2021_11_02___2021_11_03_pgo_floorplans_with_conf_0.93"
	visualize_side_by_side(openmvg_dir=openmvg_dir, opensfm_dir=opensfm_dir, salve_dir=salve_dir)

