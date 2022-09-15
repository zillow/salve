"""Script to run OpenSfM on ZInD panoramas via system calls."""

import glob
import os
import shutil
from pathlib import Path

import click

# import salve.baselines.opensfm as opensfm_utils
import salve.utils.subprocess_utils as subprocess_utils
from salve.dataset.zind_partition import DATASET_SPLITS
from salve.utils.logger_utils import get_logger

logger = get_logger()


def run_opensfm_over_all_zind(raw_dataset_dir: str, opensfm_repo_root: str, overrides_fpath: str) -> None:
    """Run OpenSfM in spherical geometry mode, over all tours inside ZinD.

    We copy all of the panos from a particular floor of a ZInD building to a directory, and then feed this to OpenSfM.

    Args:
        raw_dataset_dir: path to where ZInD dataset is stored on disk (after download from Bridge API).
        opensfm_repo_root: TODO
        overrides_fpath: path to JSON file with camera override parameters.
    """
    building_ids = [Path(dirpath).stem for dirpath in glob.glob(f"{raw_dataset_dir}/*")]
    building_ids.sort()

    for building_id in building_ids:

        # We are only evaluating OpenSfM on ZInD's test split.
        if building_id not in DATASET_SPLITS["test"]:
            continue

        floor_ids = ["floor_00", "floor_01", "floor_02", "floor_03", "floor_04", "floor_05"]

        for floor_id in floor_ids:
            try:
                src_pano_dir = f"{raw_dataset_dir}/{building_id}/panos"
                pano_fpaths = glob.glob(f"{src_pano_dir}/{floor_id}_*.jpg")

                if len(pano_fpaths) == 0:
                    continue

                floor_opensfm_datadir = (
                    f"{opensfm_repo_root}/data/ZinD_{building_id}_{floor_id}__opensfm_results"
                )
                os.makedirs(f"{floor_opensfm_datadir}/images", exist_ok=True)
                # reconstruction_json_fpath = f"{floor_opensfm_datadir}/reconstruction.json"

                dst_dir = f"{floor_opensfm_datadir}/images"
                for pano_fpath in pano_fpaths:
                    fname = Path(pano_fpath).name
                    dst_fpath = f"{dst_dir}/{fname}"
                    shutil.copyfile(src=pano_fpath, dst=dst_fpath)

                # See https://opensfm.readthedocs.io/en/latest/using.html#providing-your-own-camera-parameters
                shutil.copyfile(overrides_fpath, f"{floor_opensfm_datadir}/camera_models_overrides.json")

                cmd = f"bin/opensfm_run_all {floor_opensfm_datadir} 2>&1 | tee {floor_opensfm_datadir}/opensfm.log"
                print(cmd)
                subprocess_utils.run_command(cmd)

                # Delete copy of all of the copies of the panos.
                shutil.rmtree(dst_dir)

                # Take up way too much space!
                features_dir = f"{floor_opensfm_datadir}/features"
                shutil.rmtree(features_dir)

                # Directory contains resampled-perspective images and depth maps.
                undistorted_depthmaps_dir = f"{floor_opensfm_datadir}/undistorted/depthmaps"
                shutil.rmtree(undistorted_depthmaps_dir)

                undistorted_imgs_dir = f"{floor_opensfm_datadir}/undistorted/images"
                shutil.rmtree(undistorted_imgs_dir)

                # opensfm_utils.load_opensfm_reconstructions_from_json(reconstruction_json_fpath)
                # measure_algorithm_localization_accuracy(
                #     reconstruction_json_fpath, building_id, floor_id, raw_dataset_dir, algorithm_name="opensfm"
                # )

            except Exception as e:
                logger.exception(f"OpenSfM failed for {building_id} {floor_id}")
                print(f"failed on Building {building_id} {floor_id}", e)
                continue


@click.command(help="Script to execute SfM using OpenSfM on ZInD panorama data.")
@click.option(
    "--raw_dataset_dir",
    type=click.Path(exists=True),
    required=True,
    # default="/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05"
    help="Path to where ZInD dataset is stored on disk (after download from Bridge API).",
)
@click.option(
    "--opensfm_repo_root",
    type=click.Path(exists=True),
    required=True,
    # default = "/Users/johnlam/Downloads/OpenSfM"
    help="Path to cloned OpenSfM repository.",
)
def launch_opensfm_over_all_zind(raw_dataset_dir: str, opensfm_repo_root: str) -> None:
    """Click entry point for OpenSfM wrapper."""

    # Path to JSON file with camera override parameters.
    overrides_fpath = (
        Path(__file__).resolve().parent.parent / "salve" / "baselines" / "opensfm" / "camera_models_overrides.json"
    )
    run_opensfm_over_all_zind(raw_dataset_dir, opensfm_repo_root, overrides_fpath)


if __name__ == "__main__":
    """
    cd ~/Downloads/OpenSfM
    python ~/Downloads/jlambert-auto-floorplan/salve/baselines/opensfm.py
    # reconstruction_json_fpath = "/Users/johnlam/Downloads/OpenSfM/data/ZinD_1442_floor_01/reconstruction.json"
    """
    launch_opensfm_over_all_zind()
