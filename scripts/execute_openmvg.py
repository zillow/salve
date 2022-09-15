"""Script to execute OpenMVG's incremental SfM on ZInD's 360 degree equirectangular panoramas.

Reference:
https://github.com/openMVG/openMVG/blob/develop/docs/sphinx/rst/software/SfM/SfM.rst#notes-about-spherical-sfm
"""

import glob
import os
import shutil
from pathlib import Path

import click

import salve.utils.subprocess_utils as subprocess_utils
from salve.common.posegraph2d import REDTEXT, ENDCOLOR
from salve.dataset.zind_partition import DATASET_SPLITS
from salve.utils.function_timeout import timeout


def run_openmvg_commands_single_tour(
    openmvg_sfm_bin: str, image_dirpath: str, matches_dirpath: str, reconstruction_dirpath: str
) -> None:
    """Run OpenMVG over a single floor of a single building, by sequentially executing binaries for each SfM stage.

    The reconstruction result / estimated global poses will be written to a file named "sfm_data.json".
    Alternative: try OpenSfM with 3 different seed pairs, and choose the best among all.

    Args:
        openmvg_sfm_bin: path to directory containing all compiled OpenMVG binaries.
        image_dirpath: path to temp. directory containing all images to use for reconstruction of a single floor.
        matches_dirpath: path to directory where matches info will be stored.
        reconstruction_dirpath: path to directory where reconstruction info will be stored.
    """
    # seed_fname1, seed_fname2 = find_seed_pair(image_dirpath)

    use_spherical_angular = False
    use_spherical_angular_upright = True

    # Configure the scene to use the Spherical camera model and a unit focal length
    # "-c" is "camera_model" and "-f" is "focal_pixels"
    # defined here: https://github.com/openMVG/openMVG/blob/develop/src/openMVG/cameras/Camera_Common.hpp#L48
    cmd = f"{openmvg_sfm_bin}/openMVG_main_SfMInit_ImageListing -i {image_dirpath} -o {matches_dirpath} -c 7 -f 1"
    stdout, stderr = subprocess_utils.run_command(cmd, return_output=True)
    print("STDOUT: ", stdout)
    print("STDERR: ", stderr)

    # Extract the features (using the HIGH preset is advised, since the spherical image introduced distortions)
    # Can also pass "-u 1" for upright, per:
    #     https://github.com/openMVG/openMVG/blob/develop/docs/sphinx/rst/software/SfM/ComputeFeatures.rst
    cmd = f"{openmvg_sfm_bin}/openMVG_main_ComputeFeatures"
    cmd += f" -i {matches_dirpath}/sfm_data.json -o {matches_dirpath} -m SIFT -p HIGH -u 1"
    stdout, stderr = subprocess_utils.run_command(cmd, return_output=True)
    print("STDOUT: ", stdout)
    print("STDERR: ", stderr)

    # Computes the matches (using the Essential matrix with an angular constraint)
    # can also pass the "-u" for upright, per https://github.com/openMVG/openMVG/issues/1731
    # "a: essential matrix with an angular parametrization,\n"
    # See https://github.com/openMVG/openMVG/blob/develop/src/software/SfM/main_ComputeMatches.cpp#L118
    # "u: upright essential matrix.\n"
    # GeometricFilter_ESphericalMatrix_AC_Angular
    # https://github.com/openMVG/openMVG/blob/develop/src/software/SfM/main_ComputeMatches.cpp#L496
    # ESSENTIAL_MATRIX_ANGULAR -> GeometricFilter_ESphericalMatrix_AC_Angular<false>
    # ESSENTIAL_MATRIX_UPRIGHT -> GeometricFilter_ESphericalMatrix_AC_Angular<true>
    # https://github.com/openMVG/openMVG/blob/develop/src/openMVG/matching_image_collection/E_ACRobust_Angular.hpp
    cmd = f"{openmvg_sfm_bin}/openMVG_main_ComputeMatches -i {matches_dirpath}/sfm_data.json -o {matches_dirpath}"
    if use_spherical_angular:
        cmd += " -g a"
    elif use_spherical_angular_upright:
        cmd += " -g u"

    stdout, stderr = subprocess_utils.run_command(cmd, return_output=True)
    print("STDOUT: ", stdout)
    print("STDERR: ", stderr)

    # Compute the reconstruction
    cmd = f"{openmvg_sfm_bin}/openMVG_main_IncrementalSfM -i {matches_dirpath}/sfm_data.json"
    cmd += f" -m {matches_dirpath} -o {reconstruction_dirpath}"  # " -a {seed_fname1} -b {seed_fname2}"
    # Since the spherical geometry is different than classic pinhole images, the best is to provide the initial pair
    # by hand with the -a -b image basenames (i.e. R0010762.JPG).

    try:
        with timeout(seconds=60 * 5):
            stdout, stderr = subprocess_utils.run_command(cmd, return_output=True)
            print("STDOUT: ", stdout)
            print("STDERR: ", stderr)
    except TimeoutError:
        print("Execution timed out, OpenMVG is stuck")
        return

    # If execution successful, convert result from binary file to JSON
    input_fpath = f"{reconstruction_dirpath}/sfm_data.bin"
    output_fpath = f"{reconstruction_dirpath}/sfm_data.json"
    # Convert the "VIEWS", "INTRINSICS", "EXTRINSICS" with -V -I -E
    cmd = f"{openmvg_sfm_bin}/openMVG_main_ConvertSfM_DataFormat -i {input_fpath} -o {output_fpath} -V -I -E"
    subprocess_utils.run_command(cmd)


def run_openmvg_all_tours(raw_dataset_dir: str, openmvg_sfm_bin: str, openmvg_demo_root: str) -> None:
    """Run OpenMVG in spherical geometry mode, over all tours inside ZinD.

    We copy all of the panos from a particular floor of a ZInD building to a directory, and then feed this to OpenMVG.

    Args:
        raw_dataset_dir: Path to where ZInD dataset is stored on disk (after download from Bridge API)
        openmvg_sfm_bin: Path to directory containing all compiled OpenMVG binaries.
        openmvg_demo_root
    """

    building_ids = [Path(dirpath).stem for dirpath in glob.glob(f"{raw_dataset_dir}/*")]
    building_ids.sort()

    for building_id in building_ids:

        # We are only evaluating OpenMVG on ZInD's test split.
        if building_id not in DATASET_SPLITS["test"]:
            continue

        floor_ids = ["floor_00", "floor_01", "floor_02", "floor_03", "floor_04", "floor_05"]

        try:
            # Verify if building directory is legitimate (whether we can cast the building ID to an integer).
            _ = int(building_id)
        except Exception as e:
            print(f"Invalid building id {building_id}, skipping...", e)
            continue

        for floor_id in floor_ids:
            print(f"Running OpenMVG on {building_id}, {floor_id}")

            floor_openmvg_datadir = f"{openmvg_demo_root}/ZinD_{building_id}_{floor_id}__2021_12_02"

            matches_dirpath = f"{floor_openmvg_datadir}/matches"
            reconstruction_json_fpath = f"{floor_openmvg_datadir}/reconstruction/sfm_data.json"
            if Path(reconstruction_json_fpath).exists() or Path(matches_dirpath).exists():
                print(f"\tResults already exists for Building {building_id}, {floor_id}, skipping...")
                continue

            src_pano_dir = f"{raw_dataset_dir}/{building_id}/panos"
            pano_fpaths = glob.glob(f"{src_pano_dir}/{floor_id}_*.jpg")

            if len(pano_fpaths) == 0:
                print(REDTEXT + f"\tFloor {floor_id} does not exist for building {building_id}, skipping" + ENDCOLOR)
                continue

            os.makedirs(f"{floor_openmvg_datadir}/images", exist_ok=True)

            # Make a copy of all of the panos.
            dst_dir = f"{floor_openmvg_datadir}/images"
            for pano_fpath in pano_fpaths:
                fname = Path(pano_fpath).name
                dst_fpath = f"{dst_dir}/{fname}"
                shutil.copyfile(src=pano_fpath, dst=dst_fpath)

            matches_dirpath = f"{floor_openmvg_datadir}/matches"
            reconstruction_dirpath = f"{floor_openmvg_datadir}/reconstruction"

            run_openmvg_commands_single_tour(
                openmvg_sfm_bin=openmvg_sfm_bin,
                image_dirpath=dst_dir,  # [full path image directory]
                matches_dirpath=matches_dirpath,  # [matches directory]
                reconstruction_dirpath=reconstruction_dirpath,  # [reconstruction directory]
            )
            # Delete copy of all of the copies of the panos.
            shutil.rmtree(dst_dir)


@click.command(help="Script to execute SfM using OpenMVG on ZInD panorama data.")
@click.option(
    "--raw_dataset_dir",
    type=click.Path(exists=True),
    required=True,
    # default="/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05"
    help="Path to where ZInD dataset is stored on disk (after download from Bridge API).",
)
@click.option(
    "--openmvg_sfm_bin",
    type=click.Path(exists=True),
    required=True,
    # default="/Users/johnlam/Downloads/openMVG_Build/Darwin-x86_64-RELEASE"
    help="Path to directory containing all compiled OpenMVG binaries.",
)
@click.option(
    "--openmvg_demo_root",
    type=click.Path(exists=True),
    required=True,
    # default = "/Users/johnlam/Downloads/openmvg_demo_NOSEEDPAIR_UPRIGHTMATCHING"
    # default = "/Users/johnlam/Downloads/openmvg_demo_NOSEEDPAIR_UPRIGHTMATCHING__UPRIGHT_ESSENTIAL_ANGULAR"
    help="Path to OpenMVG output",
)
def launch_openmvg_on_all_tours(raw_dataset_dir: str, openmvg_sfm_bin: str, openmvg_demo_root: str) -> None:
    """Click entry point for OpenMVG execution on the ZInD dataset."""
    run_openmvg_all_tours(
        raw_dataset_dir=str(raw_dataset_dir), openmvg_sfm_bin=openmvg_sfm_bin, openmvg_demo_root=openmvg_demo_root
    )


if __name__ == "__main__":
    launch_openmvg_on_all_tours()
