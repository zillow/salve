"""Script to evaluate results from a 3rd party SfM algorithm, such as OpenMVG or OpenSfM.

# TODO: only measure localization precision for connected components with 3+ cameras. (or measure separately)
# 2-camera connected components should always achieve perfect translation alignment under Sim(3)?
(not true -- close -- but rotation also plays a role in it.)
(TODO: write unit tests for this).
"""

import glob
from pathlib import Path

import click
import numpy as np

import salve.common.floor_reconstruction_report as floor_reconstruction_report
import salve.baselines.sfm_eval as sfm_eval
from salve.common.floor_reconstruction_report import FloorReconstructionReport
from salve.dataset.zind_partition import DATASET_SPLITS
from salve.utils.logger_utils import get_logger

logger = get_logger()


def eval_openmvg_errors_all_tours(raw_dataset_dir: str, openmvg_results_dir: str, save_dir: str) -> None:
    """Evaluate the OpenMVG output files (dumped sfm_data.json) from every tour against ZinD ground truth.

    Args:
        raw_dataset_dir: Path to where ZInD dataset is stored (directly downloaded from Bridge API).
        openmvg_results_dir:
        save_dir:
    """
    building_ids = [Path(dirpath).stem for dirpath in glob.glob(f"{raw_dataset_dir}/*")]
    building_ids.sort()
    reconstruction_reports = []

    for building_id in building_ids:
        floor_ids = ["floor_00", "floor_01", "floor_02", "floor_03", "floor_04", "floor_05"]

        if building_id not in DATASET_SPLITS["test"]:
            continue

        for floor_id in floor_ids:

            matches_dirpath = f"{openmvg_results_dir}/ZinD_{building_id}_{floor_id}__openmvg_results/matches"
            if not Path(matches_dirpath).exists():
                # This floor doesn't exist in ZInD, so skip.
                continue

            print(f"On Building {building_id}, {floor_id}")

            reconstruction_json_fpath = (
                f"{openmvg_results_dir}/ZinD_{building_id}_{floor_id}__openmvg_results/reconstruction/sfm_data.json"
            )

            # Whether we want consider failed reconstructions (when OpenMVG times out / runs indefinitely)
            if Path(matches_dirpath).exists() and not Path(reconstruction_json_fpath).exists():
                # import pdb; pdb.set_trace()
                # save_empty_json_results_file(openmvg_results_dir, building_id, floor_id)
                reconstruction_reports.append(
                    FloorReconstructionReport(
                        avg_abs_rot_err=np.nan, avg_abs_trans_err=np.nan, percent_panos_localized=0, floorplan_iou=0.0
                    )
                )

            if not Path(reconstruction_json_fpath).exists():
                continue

            report = sfm_eval.measure_algorithm_localization_accuracy(
                building_id=building_id,
                floor_id=floor_id,
                raw_dataset_dir=raw_dataset_dir,
                algorithm_name="openmvg",
                save_dir=save_dir,
                reconstruction_json_fpath=reconstruction_json_fpath,
            )
            reconstruction_reports.append(report)

    print("OpenMVG test set eval complete.")
    floor_reconstruction_report.summarize_reports(reconstruction_reports)


def eval_opensfm_errors_all_tours(raw_dataset_dir: str, opensfm_results_dir: str, save_dir: str) -> None:
    """Evaluate the OpenSfM output files from every tour against ZinD ground truth. JSON summaries are saved to disk.

    Args:
        raw_dataset_dir: Path to where ZInD dataset is stored (directly downloaded from Bridge API).
        opensfm_results_dir: Location where OpenSfM results are saved, e.g. if the reconstruction result was saved to
            /Users/johnlam/Downloads/OpenSfM/data/ZinD_1442_floor_01/reconstruction.json, then the input arg should
            be /Users/johnlam/Downloads/OpenSfM/data
        save_dir: directory where to store JSON result summaries and visualizations.
    """
    building_ids = [Path(dirpath).stem for dirpath in glob.glob(f"{raw_dataset_dir}/*")]
    building_ids.sort()

    reconstruction_reports = []

    for building_id in building_ids:
        floor_ids = ["floor_00", "floor_01", "floor_02", "floor_03", "floor_04", "floor_05"]

        if building_id not in DATASET_SPLITS["test"]:
            continue

        for floor_id in floor_ids:
            src_pano_dir = f"{raw_dataset_dir}/{building_id}/panos"
            pano_fpaths = glob.glob(f"{src_pano_dir}/{floor_id}_*.jpg")

            if len(pano_fpaths) == 0:
                continue

            print(f"On Building {building_id}, {floor_id}")

            FLOOR_OPENSFM_DATADIR = f"{opensfm_results_dir}/ZinD_{building_id}_{floor_id}__opensfm_results"
            reconstruction_json_fpath = f"{FLOOR_OPENSFM_DATADIR}/reconstruction.json"

            # load_opensfm_reconstructions_from_json(reconstruction_json_fpath)
            report = sfm_eval.measure_algorithm_localization_accuracy(
                building_id=building_id,
                floor_id=floor_id,
                raw_dataset_dir=raw_dataset_dir,
                algorithm_name="opensfm",
                save_dir=save_dir,
                reconstruction_json_fpath=reconstruction_json_fpath,
            )
            reconstruction_reports.append(report)

            # except Exception as e:
            #     logger.exception(f"OpenSfM failed for {building_id} {floor_id}")
            #     print(f"failed on Building {building_id} {floor_id}")
            #     continue

    print("OpenSfM test set eval complete.")
    floor_reconstruction_report.summarize_reports(reconstruction_reports)


@click.command(help="Script to run ")
@click.option(
    "--raw_dataset_dir",
    type=click.Path(exists=True),
    required=True,
    # default="/srv/scratch/jlambert30/salve/zind_bridgeapi_2021_10_05",
    help="Path to where ZInD dataset is stored on disk (after download from Bridge API).",
)
@click.option(
    "--save_dir",
    type=str,
    required=True,
    default="/srv/scratch/jlambert30/salve/ZInD_results_2021_12_11",
    help="Directory where to store JSON result summaries and visualizations.",
)
@click.option(
    "--baseline_name",
    type=click.Choice(["opensfm", "openmvg"]),
    required=True,
    help="Name of SfM library/algorithm to evaluate",
)
@click.option(
    "--results_dir",
    type=click.Path(exists=True),
    # default="/srv/scratch/jlambert30/salve/openmvg_demo_NOSEEDPAIR_UPRIGHTMATCHING__UPRIGHT_ESSENTIAL_ANGULAR/OpenMVG_results_2021_12_03",
    # default="/srv/scratch/jlambert30/salve/OpenSfM_results_2021_12_02_BridgeAPI",
    # default="/Users/johnlam/Downloads/OpenSfM/data/OpenSfM_results_2021_12_02_BridgeAPI"
    required=True,
    help="Location where OpenSfM or OpenMVG raw JSON results are saved (default would be to ~/OpenSfM/data "
    "or OPENMVG_DEMO_ROOT).",
)
def run_evaluate_sfm_baseline(raw_dataset_dir: str, save_dir: str, baseline_name: str, results_dir: str) -> None:
    """Click entry point for camera localization evaluation of third-party SfM libraries.

    # for OpenSFM -- analyze error results dumped to JSON files.
    json_results_dir = "/Users/johnlam/Downloads/salve/opensfm_zind_results"
    """
    save_dir += f"_{baseline_name}"
    if not Path(results_dir).exists():
        raise RuntimeError("Results directory does not exist.")

    if baseline_name == "opensfm":
        eval_opensfm_errors_all_tours(
            raw_dataset_dir=raw_dataset_dir, opensfm_results_dir=results_dir, save_dir=save_dir
        )

    elif baseline_name == "openmvg":
        eval_openmvg_errors_all_tours(
            raw_dataset_dir=raw_dataset_dir, openmvg_results_dir=results_dir, save_dir=save_dir
        )

    # then analyze the mean statistics
    # json_results_dir = "/Users/johnlam/Downloads/salve/openmvg_zind_results"

    # sfm_eval.analyze_algorithm_results(
    #     raw_dataset_dir=raw_dataset_dir, json_results_dir=f"{save_dir}/result_summaries"
    # )


if __name__ == "__main__":

    run_evaluate_sfm_baseline()
