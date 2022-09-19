"""Sanity check that GT pose graphs can be loaded for all floors of all buildings of ZInD."""

from pathlib import Path

import click

import salve.common.posegraph2d as posegraph2d


def sanity_check_dataset_pose_graphs(raw_dataset_dir: str) -> None:
    """ """
    building_ids = [Path(fpath).stem for fpath in glob.glob(f"{raw_dataset_dir}/*") if Path(fpath).is_dir()]
    building_ids.sort()

    for building_id in building_ids:

        if building_id != "000":  # '1442':
            continue

        print(f"Render floor maps for {building_id}")
        pano_dir = f"{raw_dataset_dir}/{building_id}/panos"
        json_annot_fpath = f"{raw_dataset_dir}/{building_id}/zind_data.json"
        floor_pg_dict = posegraph2d.get_single_building_pose_graphs(
            building_id=building_id, pano_dir=pano_dir, json_annot_fpath=json_annot_fpath
        )


@click.command(help="Script to run batched depth map inference using a pretrained HoHoNet model.")
@click.option(
    "--raw_dataset_dir",
    type=click.Path(exists=True),
    required=True,
    help="Path to where ZInD dataset is stored on disk (after download from Bridge API).",
)
def run_sanity_check_dataset_pose_graphs()
	"""Click entry point for ..."""
	sanity_check_dataset_pose_graphs(raw_dataset_dir=raw_dataset_dir)


if __name__ == "__main__":
    """ """
    run_sanity_check_dataset_pose_graphs()