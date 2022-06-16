"""Measure the average scaling factor to convert from world-normalized Cartesian to world-metric Cartesian."""

import argparse
import glob

import gtsfm.utils.io as io_utils
import numpy as np


def main(data_root: str) -> None:
    """
    Measured over 2453 scales from 1575 JSON files.
    Min:  2.344799085543489
    Mean:  3.5083251091357224
    Median:  3.554684877476505
    Max:  5.465154984596182
    """
    all_valid_scales = []

    gt_fpaths = glob.glob(f"{data_root}/**/zind_data.json")
    for gt_fpath in gt_fpaths:
        building_data = io_utils.read_json_file(gt_fpath)

        scales_dict = building_data["scale_meters_per_coordinate"]

        valid_scales = [v for v in scales_dict.values() if v is not None]

        all_valid_scales.extend(valid_scales)

    print(f"Measured over {len(all_valid_scales)} scales from {len(gt_fpaths)} JSON files.")
    print("Min: ", np.amin(all_valid_scales))
    print("Mean: ", np.mean(all_valid_scales))
    print("Median: ", np.median(all_valid_scales))
    print("Max: ", np.amax(all_valid_scales))
    import pdb

    pdb.set_trace()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05",
        help="dirpath to downloaded ZInD dataset.",
    )
    args = parser.parse_args()
    main(args.data_root)
