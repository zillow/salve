import argparse
import csv
import json
import os
import tqdm


def main(csv_path, out_dir) -> None:
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        vanishing_angles = {}
        for i_row, row in enumerate(csv_reader):
            if i_row == 0:
                continue

            i_building, pano_id, degree = row
            building_id = "%04d" % int(i_building)
            pano_id = pano_id.split('.')[0]
            degree = float(degree)

            if building_id not in vanishing_angles:
                vanishing_angles[building_id] = {}

            vanishing_angles[building_id][pano_id] = degree

    for building_id, vps in tqdm.tqdm(vanishing_angles.items()):
        with open(os.path.join(out_dir, f'{building_id}.json'), 'w') as f:
            json.dump(vps, f)
    print("Vanishing angle extraction complete.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to vanishing angle csv file.",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Directory path to splited json files for vanishing angles.",
    )
    args = parser.parse_args()
    main(args.csv, args.out)
