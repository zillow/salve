


"""
Global alignment.
"""

import glob
from pathlib import Path
from typing import NamedTuple

from argoverse.utils.json_utils import read_json_file

from cycle_consistency import filter_to_cycle_consistent_edges


class TwoViewEdge(NamedTuple)

def main():
	""" """

	label_dict = [
		"incorrect_alignment": 0,
		"gt_alignment_approx": 1
	]

	#hypotheses_dir = "/Users/johnlam/Downloads/ZinD_alignment_hypotheses_2021_06_25"
	hypotheses_dir = "/Users/johnlam/Downloads/DGX-rendering-2021_06_25/ZinD_alignment_hypotheses_2021_06_25"

	building_ids = [Path(dirpath).stem for dirpath in glob.glob(f"{hypotheses_dir}/*")]

	for building_id in building_ids:

		floor_ids = [Path(dirpath) for dirpath in glob.glob(f"{hypotheses_dir}/{building_id}/*")]
		for floor_id in floor_ids:

			floor_label_idxs = []
			floor_sim2_json_fpaths = []

			for label_type, label_idx in label_dict.items():
				label_json_fpaths = glob.glob(f"{hypotheses_dir}/{building_id}/{floor_id}/{label_type}/*")
				label_idxs = [label_idx] * len(label_json_fpaths)

				sim2_json_fpaths.extend(label_json_fpaths)
				floor_label_idxs.extend(label_idxs)

			floor_label_idxs = np.array(floor_label_idxs)

			import pdb; pdb.set_trace()

			# TODO: cache all of the model results beforehand (suppose we randomly pollute 8.5% of the results)
			POLLUTION_FRAC = 0.085

			num_floor_labels = len(floor_label_idxs)
			idxs_to_pollute = np.random.choice(a=num_floor_labels, size=int(POLLUTION_FRAC * num_floor_labels))

			floor_label_idxs[idxs_to_pollute] = 1 - floor_label_idxs[idxs_to_pollute]

			# for a single floor, find all of the triplets
			two_view_reports_dict = {}

			i2Ri1_dict = {}
			i2Ui1_dict = {}

			# check if the triplet is self consistent

			i2Ri1_consistent, i2Ui1_consistent = filter_to_cycle_consistent_edges(
    			i2Ri1_dict,
    			i2Ui1_dict,
    			two_view_reports_dict,
    			visualize=False
    		)

			# if so, admit the 3 edges to the graph





if __name__ == "__main__":
	main()


