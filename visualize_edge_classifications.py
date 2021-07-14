
import glob
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
from argoverse.utils.json_utils import read_json_file

from posegraph2d import PoseGraph2d, get_gt_pose_graph
from pr_utils import assign_tp_fp_fn_tn


class EdgeClassification(NamedTuple):
	"""
	i1 and i2 are panorama id's
	"""
	i1: int
	i2: int
	prob: float
	y_hat: int
	y_true: int
	pair_idx: int


def get_edge_classifications_from_serialized_preds(serialized_preds_json_dir: str) -> Dict[Tuple[str,str], List[EdgeClassification]]:
	"""

	Args:
	    serialized_preds_json_dir: 

	Returns:
	    floor_edgeclassifications_dict
	"""
	floor_edgeclassifications_dict = defaultdict(list)

	json_fpaths = glob.glob(f"{serialized_preds_json_dir}/batch*.json")
	for json_fpath in json_fpaths:

		json_data = read_json_file(json_fpath)
		y_hat_list = json_data["y_hat"]
		y_true_list = json_data["y_true"]
		y_hat_prob_list = json_data["y_hat_probs"]
		fp0_list = json_data["fp0"]
		fp1_list = json_data["fp1"]
		fp2_list = json_data["fp2"]
		fp3_list = json_data["fp3"]

		for y_hat, y_true, y_hat_prob, fp0, fp1, fp2, fp3 in zip(y_hat_list, y_true_list, y_hat_prob_list, fp0_list, fp1_list, fp2_list, fp3_list):
			i1 = int(Path(fp0).stem.split('_')[-1])
			i2 = int(Path(fp1).stem.split('_')[-1])
			building_id = Path(fp0).parent.stem

			s = Path(fp0).stem.find('floor_')
			e = Path(fp0).stem.find('_partial')
			floor_id = Path(fp0).stem[s:e]

			pair_idx = Path(fp0).stem.split('_')[1]

			floor_edgeclassifications_dict[(building_id, floor_id)] += [EdgeClassification(i1, i2, y_hat_prob, y_hat, y_true, pair_idx)]
	return floor_edgeclassifications_dict


def vis_edge_classifications(serialized_preds_json_dir: str, raw_dataset_dir: str) -> None:
	""" """
	floor_edgeclassifications_dict = get_edge_classifications_from_serialized_preds(serialized_preds_json_dir)

	color_dict = {
		'TP': 'green',
		'FP': 'red',
		'FN': 'orange',
		'TN': 'blue'
	}

	# loop over each building and floor
	for (building_id, floor_id), measurements in floor_edgeclassifications_dict.items():

		if building_id != '1490': # '1394':# '1635':
			continue

		print(f"On building {building_id}, {floor_id}")
		gt_floor_pose_graph = get_gt_pose_graph(building_id, floor_id, raw_dataset_dir)

		# gather all of the edge classifications
		y_hat = np.array([m.y_hat for m in measurements])
		y_true = np.array([m.y_true for m in measurements])

		# classify into TPs, FPs, FNs, TNs
		is_TP, is_FP, is_FN, is_TN = assign_tp_fp_fn_tn(y_true, y_pred=y_hat)
		for m, is_tp, is_fp, is_fn, is_tn in zip(measurements, is_TP, is_FP, is_FN, is_TN):

			# then render the edges
			if is_tp:
				color = color_dict['TP']

			elif is_fp:
				color = color_dict['FP']
				gt_floor_pose_graph.draw_edge(m.i1, m.i2, color)
				print(f"\tFP: ({m.i1},{m.i2}) for pair {m.pair_idx}")
			elif is_fn:
				color = color_dict['FN']
				#gt_floor_pose_graph.draw_edge(m.i1, m.i2, color)
			elif is_tn:
				color = color_dict['TN']

			# if m.i1 or m.i2 not in gt_floor_pose_graph.nodes:
			# 	import pdb; pdb.set_trace()
			

		#import pdb; pdb.set_trace()
		# render the pose graph first
		gt_floor_pose_graph.render_estimated_layout()
		#continue



def run_incremental_reconstruction(serialized_preds_json_dir: str, raw_dataset_dir: str):
	""" """
	floor_edgeclassifications_dict = get_edge_classifications_from_serialized_preds(serialized_preds_json_dir)

	# loop over each building and floor
	# for each building/floor tuple
	for (building_id, floor_id), measurements in floor_edgeclassifications_dict.items():

		print(f"On building {building_id}, {floor_id}")

		# find all of the predictions where pred class is 1

		#look up the associated Sim(2) file for this prediction, by looping through the pair idxs again


if __name__ == "__main__":
	""" """
	#serialized_preds_json_dir = "/Users/johnlam/Downloads/2021_07_13_binary_model_edge_classifications"
	serialized_preds_json_dir = "/Users/johnlam/Downloads/2021_07_13_edge_classifications_fixed_argmax_bug/2021_07_13_edge_classifications_fixed_argmax_bug"

	raw_dataset_dir = "/Users/johnlam/Downloads/ZInD_release/complete_zind_paper_final_localized_json_6_3_21"
	#raw_dataset_dir = "/Users/johnlam/Downloads/2021_05_28_Will_amazon_raw"
	vis_edge_classifications(serialized_preds_json_dir, raw_dataset_dir)

	run_incremental_reconstruction(serialized_preds_json_dir, raw_dataset_dir)



