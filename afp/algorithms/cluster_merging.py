
from typing import List, NamedTuple, Set, Tuple

import networkx as nx
import numpy as np

from afp.common.posegraph2d import PoseGraph2d
from afp.algorithms.spanning_tree import greedily_construct_st_Sim2


class EdgeWDOPair(NamedTuple):
    i1: int
    i2: int
    wdo_pair_uuid: str


def get_connected_components(edges: List[Tuple[int, int]]) -> List[Set[int]]:
	"""

	Args:
	    edges: edges of the bi-directional graph.
	Returns:
	    Nodes of each connected component of the input graph, for each connected component
	"""
	if len(edges) == 0:
		return []

	input_graph = nx.Graph()
	input_graph.add_edges_from(edges)

	# get the largest connected component
	ccs = nx.connected_components(input_graph)

	return list(ccs)


def merge_clusters(i2Si1_dict, i2Si1_dict_consistent, per_edge_wdo_dict, gt_floor_pose_graph: PoseGraph2d, two_view_reports_dict):
	""""
	Incremental cluster merging.

	Note: no guarantee there should be high IoU, or zero IoU, unless you knew it was the same room

	Args:
	    i2Si1_dict: low-confidence edges
	    i2Si1_dict_consistent: high-confidence edges

	Returns:
	    est_pose_graph
	"""
	skeleton_nodes = set()
	for (i1,i2) in i2Si1_dict_consistent.keys():
		skeleton_nodes.add(i1)
		skeleton_nodes.add(i2)

	# find connected components of the cleaned up graph
	ccs = get_connected_components(edges=i2Si1_dict_consistent.keys())

	# sort them from large to small
	ccs = sorted(ccs, key=len, reverse=True)

	pano_to_cc_map = {}
	for cc_idx, cc in enumerate(ccs):
		for panoid in cc:
			pano_to_cc_map[panoid] = cc_idx

	# choose two largest clusters
	cc0 = ccs[0]
	cc1 = ccs[1]

	cut_crossings = []
	cut_crossing_confs = []

	# find all edges that could connect these two clusters
	for (i1,i2) in i2Si1_dict.keys():

		if i1 not in skeleton_nodes or i2 not in skeleton_nodes:
			continue

		in_different_ccs = pano_to_cc_map[i1] != pano_to_cc_map[i2]
		# or could say that pano_to_cc_map has to union to [0,1]
		in_ccs_of_interest = (i1 in cc0 or i1 in cc1) and (i2 in cc0 or i2 in cc1)

		# ensure that there are leftover WDOs to actually merge these
		if in_different_ccs and in_ccs_of_interest:
			cut_crossings.append((i1,i2))
			edge_conf = two_view_reports_dict[(i1,i2)].confidence
			cut_crossing_confs.append(edge_conf)

	cut_crossing_confs = np.array(cut_crossing_confs)


	find_unused_WDOs(cut_crossings, gt_floor_pose_graph, per_edge_wdo_dict, i2Si1_dict_consistent)


	for i in range(len(cut_crossing_confs)):

		print(f"\tTrying {i}th most confident crossing of cut...")

		# rank the cut crossings by highest score in the model
		next_most_confident_crossing_idx = np.argsort(-cut_crossing_confs)[i]
		cut_crossing = cut_crossings[next_most_confident_crossing_idx]

		import copy
		i2Si1_dict_consistent_temp = copy.deepcopy(i2Si1_dict_consistent)

		i2Si1_dict_consistent_temp[cut_crossing] = i2Si1_dict[cut_crossing]


		# check if there is any unused door or window or opening
		# if not, then repeat.

		# can we add in another cycle by doing this?
	
		# everything is copy pasted below

		wSi_list = greedily_construct_st_Sim2(i2Si1_dict_consistent_temp, verbose=False)

		if wSi_list is None:
			print(f"Could not build spanning tree, since {len(i2Si1_dict_consistent_temp)} edges in i2Si1 dictionary.")
			print()
			print()
			return

		num_localized_panos = np.array([wSi is not None for wSi in wSi_list]).sum()
		num_floor_panos = len(gt_floor_pose_graph.nodes)
		print(
			f"Localized {num_localized_panos/num_floor_panos*100:.2f}% of panos: {num_localized_panos} / {num_floor_panos}"
		)

		# TODO: try spanning tree version, vs. Shonan version
		wRi_list = [wSi.rotation if wSi else None for wSi in wSi_list]
		wti_list = [wSi.translation if wSi else None for wSi in wSi_list]

		est_floor_pose_graph = PoseGraph2d.from_wRi_wti_lists(
			wRi_list, wti_list, gt_floor_pose_graph, gt_floor_pose_graph.building_id, gt_floor_pose_graph.floor_id
		)

		mean_abs_rot_err, mean_abs_trans_err = est_floor_pose_graph.measure_abs_pose_error(gt_floor_pg=gt_floor_pose_graph)
		print(f"\tAvg translation error: {mean_abs_trans_err:.2f}")
		est_floor_pose_graph.render_estimated_layout(
			show_plot=False,
			save_plot=True,
			plot_save_dir="merged_clusters",
			gt_floor_pg=gt_floor_pose_graph
		)

		from afp.utils.overlap_utils import determine_invalid_wall_overlap

		# import pdb; pdb.set_trace()
		self_intersecting = False
		for pano1_id in cc0:
			for pano2_id in cc1:

				i_wdo = -1
				j_wdo = -1 #dummy values

				pano1_room_vertices = est_floor_pose_graph.nodes[pano1_id].room_vertices_global_2d
				pano2_room_vertices = est_floor_pose_graph.nodes[pano2_id].room_vertices_global_2d

				is_valid = determine_invalid_wall_overlap(
					pano1_id, pano2_id, i_wdo, j_wdo, pano1_room_vertices, pano2_room_vertices, shrink_factor=0.40
				)
				#print(f"({pano1_id},{pano2_id}) is valid? {is_valid}")

				self_intersecting = self_intersecting or not is_valid
		
		if i == 7 or i == 18:
			import pdb; pdb.set_trace()

		# check if there is any self-intersection/penetration of walls/free space, which should not occur
		if not self_intersecting:
			print("Accepted and greedy merge completed.")
			break


	return est_floor_pose_graph



def find_unused_WDOs(cut_crossings, gt_floor_pose_graph, per_edge_wdo_dict, i2Si1_dict_consistent):
	""" """
	used_wdo_dict = defaultdict(dict)

	# figure out all used WDOs

	# then delete cut crossing if it acesses a used WDO








	# for alignment_object in ["windows", "doors", "openings"]:

	# 	for i, pano_data in gt_floor_pose_graph.nodes.items():
	# 		for wdo in getattr(pano_data, alignment_object):

	# 			import pdb; pdb.set_trace()

	# 			# for i, pano_data in gt_floor_pose_graph.nodes.items():
	# 			# 	for wdo in getattr(pano_data, alignment_object):
					
			




