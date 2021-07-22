
from typing import List, NamedTuple, Set, Tuple

import networkx as nx
import numpy as np

from afp.common.posegraph2d import PoseGraph2d


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

	import pdb; pdb.set_trace()
	# rank the cut crossings by highest score in the model
	most_confident_crossing_idx = np.argmax(cut_crossing_confs)
	cut_crossing = cut_crossings[most_confident_crossing_idx]

	find_unused_WDOs(gt_floor_pose_graph, per_edge_wdo_dict, i2Si1_dict_consistent)

	# check if there is any unused door or window or opening
	# if not, then repeat.

	# can we add in another cycle by doing this?
	
	
	# check if there is any self-intersection/penetration of walls/free space, which should not occur


	return est_pose_graph



def find_unused_WDOs(gt_floor_pose_graph, per_edge_wdo_dict, i2Si1_dict_consistent):
	""" """



