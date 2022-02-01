
"""Growing Consensus Algorithm.

Reference:
https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Son_Solving_Small-Piece_Jigsaw_CVPR_2016_paper.pdf
"""

def main() -> None:
    """ """
    elif method == "growing_consensus":
    growing_consensus(
        building_id,
        floor_id,
        i2Si1_dict,
        i2Ri1_dict,
        i2Ui1_dict,
        gt_edges,
        two_view_reports_dict,
        gt_floor_pose_graph,
    )


def find_max_degree_vertex(i2Ti1_dict: Dict[Tuple[int, int], Any]) -> int:
    """Find the node inside of a graph G=(V,E) with highest degree.

    Args:
        i2Ti1_dict: edges E of a graph G=(V,E)

    Returns:
        seed_node: integer id of
    """
    # find the seed (vertex with highest degree)
    adj_list = cycle_utils.create_adjacency_list(i2Ti1_dict)
    seed_node = -1
    max_neighbors = 0
    for v, neighbors in adj_list.items():
        if len(neighbors) > max_neighbors:
            seed_node = v
            max_neighbors = len(neighbors)

    return seed_node


def growing_consensus(
    building_id: str,
    floor_id: str,
    i2Si1_dict: Dict[Tuple[int, int], Sim2],
    i2Ri1_dict: Dict[Tuple[int, int], np.ndarray],
    i2Ui1_dict: Dict[Tuple[int, int], np.ndarray],
    gt_edges,
    two_view_reports_dict: Dict[Tuple[int, int], TwoViewEstimationReport],
    gt_floor_pose_graph: PoseGraph2d,
) -> None:
    """Implements a variant of the Growing Consensus" algorithm described below:

    Args:
    """
    i2Ri1_dict = {(i1, i2): i2Si1.rotation for (i1, i2), i2Si1 in i2Si1_dict.items()}

    seed_node = find_max_degree_vertex(i2Si1_dict)
    unused_triplets = cycle_utils.extract_triplets(i2Si1_dict)
    unused_triplets = set(unused_triplets)

    seed_cycle_errors = []
    # compute cycle errors for each triplet connected to seed_node
    connected_triplets = [t for t in unused_triplets if seed_node in t]
    for triplet in connected_triplets:
        cycle_error, _, _ = cycle_utils.compute_rot_cycle_error(
            i2Ri1_dict,
            cycle_nodes=triplet,
            two_view_reports_dict=two_view_reports_dict,
            verbose=True,
        )
        seed_cycle_errors.append(cycle_error)

    import pdb

    pdb.set_trace()

    min_error_triplet_idx = np.argmin(np.array(seed_cycle_errors))
    seed_triplet = connected_triplets[min_error_triplet_idx]
    # find one with low cycle_error

    unused_triplets.remove(seed_triplet)

    edges = [(i0, i1), (i1, i2), (i0, i2)]

    i2Si1_dict_consensus = {edge: i2Si1_dict[edge] for edge in edges}

    while True:
        candidate_consensus = copy.deepcopy(i2Si1_dict_consensus)

        for triplet in unused_triplets:
            import pdb

            pdb.set_trace()

        # can we do this? i3Ui3 = i3Ri2 * i2Ui3 = i2Ri1 * i1Ui3 (cycle is 3->1->2)

        # compute all cycle errors with these new edges added
        # if all errors are low, then:
        # unused_triplets.remove(triplet)
