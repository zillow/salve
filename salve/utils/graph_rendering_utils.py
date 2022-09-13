"""Utilities for drawing graph topology, either for a multi-graph, or for a typical undirected graph."""

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx

import salve.dataset.hnet_prediction_loader as hnet_prediction_loader
from salve.algorithms.cycle_consistency import TwoViewEstimationReport
from salve.common.edge_classification import EdgeClassification
from salve.common.posegraph2d import PoseGraph2d

# colors that can be used for coloring nodes or edges
PLUM1 = [173, 127, 168]
SKYBLUE = [135, 206, 250]
GREEN = [0, 140, 25]
CYAN = [0, 255, 255]


def draw_graph_topology(
    edges: List[Tuple[int, int]],
    gt_floor_pose_graph: PoseGraph2d,
    two_view_reports_dict: Dict[Tuple[int, int], TwoViewEstimationReport],
    title: str,
    show_plot: bool = True,
    save_fpath: str = None,
) -> None:
    """Draw the topology of an undirected graph, with vertices placed in their ground truth locations.

    False positive edges are colored red, and true positive edges are colored green.

    Args:
        edges: List of (i1,i2) pairs.
        gt_floor_pose_graph: ground truth 2d pose graph for this floor.
        two_view_reports_dict:
        title: desired title of figure.
        show_plot: whether to show the plot in the matplotlib GUI.
        save_fpath: file path where plot should be saved to disk (if None, will not be saved to disk).
    """
    plt.figure(figsize=(16, 10))
    G = nx.Graph()
    G.add_edges_from(edges)
    nodes = list(G.nodes)
    GREEN = [0, 1, 0]
    RED = [1, 0, 0]
    edge_colors = [GREEN if two_view_reports_dict[edge].gt_class == 1 else RED for edge in edges]

    nx.drawing.nx_pylab.draw_networkx(
        G,
        edgelist=edges,
        edge_color=edge_colors,
        pos={v: gt_floor_pose_graph.nodes[v].global_Sim2_local.translation for v in nodes},
        arrows=True,
        with_labels=True,
    )
    plt.axis("equal")
    plt.title(title)

    if save_fpath is not None:
        plt.savefig(save_fpath, dpi=500)

    if show_plot:
        plt.show()
    plt.close("all")


def draw_multigraph(
    measurements: List[EdgeClassification],
    input_floor_pose_graph: PoseGraph2d,
    raw_dataset_dir: str,
    confidence_threshold: float = 0.5,
    save_dir: str = "./"
) -> None:
    """Draw the topology of a pose graph, with colored nodes and colored edges (allowed edges are conf.-thresholded).

    If the input pose graph is an estimated pose graph, some cameras may not be localized. In this case, we may
    want to render the remaining nodes at the GT locations.

    Args:
        measurements: possible relative pose hypotheses, before confidence thresholding is applied.
        input_floor_pose_graph: poses where graph nodes will be plotted.
        confidence_threshold: minimum required predicted confidence by model to plot an edge.
        raw_dataset_dir: path to where ZInD has been downloaded on disk, from Bridge API.
        save_dir: experiment directory, which will be parent directory for visualizations subdir.
    """

    edges = []
    edge_colors = []

    G = nx.MultiGraph()

    for m in measurements:
        # find all of the predictions where pred class is 1
        if m.y_hat != 1:
            continue

        if m.prob < confidence_threshold:
            continue

        # TODO: this should never happen bc sorted, figure out why it occurs
        if m.i1 >= m.i2:
            i2 = m.i1
            i1 = m.i2

            m.i1 = i1
            m.i2 = i2

        # EDGE_COLOR = SKYBLUE
        EDGE_COLOR = CYAN

        edge_color = [v / 255 for v in EDGE_COLOR]  # "g" if m.y_true == 1 else "r"
        weight = m.prob * 1.5
        # weight = 5 if m.y_true == 1 else 1
        G.add_edge(m.i1, m.i2, color=edge_color, weight=weight)

    NODE_COLOR = [v / 255 for v in GREEN]
    node_color_map = [NODE_COLOR for node in G]
    node_sizes = [15 for node in G]

    edges = G.edges()

    colors = []
    weight = []

    for (u, v, attrib_dict) in list(G.edges.data()):
        colors.append(attrib_dict["color"])
        weight.append(attrib_dict["weight"])

    floor_pose_graphs = hnet_prediction_loader.load_inferred_floor_pose_graphs(
        query_building_id=input_floor_pose_graph.building_id, raw_dataset_dir=raw_dataset_dir
    )
    if floor_pose_graphs is None:
        raise ValueError(
            f"HorizonNet predictions missing for all floors "
            f"of ZInD Building {input_floor_pose_graph.building_id}."
        )
    if input_floor_pose_graph.floor_id not in floor_pose_graphs:
        raise ValueError(
            f"HorizonNet predictions missing for {input_floor_pose_graph.floor_id} "
            f"of ZInD Building {input_floor_pose_graph.building_id}."
        )
    true_gt_floor_pose_graph = floor_pose_graphs[input_floor_pose_graph.floor_id]

    nodes = list(G.nodes)
    node_positions = {}
    for v in nodes:
        if v in input_floor_pose_graph.nodes:
            pos = input_floor_pose_graph.nodes[v].global_Sim2_local.translation
        else:
            pos = true_gt_floor_pose_graph.nodes[v].global_Sim2_local.translation

        node_positions[v] = pos

    nx.drawing.nx_pylab.draw_networkx(
        G=G,
        pos=node_positions,
        edgelist=edges,
        edge_color=colors,
        width=weight,
        with_labels=False,
        node_color=node_color_map,
        node_size=node_sizes,
    )
    building_id = input_floor_pose_graph.building_id
    floor_id = input_floor_pose_graph.floor_id

    plt.axis("equal")
    # plt.show()
    plt.tight_layout()
    multigraph_save_dir = os.path.join(save_dir, f"thresholded_multigraphs_conf{confidence_threshold}")
    os.makedirs(multigraph_save_dir, exist_ok=True)
    save_fpath = os.path.join(multigraph_save_dir, f"multigraph_{building_id}_{floor_id}.pdf")
    plt.savefig(save_fpath, dpi=500)
    plt.close("all")