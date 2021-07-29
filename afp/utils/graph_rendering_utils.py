from afp.common.posegraph2d import PoseGraph2d
from afp.algorithms.cycle_consistency import TwoViewEstimationReport

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx


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
        edges: List of (i1,i2) pairs
        gt_floor_pose_graph
        two_view_reports_dict
        title
        show_plot
        save_fpath
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


def plot_multigraph(
    measurements: List[EdgeClassification], gt_floor_pose_graph: PoseGraph2d, confidence_threshold: float = 0.5
):
    """
    Args:
        measurements
        gt_floor_pose_graph
        confidence_threshold
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

        edge_color = "g" if m.y_true == 1 else "r"
        weight = m.prob
        weight = 10 if m.y_true == 1 else 1
        G.add_edge(m.i1, m.i2, color=edge_color, weight=weight)

    edges = G.edges()

    colors = []
    weight = []

    for (u, v, attrib_dict) in list(G.edges.data()):
        colors.append(attrib_dict["color"])
        weight.append(attrib_dict["weight"])

    nodes = list(G.nodes)
    nx.drawing.nx_pylab.draw_networkx(
        G=G,
        pos={v: gt_floor_pose_graph.nodes[v].global_Sim2_local.translation for v in nodes},
        edgelist=edges,
        edge_color=colors,
        width=weight,
    )

    plt.axis("equal")
    plt.show()
