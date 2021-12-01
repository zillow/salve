
"""Utilities for working with graphs"""

from typing import List, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns


def find_connected_components(nodes: List[int], edges: List[Tuple[int,int]]) -> List[Set[int]]:
    """Find connected components of a graph."""
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    ccs = nx.connected_components(G)
    return list(ccs)


def analyze_cc_distribution(nodes: List[int], edges: List[Tuple[int,int]], visualize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Analyze connected component distribution.

    Cannnot derive nodes from edges, as some may be missing.

    Returns:
        array of shape (C,) representing probability density function. values in [0,1]
        array of shape (C,) representing cumulative density function. values in [0,1]
    """
    ccs = find_connected_components(nodes=nodes, edges=edges)
    ccs_increasing = sorted(ccs, key=len)
    ccs_decreasing = ccs_increasing[::-1]

    C = len(ccs)
    cc_sizes = [len(cc) for cc in ccs_decreasing]
    # compute CC localization percents.
    pdf = np.array(cc_sizes) / len(nodes)
    cdf = np.cumsum(pdf)

    if visualize:
        plot_pdf_cdf(pdf, cdf)
    return pdf, cdf


def plot_pdf_cdf(pdf: np.ndarray, cdf: np.ndarray, truncation_limit: int = 5) -> None:
    """
    Args:
        pdf: values in [0,1]
        cdf: values in [0,1]
        truncation_limit: max number of CCs to plot PDF/CDF information about.
    """
    plt.style.use("ggplot")
    #sns.set_style({"font.family": "Times New Roman"})

    palette = np.array(sns.color_palette("hls", 2))

    C = len(pdf)
    C = min(C, truncation_limit)

    # convert from fractions to percentages
    pdf *= 100
    cdf *= 100

    print("First CCs have percentages:")
    print(np.round(pdf))

    plt.plot(range(C), pdf[:C], color=palette[0], label="p.d.f.")
    plt.plot(range(C), cdf[:C], color=palette[1], label="c.d.f.")
    plt.scatter(range(C), pdf[:C], 100, color=palette[0], marker='.')
    plt.scatter(range(C), cdf[:C], 100, color=palette[1], marker='.')
    plt.xticks(np.arange(C))
    plt.xlabel(r"$i$'th Connected Component")
    plt.ylabel("% of Panoramas Localized")
    plt.ylim([0,100])
    plt.legend()
    plt.savefig("2021_11_17_conn_comp_distribution_v1.pdf")
    plt.show()

