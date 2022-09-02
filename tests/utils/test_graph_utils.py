"""Unit tests on graph utilites."""

import numpy as np

import salve.utils.graph_utils as graph_utils


def test_find_connected_components1() -> None:
    """Ensures that CCs are computed correctly for 2 cluster scenario. One has 3 nodes, the other is a singleton."""
    nodes = [1, 2, 3, 4]
    edges = [(2, 3), (3, 4)]
    ccs = graph_utils.find_connected_components(nodes, edges)
    expected_ccs = [{1}, {2, 3, 4}]
    assert ccs == expected_ccs


def test_find_connected_components2() -> None:
    """Ensures that CCs are computed correctly for 4 cluster scenario. Two CCs have 2-nodes, and 2 singletons."""
    nodes = [1, 2, 3, 4, 5, 6]
    edges = [(1, 2), (5, 6)]
    ccs = graph_utils.find_connected_components(nodes, edges)
    expected_ccs = [{1, 2}, {3}, {4}, {5, 6}]
    assert ccs == expected_ccs


def test_analyze_cc_distribution() -> None:
    """Ensures that PDF and CDF are computed correctly for 4-connected component case."""
    # Nodes `3` and `4` are orphaned in their own CCs.
    nodes = [1, 2, 3, 4, 5, 6]
    edges = [(1, 2), (5, 6)]

    pdf, cdf = graph_utils.analyze_cc_distribution(nodes=nodes, edges=edges)

    # PDF is ordered from largest CC to smallest CC.
    # CCs contain 2/6, 2/6, 1/6, 1/6 respectively for PDF (percent of nodes in each CC).
    expected_pdf = np.array([2 / 6, 2 / 6, 1 / 6, 1 / 6])
    expected_cdf = np.array([2 / 6, 4 / 6, 5 / 6, 6 / 6])

    assert np.allclose(pdf, expected_pdf)
    assert np.allclose(cdf, expected_cdf)
