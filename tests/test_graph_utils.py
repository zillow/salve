
"""Unit tests on graph utilites."""

import numpy as np

import salve.utils.graph_utils as graph_utils

def test_find_connected_components1() -> None:
    """Two clusters. One has three nodes, the other is a singleton."""
    nodes = [1,2,3,4]
    edges = [(2,3),(3,4)]
    ccs = graph_utils.find_connected_components(nodes, edges)
    expected_ccs = [{1}, {2, 3, 4}]
    assert ccs == expected_ccs


def test_find_connected_components2() -> None:
    """Four clusters. Two have 2-nodes, and 2 singletons."""
    nodes = [1,2,3,4,5,6]
    edges = [(1,2), (5,6)]
    ccs = graph_utils.find_connected_components(nodes, edges)
    expected_ccs = [{1,2}, {3}, {4}, {5,6}]
    assert ccs == expected_ccs


def test_analyze_cc_distribution() -> None:
    """
    For four connected components, examine PDF / CDF of nodes present, from largest to smallest CCs.
    """
    nodes = [1,2,3,4,5,6]
    edges = [(1,2), (5,6)]

    pdf, cdf = graph_utils.analyze_cc_distribution(nodes=nodes, edges=edges)


    # contain 2/6, 2/6, 1/6, 1/6 for PDF (percent of nodes in each CC)
    expected_pdf = np.array([2/6, 2/6, 1/6, 1/6])
    expected_cdf = np.array([2/6, 4/6, 5/6, 6/6 ])

    assert np.allclose(pdf, expected_pdf)
    assert np.allclose(cdf, expected_cdf)


if __name__ == "__main__":
    test_analyze_cc_distribution()

    #test_find_connected_components1()
    test_find_connected_components2()

