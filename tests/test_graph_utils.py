
"""Unit tests on graph utilites."""

import afp.utils.graph_utils as graph_utils

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
