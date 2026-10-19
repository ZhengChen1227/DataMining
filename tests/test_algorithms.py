import inspect
import math

import networkx as nx
import numpy as np
import pytest

from src.algorithms import METHODS, allocate, predict


def test_cn_aa_match_reference():
    g = nx.Graph([(0, 2), (2, 1), (0, 3), (3, 1), (3, 4)])
    pair = np.array([[0, 1]])
    assert predict(g, pair, "cn").scores()[0] == 2
    assert predict(g, pair, "aa").scores()[0] == pytest.approx(1/math.log(2)+1/math.log(3))


@pytest.mark.parametrize("seed", range(5))
def test_full_three_hop_matches_matrix_cube(seed):
    graph = nx.gnp_random_graph(15, .22, seed=seed)
    pairs = np.array(list(nx.non_edges(graph)))
    adjacency = nx.to_numpy_array(graph, nodelist=range(15), dtype=np.int64)
    expected = np.linalg.matrix_power(adjacency, 3)
    result = predict(graph, pairs, "lp")
    assert np.array_equal(result.paths, [expected[u, v] for u, v in pairs])
    assert np.array_equal(result.visits, [graph.degree(u)*graph.degree(v) for u, v in pairs])


def test_chain_and_isolated_node():
    graph = nx.path_graph(4)
    graph.add_node(4)
    result = predict(graph, [[0, 3], [0, 4]], "lp")
    assert result.paths.tolist() == [1, 0]
    assert result.scores().tolist() == [.01, 0]


@pytest.mark.parametrize("method", ["fixed", "ablp", "gate_uniform", "degree_only"])
@pytest.mark.parametrize("budget", [0, 1, 3, 10000])
def test_budget_limits_and_reproducibility(method, budget):
    graph = nx.gnp_random_graph(20, .2, seed=12)
    pairs = np.array(list(nx.non_edges(graph)))
    result = predict(graph, pairs, method, budget)
    assert result.visits.sum() <= len(pairs)*budget
    assert np.array_equal(result.visits, result.allocated)
    assert np.array_equal(result.paths, predict(graph, pairs, method, budget, seed=999).paths)
    if budget == 0:
        assert result.paths.sum() == 0
    if method in ("ablp", "gate_uniform"):
        assert result.visits[result.cn > 0].sum() == 0
    if method == "fixed" and budget == 10000:
        assert np.array_equal(result.paths, predict(graph, pairs, "lp").paths)


def test_allocation_saturates_and_redistributes():
    pairs = np.array([[0, 1], [0, 2], [0, 3]])
    assert allocate([1, 100, 100], [10, 1, 1], 11, pairs).tolist() == [1, 5, 5]
    assert allocate([5, 5, 5], [1, 1, 1], 2, pairs).tolist() == [1, 1, 0]
    assert allocate([5, 5, 5], [0, 0, 0], 10, pairs).sum() == 0
    assert allocate([1, 2, 3], [1, 1, 1], 100, pairs).tolist() == [1, 2, 3]


def test_allocation_invariant_to_candidate_order():
    pairs = np.array([[1, 2], [2, 3], [0, 3]])
    result = allocate([9, 9, 9], [1, 1, 1], 5, pairs)
    permutation = [2, 0, 1]
    other = allocate([9, 9, 9], [1, 1, 1], 5, pairs[permutation])
    assert np.array_equal(other, result[permutation])


def test_reversed_pairs_same_score():
    graph = nx.gnp_random_graph(10, .3, seed=6)
    pairs = np.array(list(nx.non_edges(graph)))
    for method in ("lp", "fixed"):
        assert np.array_equal(predict(graph, pairs, method, 5).paths,
                              predict(graph, pairs[:, ::-1], method, 5).paths)


def test_scoring_has_no_labels_parameter():
    assert set(inspect.signature(predict).parameters) == {"graph", "pairs", "method", "budget", "seed", "deadline"}


def test_deadline_and_invalid_candidate():
    graph = nx.path_graph(4)
    with pytest.raises(TimeoutError):
        predict(graph, [[0, 3]], "lp", deadline=0)
    with pytest.raises(ValueError):
        predict(graph, [[0, 1]], "cn")
