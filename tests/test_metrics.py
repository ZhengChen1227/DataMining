"""指标实现的正确性测试。

这些测试的价值在于：指标写错的代价远高于模型写错，
而且错了之后往往「看起来还挺合理」，不会自己暴露。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.metrics import (
    _buffer_labels,
    anomaly_ranges,
    auc_pr,
    auc_roc,
    best_f1,
    compute_metrics,
    point_adjust_f1,
    vus_pr,
)


def test_auc_roc_perfect_and_reversed():
    labels = np.array([0, 0, 1, 1])
    assert auc_roc(labels, np.array([0.1, 0.2, 0.3, 0.4])) == pytest.approx(1.0)
    assert auc_roc(labels, np.array([0.4, 0.3, 0.2, 0.1])) == pytest.approx(0.0)


def test_auc_roc_all_ties_is_half():
    labels = np.array([0, 1, 0, 1])
    scores = np.zeros(4)
    assert auc_roc(labels, scores) == pytest.approx(0.5)


def test_auc_pr_perfect():
    labels = np.array([0, 0, 1, 1])
    assert auc_pr(labels, np.array([0.1, 0.2, 0.3, 0.4])) == pytest.approx(1.0)


def test_auc_pr_matches_known_value():
    # 只有一个正类且排名第一 -> AP = 1.0
    labels = np.array([1, 0, 0, 0])
    assert auc_pr(labels, np.array([0.9, 0.8, 0.7, 0.6])) == pytest.approx(1.0)
    # 正类排在最后 -> AP = 1/4
    assert auc_pr(labels, np.array([0.1, 0.9, 0.8, 0.7])) == pytest.approx(0.25)


def test_best_f1_on_separable_data():
    labels = np.array([0, 0, 1, 1])
    detail = best_f1(labels, np.array([0.1, 0.2, 0.3, 0.4]))
    assert detail["f1"] == pytest.approx(1.0)
    assert detail["precision"] == pytest.approx(1.0)
    assert detail["recall"] == pytest.approx(1.0)


def test_best_f1_no_anomaly_is_nan():
    labels = np.zeros(10, dtype=int)
    assert np.isnan(best_f1(labels, np.random.default_rng(0).normal(size=10))["f1"])


def test_anomaly_ranges():
    labels = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1])
    assert anomaly_ranges(labels) == [(1, 2), (5, 5), (7, 9)]


def test_buffer_labels_extends_ranges():
    labels = np.array([0, 0, 1, 1, 0, 0, 0])
    buffered = _buffer_labels(labels, 1)
    assert list(buffered) == [0, 1, 1, 1, 1, 0, 0]


def test_point_adjust_is_more_optimistic_than_plain_f1():
    labels = np.zeros(20, dtype=int)
    labels[10:] = 1
    scores = np.zeros(20)
    scores[12] = 1.0  # 区间内只检出一个点

    plain = best_f1(labels, scores)["f1"]
    adjusted = point_adjust_f1(labels, scores)["point_adjust_f1"]
    assert plain < 0.3
    assert adjusted == pytest.approx(1.0)


def test_vus_pr_is_monotone_in_buffer_for_shifted_detection():
    labels = np.zeros(200, dtype=int)
    labels[100:120] = 1
    scores = np.zeros(200)
    scores[125:145] = 1.0  # 检出的区间整体滞后 25 个时刻

    base = vus_pr(labels, scores, buffers=[0])["vus_pr"]
    wide = vus_pr(labels, scores, buffers=[30])["vus_pr"]
    assert wide > base


def test_vus_pr_zero_buffer_equals_auc_pr():
    rng = np.random.default_rng(0)
    labels = (rng.random(300) < 0.1).astype(int)
    scores = rng.normal(size=300) + labels * 1.5
    assert vus_pr(labels, scores, buffers=[0])["vus_pr"] == pytest.approx(auc_pr(labels, scores))


def test_compute_metrics_handles_all_normal_series():
    result = compute_metrics(np.zeros(50, dtype=int), np.random.default_rng(0).normal(size=50))
    assert np.isnan(result["auc_pr"])
    assert result["n_anomaly"] == 0


def test_compute_metrics_rejects_length_mismatch():
    with pytest.raises(ValueError):
        compute_metrics(np.zeros(10, dtype=int), np.zeros(9))


@pytest.mark.parametrize("n_pos,n_neg", [(5, 95), (50, 50)])
def test_auc_roc_matches_sklearn(n_pos, n_neg):
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    labels = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
    scores = rng.normal(size=n_pos + n_neg) + labels * 0.8
    assert auc_roc(labels, scores) == pytest.approx(sklearn.roc_auc_score(labels, scores))
    assert auc_pr(labels, scores) == pytest.approx(sklearn.average_precision_score(labels, scores))