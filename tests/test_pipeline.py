"""数据管线、漂移协议与端到端流程的测试。"""

from __future__ import annotations

import numpy as np
import pytest

from src.datasets import Bundle, Series, make_synthetic, normalize
from src.datasets.tsb_ad import list_subsets
from src.drift import build_drift, distribution_gap, list_protocols
from src.eval import run_experiment
from src.models import build_model, list_models
from src.models.window import make_windows, window_scores_to_timesteps
from src.utils.config import Config, load_config, set_by_path


@pytest.fixture(scope="module")
def bundle() -> Bundle:
    return make_synthetic(n_series=2, n_channels=6, n_train=800, n_test=400, seed=0)


def test_synthetic_bundle_shapes(bundle):
    assert len(bundle) == 2
    for series in bundle:
        assert series.train.shape == (800, 6)
        assert series.test.shape == (400, 6)
        assert series.train_label.shape == (800,)
        assert series.test_label.shape == (400,)
        assert series.anomaly_ratio > 0


def test_series_rejects_channel_mismatch():
    with pytest.raises(ValueError):
        Series(
            name="bad",
            train=np.zeros((10, 3)),
            train_label=np.zeros(10),
            test=np.zeros((10, 4)),
            test_label=np.zeros(10),
        )


@pytest.mark.parametrize("name", ["none", "temporal", "channel", "channel_mask", "amplitude"])
def test_drift_protocols_keep_shape_and_labels(bundle, name):
    drifted = build_drift({"name": name}, seed=0).apply(bundle)
    for original, series in zip(bundle, drifted):
        assert series.test.shape == original.test.shape
        assert np.array_equal(series.test_label, original.test_label)
        assert series.meta["drift"]["name"] == name


@pytest.mark.parametrize("name", ["temporal", "channel", "channel_mask", "amplitude"])
def test_drift_is_deterministic(bundle, name):
    first = build_drift({"name": name}, seed=7).apply(bundle)
    second = build_drift({"name": name}, seed=7).apply(bundle)
    for a, b in zip(first, second):
        assert np.allclose(a.train, b.train)
        assert np.allclose(a.test, b.test)


def test_channel_permutation_preserves_marginals(bundle):
    drifted = build_drift({"name": "channel"}, seed=3).apply(bundle)
    for original, series in zip(bundle, drifted):
        # 置换不改变任何单通道的边缘分布（排序后逐通道比较）
        assert np.allclose(
            np.sort(series.test, axis=0), np.sort(original.test, axis=0), atol=1e-6
        )
        # 但改变了通道间关系
        assert not np.allclose(series.test, original.test)


def test_amplitude_drift_actually_shifts(bundle):
    normalized = normalize(bundle, "zscore")
    drifted = build_drift({"name": "amplitude"}, seed=1).apply(normalized)
    for original, series in zip(normalized, drifted):
        assert not np.allclose(series.test, original.test)
        assert distribution_gap(series.test, original.train) > 0.0


def test_temporal_drift_shrinks_train(bundle):
    drifted = build_drift({"name": "temporal", "params": {"n_segments": 5, "keep": 1}}, seed=0).apply(bundle)
    for original, series in zip(bundle, drifted):
        assert len(series.train) < len(original.train)
        assert series.meta["drift_detail"]["train_ratio"] == pytest.approx(0.2, abs=0.05)


def test_unknown_drift_raises():
    with pytest.raises(ValueError):
        build_drift({"name": "definitely-not-a-protocol"})


def test_registries_are_populated():
    assert "ours" in list_models()
    assert "convae" in list_models()
    assert "iforest" in list_models()
    assert set(list_protocols()) >= {"none", "temporal", "channel", "channel_mask", "amplitude"}


def test_build_model_accepts_flat_and_nested_config():
    flat = build_model({"name": "zscore", "window": 3})
    nested = build_model({"name": "zscore", "params": {"window": 3}})
    assert flat.window == nested.window == 3


def test_config_deep_merge_and_override(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("seed: 1\ndata:\n  name: a\n  normalize: zscore\n", encoding="utf-8")
    patch = tmp_path / "patch.yaml"
    patch.write_text("data:\n  name: b\n", encoding="utf-8")

    cfg = load_config(str(base), str(patch), overrides=["drift.name=temporal", "eval.buffer_sizes=[0,5]"])
    assert cfg.seed == 1
    assert cfg.data.name == "b"
    assert cfg.data.normalize == "zscore"
    assert cfg.drift.name == "temporal"
    assert cfg.eval.buffer_sizes == [0, 5]

    set_by_path(cfg, "model.params.window", "16")
    assert cfg.model["params"]["window"] == 16


def test_config_missing_key_raises_attribute_error():
    cfg = Config({"a": 1})
    with pytest.raises(AttributeError):
        _ = cfg.nope


def test_window_helpers_cover_all_timesteps():
    x = np.arange(20, dtype=np.float32)[:, None]
    windows, starts = make_windows(x, window=5, stride=3)
    assert windows.shape[-2:] == (5, 1)
    scores = np.ones(len(windows))
    reduced = window_scores_to_timesteps(scores, starts, 5, 20)
    assert reduced.shape == (20,)
    assert np.allclose(reduced, 1.0)  # 没有时刻被漏掉


def test_window_helpers_handle_short_series():
    x = np.arange(3, dtype=np.float32)[:, None]
    windows, starts = make_windows(x, window=8, stride=1)
    assert len(windows) == 1
    assert starts.tolist() == [0]


def test_end_to_end_synthetic_run():
    cfg = Config(
        {
            "seed": 0,
            "data": {"name": "synthetic", "normalize": "zscore", "params": {"n_series": 2, "n_channels": 5, "n_train": 600, "n_test": 300, "seed": 0}},
            "drift": {"name": "none"},
            "model": {"name": "zscore", "params": {"window": 1}},
            "eval": {"metrics": ["auc_roc", "auc_pr", "best_f1", "vus_pr"], "buffer_sizes": [0, 5]},
        }
    )
    rows = run_experiment(cfg)
    assert len(rows) == 2
    for row in rows:
        assert row["model"] == "zscore"
        assert 0.0 <= row["auc_roc"] <= 1.0
        assert 0.0 <= row["auc_pr"] <= 1.0
        assert "vus_pr" in row


def test_end_to_end_with_drift_changes_scores():
    common = {
        "seed": 0,
        "data": {"name": "synthetic", "normalize": "zscore", "params": {"n_series": 1, "n_channels": 5, "n_train": 600, "n_test": 300, "seed": 0}},
        "model": {"name": "zscore", "params": {"window": 1}},
        "eval": {"metrics": ["auc_pr"], "buffer_sizes": [0]},
    }
    clean = run_experiment(Config({**common, "drift": {"name": "none"}}))
    drifted = run_experiment(Config({**common, "drift": {"name": "amplitude", "params": {"scale_std": 0.5}}}))
    assert clean[0]["auc_pr"] != drifted[0]["auc_pr"]


def test_tsb_ad_missing_root_raises(tmp_path):
    from src.datasets import build_dataset

    with pytest.raises(FileNotFoundError):
        build_dataset({"name": "tsb_ad", "root": str(tmp_path / "does-not-exist")})
    assert list_subsets(tmp_path) == []