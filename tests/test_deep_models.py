"""深度模型测试。没有 GPU / torch 时自动跳过。"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.datasets import make_synthetic  # noqa: E402
from src.models import build_model  # noqa: E402


@pytest.fixture(scope="module")
def small_bundle():
    return make_synthetic(n_series=1, n_channels=4, n_train=400, n_test=200, seed=0)


@pytest.mark.parametrize(
    "params",
    [
        {"window": 16, "stride": 4, "epochs": 1},
        {"window": 16, "stride": 4, "epochs": 1, "topk": 2, "n_env": 2, "env_penalty": 0.1},
    ],
)
def test_convae_runs_and_produces_timestep_scores(small_bundle, params):
    series = small_bundle[0]
    detector = build_model({"name": "convae", "params": params})
    scores = detector.fit_score(series.train, series.test)
    assert scores.shape == series.test.shape[:1]
    assert np.all(np.isfinite(scores))


@pytest.mark.parametrize(
    "params",
    [
        {"window": 16, "stride": 4, "epochs": 1},
        {"window": 16, "stride": 4, "epochs": 1, "topk": 2, "n_env": 2, "env_penalty": 0.1},
        {"window": 16, "stride": 4, "epochs": 1, "topk": 2, "tta": True, "tta_steps": 2},
        {"window": 16, "stride": 4, "epochs": 1, "topk": 2, "tta": True, "tta_finetune": True, "tta_steps": 2},
    ],
)
def test_ours_runs_in_all_variants(small_bundle, params):
    series = small_bundle[0]
    detector = build_model({"name": "ours", "params": params})
    scores = detector.fit_score(series.train, series.test)
    assert scores.shape == series.test.shape[:1]
    assert np.all(np.isfinite(scores))


def test_ours_exposes_learned_adjacency(small_bundle):
    series = small_bundle[0]
    detector = build_model(
        {"name": "ours", "params": {"window": 16, "stride": 4, "epochs": 1, "topk": 2}}
    )
    detector.fit(series.train)
    adjacency = detector.adjacency()
    assert adjacency.shape == (4, 4)
    assert np.allclose(adjacency.sum(axis=1), 1.0, atol=1e-5)
    # top-k 稀疏化：每行非零元素不超过 topk 个
    assert (adjacency > 0).sum(axis=1).max() <= 2


def test_external_adapter_contract():
    series = make_synthetic(n_series=1, n_channels=3, n_train=200, n_test=100, seed=0)[0]
    detector = build_model(
        {"name": "external", "import_path": "adapters.example", "entrypoint": "fit_score",
         "params": {"window": 8, "stride": 2}}
    )
    scores = detector.fit_score(series.train, series.test)
    assert scores.shape == series.test.shape[:1]


def test_external_adapter_rejects_wrong_length():
    series = make_synthetic(n_series=1, n_channels=3, n_train=200, n_test=100, seed=0)[0]
    detector = build_model(
        {"name": "external", "import_path": "adapters", "entrypoint": "bad_length"}
    )
    with pytest.raises(ValueError):
        detector.fit_score(series.train, series.test)