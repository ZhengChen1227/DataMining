"""合成数据：在下载真实数据之前跑通全流程。

设计成「已知 ground truth 的异常注入」，好处是可以在没有外部依赖、
没有 GPU 的环境里验证：数据管线、漂移协议、指标、评测流程是否正确。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Bundle, Series


def _generate(
    rng: np.random.Generator,
    n: int,
    n_channels: int,
    period: float,
    base: np.ndarray,
) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    signal = np.sin(2.0 * np.pi * t / period)[:, None] * base[None, :]
    signal += 0.3 * np.cos(2.0 * np.pi * t / (period * 3.7))[:, None]
    noise = rng.normal(scale=0.1, size=(n, n_channels))
    return (signal + noise).astype(np.float32)


def make_synthetic(
    n_series: int = 3,
    n_channels: int = 8,
    n_train: int = 4000,
    n_test: int = 2000,
    n_anomaly_ranges: int = 8,
    min_length: int = 30,
    max_length: int = 80,
    period: float = 50.0,
    seed: int = 0,
    name: str = "synthetic",
) -> Bundle:
    """生成带有已知异常区间的多变量序列。"""
    rng = np.random.default_rng(seed)
    series = []

    for s in range(n_series):
        base = rng.normal(loc=1.0, scale=0.5, size=n_channels)
        train = _generate(rng, n_train, n_channels, period, base)
        test = _generate(rng, n_test, n_channels, period, base)

        label = np.zeros(n_test, dtype=np.int8)
        placed = 0
        attempts = 0
        while placed < n_anomaly_ranges and attempts < n_anomaly_ranges * 20:
            attempts += 1
            length = int(rng.integers(min_length, max_length + 1))
            start = int(rng.integers(0, max(1, n_test - length)))
            if label[start : start + length].any():
                continue
            # 异常 = 通道子集上的均值漂移 + 方差放大，避免过于容易被检测
            channels = rng.choice(n_channels, size=max(1, n_channels // 3), replace=False)
            shift = rng.normal(scale=3.0, size=len(channels))
            test[start : start + length, channels] += shift[None, :]
            test[start : start + length, channels] *= 1.8
            label[start : start + length] = 1
            placed += 1

        series.append(
            Series(
                name=f"{name}_{s}",
                train=train,
                train_label=np.zeros(n_train, dtype=np.int8),
                test=test,
                test_label=label,
                meta={"source": "synthetic", "seed": seed, "period": period},
            )
        )

    return Bundle(name=name, series=series, meta={"source": "synthetic", "seed": seed})


def load_synthetic(cfg: Optional[dict] = None) -> Bundle:
    cfg = dict(cfg or {})
    return make_synthetic(**cfg)