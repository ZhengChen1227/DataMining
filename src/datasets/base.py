"""数据容器：把所有数据源统一成同一种表示。

约定：
- `train` / `test` 形状为 (T, C) 的 float32 数组，C 为通道数；
- `*_label` 形状为 (T,) 的 0/1 int8 数组，1 表示该时刻属于异常区间；
- 训练集标签只在「有监督 / 半监督」的消融实验里使用，主实验为无监督设定。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

import numpy as np


@dataclass
class Series:
    """单条多变量序列。"""

    name: str
    train: np.ndarray
    train_label: np.ndarray
    test: np.ndarray
    test_label: np.ndarray
    meta: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.train = np.asarray(self.train, dtype=np.float32)
        self.test = np.asarray(self.test, dtype=np.float32)
        if self.train.ndim == 1:
            self.train = self.train[:, None]
        if self.test.ndim == 1:
            self.test = self.test[:, None]
        self.train_label = np.asarray(self.train_label, dtype=np.int8).ravel()
        self.test_label = np.asarray(self.test_label, dtype=np.int8).ravel()

        if self.train.shape[1] != self.test.shape[1]:
            raise ValueError(
                f"{self.name}: 训练与测试通道数不一致 "
                f"({self.train.shape[1]} vs {self.test.shape[1]})"
            )
        if len(self.train_label) != len(self.train) or len(self.test_label) != len(self.test):
            raise ValueError(f"{self.name}: 标签长度与序列长度不一致")

    @property
    def n_channels(self) -> int:
        return int(self.train.shape[1])

    @property
    def anomaly_ratio(self) -> float:
        return float(self.test_label.mean()) if len(self.test_label) else 0.0

    def copy(self, **overrides) -> "Series":
        payload = dict(
            name=self.name,
            train=self.train.copy(),
            train_label=self.train_label.copy(),
            test=self.test.copy(),
            test_label=self.test_label.copy(),
            meta=dict(self.meta),
        )
        payload.update(overrides)
        return Series(**payload)

    def __repr__(self) -> str:  # pragma: no cover - 便于调试
        return (
            f"Series(name={self.name!r}, C={self.n_channels}, "
            f"T_train={len(self.train)}, T_test={len(self.test)}, "
            f"anomaly_ratio={self.anomaly_ratio:.4f})"
        )


@dataclass
class Bundle:
    """一个数据集，包含一条或多条 Series。"""

    name: str
    series: List[Series]
    meta: Dict = field(default_factory=dict)

    def __iter__(self) -> Iterator[Series]:
        return iter(self.series)

    def __len__(self) -> int:
        return len(self.series)

    def __getitem__(self, idx: int) -> Series:
        return self.series[idx]

    def copy(self, **overrides) -> "Bundle":
        payload = dict(
            name=self.name,
            series=[s.copy() for s in self.series],
            meta=dict(self.meta),
        )
        payload.update(overrides)
        return Bundle(**payload)

    def summary(self) -> str:
        lines = [f"Bundle({self.name}) 共 {len(self)} 条序列"]
        for s in self.series:
            lines.append(f"  - {s!r}")
        return "\n".join(lines)


def describe_labels(labels: np.ndarray) -> Dict[str, float]:
    """统计标签里异常区间的个数与长度分布，用于检查数据划分是否合理。"""
    labels = np.asarray(labels).ravel()
    idx = np.flatnonzero(labels)
    if idx.size == 0:
        return {"n_ranges": 0, "mean_length": 0.0, "max_length": 0.0, "ratio": 0.0}
    splits = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[splits + 1]))
    ends = np.concatenate((idx[splits], [idx[-1]]))
    lengths = ends - starts + 1
    return {
        "n_ranges": int(len(lengths)),
        "mean_length": float(lengths.mean()),
        "max_length": float(lengths.max()),
        "ratio": float(labels.mean()),
    }