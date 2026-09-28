"""时间维漂移：训练段与测试段来自不同工况。

做法：把原始训练段按时间切成 K 个连续 chunk，只保留与测试段
统计差异最大的若干 chunk 作为训练集，其余丢弃。

- 测试段与标签**完全不动**，因此指标与其它协议严格可比；
- 训练集变小、且明显偏离测试分布，模拟「上线后工况切换」；
- `keep=1` 时训练集只有原训练段的 1/K，可以在结果表里顺带报告
  「数据量减少」这条混淆因素（见 docs/drift_protocol.md 的消融要求）。
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..datasets import Bundle, Series
from .protocol import DriftProtocol, distribution_gap, register


@register
class TemporalShift(DriftProtocol):
    name = "temporal"
    description = (
        "把训练段切成 K 个连续 chunk，只保留与测试段分布差异最大的 chunk 作为训练集，"
        "模拟工况切换（测试段与标签不变）。"
    )

    def __init__(self, n_segments: int = 5, keep: int = 1, seed: int = 0, **params) -> None:
        super().__init__(seed=seed, **params)
        if n_segments < 2:
            raise ValueError("n_segments 至少为 2")
        if not 1 <= keep < n_segments:
            raise ValueError("keep 必须满足 1 <= keep < n_segments")
        self.n_segments = int(n_segments)
        self.keep = int(keep)

    def _split(self, train: np.ndarray) -> List[np.ndarray]:
        bounds = np.linspace(0, len(train), self.n_segments + 1).astype(int)
        return [train[bounds[i] : bounds[i + 1]] for i in range(self.n_segments)]

    def apply(self, bundle: Bundle) -> Bundle:
        out = bundle.copy()
        for s in out.series:
            chunks = self._split(s.train)
            bounds = np.linspace(0, len(s.train), self.n_segments + 1).astype(int)
            gaps = [distribution_gap(chunk, s.test) for chunk in chunks]

            order = np.argsort(gaps)[::-1]
            chosen = sorted(int(i) for i in order[: self.keep])

            train = np.concatenate([chunks[i] for i in chosen], axis=0)
            label = np.concatenate(
                [s.train_label[bounds[i] : bounds[i + 1]] for i in chosen], axis=0
            )

            s.meta["drift_detail"] = {
                "chosen_segments": chosen,
                "segment_gaps": [round(float(g), 4) for g in gaps],
                "train_ratio": round(float(len(train) / max(1, len(s.train))), 4),
            }
            s.train = train
            s.train_label = label
        return self._tag(out)