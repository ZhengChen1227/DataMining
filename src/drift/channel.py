"""通道维漂移：破坏「通道间关系」或「通道可用集合」。

两个协议刻意分开，因为它们的诊断意义完全不同：

- `channel`（通道置换）：**不改变任何单通道的边缘分布**，
  只打乱通道之间的对应关系。因此逐点方法（IForest 等）几乎不受影响，
  而显式建模通道依赖的方法会受到冲击。这是支撑「通道关系建模过强会反噬」
  这一论点最干净的对照实验。
- `channel_mask`：训练段和测试段都随机屏蔽相同比例的通道，
  但屏蔽的通道集合不同（模拟传感器更换 / 重新布线）。
  零值比例保持一致，所以不会产生「零值即异常」的平凡解。
"""

from __future__ import annotations

import numpy as np

from ..datasets import Bundle
from .protocol import DriftProtocol, register


@register
class ChannelPermutation(DriftProtocol):
    name = "channel"
    description = (
        "对测试段随机置换通道顺序，保持每通道边缘分布不变，只破坏通道间关系。"
    )

    def __init__(self, seed: int = 0, per_series: bool = True, **params) -> None:
        super().__init__(seed=seed, **params)
        self.per_series = bool(per_series)

    def apply(self, bundle: Bundle) -> Bundle:
        out = bundle.copy()
        shared: np.ndarray | None = None
        for i, s in enumerate(out.series):
            n_channels = s.n_channels
            if shared is None and not self.per_series:
                rng = np.random.default_rng(self.seed)
                shared = rng.permutation(n_channels)
            elif shared is None:
                shared = None  # 逐序列独立生成
            if self.per_series:
                rng = np.random.default_rng(self.seed + i)
                perm = rng.permutation(n_channels)
            else:
                if len(shared) != n_channels:
                    raise ValueError("per_series=False 时要求所有序列通道数一致")
                perm = shared

            s.meta["drift_detail"] = {"permutation": [int(v) for v in perm]}
            s.test = np.ascontiguousarray(s.test[:, perm])
        return self._tag(out)


@register
class ChannelMask(DriftProtocol):
    name = "channel_mask"
    description = (
        "训练段与测试段各屏蔽相同比例的通道（置 0），但屏蔽集合不同，"
        "模拟传感器更换 / 重新布线。"
    )

    def __init__(self, ratio: float = 0.2, seed: int = 0, **params) -> None:
        super().__init__(seed=seed, **params)
        if not 0.0 < ratio < 1.0:
            raise ValueError("ratio 必须落在 (0, 1) 内")
        self.ratio = float(ratio)

    def apply(self, bundle: Bundle) -> Bundle:
        out = bundle.copy()
        for i, s in enumerate(out.series):
            rng = np.random.default_rng(self.seed + 1000 + i)
            n_channels = s.n_channels
            n_mask = max(1, int(round(n_channels * self.ratio)))
            if n_mask >= n_channels:
                raise ValueError(f"{s.name}: 通道数 {n_channels} 太少，无法屏蔽 {n_mask} 个")

            train_mask = rng.choice(n_channels, size=n_mask, replace=False)
            remaining = np.setdiff1d(np.arange(n_channels), train_mask)
            test_mask = rng.choice(remaining, size=n_mask, replace=False)

            s.meta["drift_detail"] = {
                "train_masked": [int(v) for v in train_mask],
                "test_masked": [int(v) for v in test_mask],
            }
            s.train = s.train.copy()
            s.train[:, train_mask] = 0.0
            s.test = s.test.copy()
            s.test[:, test_mask] = 0.0
        return self._tag(out)