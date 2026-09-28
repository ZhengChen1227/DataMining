"""归一化等前处理。

关键约定：**归一化只用训练段统计量，且必须在施加漂移之前执行。**
如果反过来（先加漂移再归一化），全局尺度漂移会被 z-score 吸收掉，
漂移实验就失效了。这条在 docs/drift_protocol.md 里有详细说明。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .base import Bundle


class Normalizer:
    """逐通道标准化器。"""

    MODES = ("zscore", "robust", "none")

    def __init__(self, mode: str = "zscore", eps: float = 1e-8, clip: Optional[float] = None):
        if mode not in self.MODES:
            raise ValueError(f"未知归一化模式: {mode}，可选 {self.MODES}")
        self.mode = mode
        self.eps = eps
        self.clip = clip
        self.stats_: Dict[str, np.ndarray] = {}

    def _fit_one(self, train: np.ndarray) -> Dict[str, np.ndarray]:
        if self.mode == "zscore":
            center = train.mean(axis=0)
            scale = train.std(axis=0)
        elif self.mode == "robust":
            center = np.median(train, axis=0)
            q75, q25 = np.percentile(train, [75, 25], axis=0)
            scale = (q75 - q25) / 1.349
        else:
            center = np.zeros(train.shape[1], dtype=np.float64)
            scale = np.ones(train.shape[1], dtype=np.float64)
        scale = np.where(np.abs(scale) < self.eps, 1.0, scale)
        return {"center": center.astype(np.float32), "scale": scale.astype(np.float32)}

    def _apply(self, x: np.ndarray, stat: Dict[str, np.ndarray]) -> np.ndarray:
        z = (x - stat["center"][None, :]) / stat["scale"][None, :]
        if self.clip is not None:
            z = np.clip(z, -self.clip, self.clip)
        return z.astype(np.float32)

    def fit(self, bundle: Bundle) -> "Normalizer":
        self.stats_ = {s.name: self._fit_one(s.train) for s in bundle.series}
        return self

    def transform(self, bundle: Bundle) -> Bundle:
        out = bundle.copy()
        for s in out.series:
            stat = self.stats_.get(s.name)
            if stat is None:
                stat = self._fit_one(s.train)
                self.stats_[s.name] = stat
            s.meta["normalizer"] = self.mode
            s.train = self._apply(s.train, stat)
            s.test = self._apply(s.test, stat)
        return out


def normalize(bundle: Bundle, mode: str = "zscore", **kwargs) -> Bundle:
    if mode in (None, "none"):
        return bundle
    return Normalizer(mode=mode, **kwargs).fit(bundle).transform(bundle)