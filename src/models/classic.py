"""经典（非深度）基线。

包含两类：

1. 不依赖额外库的轻量基线：`ZScoreDetector`、`AdaptiveZScoreDetector`。
   `AdaptiveZScoreDetector` 是审稿人一定会问的对照——它在测试流上重算
   归一化统计量，把「漂移」问题降级成「没做在线归一化」问题。
   如果我们的方法打不过它，说明方法没有价值，必须如实报告。
2. 基于 pyod 的基线：IForest / OCSVM / LOF / KNN / ECOD / COPOD。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Detector, register_model
from .window import make_windows, window_scores_to_timesteps

EPS = 1e-8


class _WindowedDetector(Detector):
    """把逐时刻特征变成滑窗特征的公共逻辑。"""

    def __init__(self, window: int = 1, stride: int = 1, eps: float = EPS, **params):
        super().__init__(window=window, stride=stride, eps=eps, **params)
        self.window = int(window)
        self.stride = int(stride)
        self.eps = float(eps)
        self.mu_: Optional[np.ndarray] = None
        self.sd_: Optional[np.ndarray] = None

    def _fit_stats(self, train: np.ndarray) -> None:
        self.mu_ = train.mean(axis=0)
        sd = train.std(axis=0)
        self.sd_ = np.where(sd < self.eps, 1.0, sd)

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mu_[None, :]) / self.sd_[None, :]).astype(np.float32)

    def _features(self, x: np.ndarray) -> np.ndarray:
        z = self._standardize(x)
        windows, _ = make_windows(z, self.window, self.stride)
        return windows.reshape(len(windows), -1)

    def _reduce(self, scores: np.ndarray, x: np.ndarray) -> np.ndarray:
        if self.window <= 1:
            return np.asarray(scores, dtype=np.float32).ravel()[: len(x)]
        _, starts = make_windows(x, self.window, self.stride)
        return window_scores_to_timesteps(scores, starts, self.window, len(x))


@register_model
class ZScoreDetector(_WindowedDetector):
    """逐时刻 z-score 的 L2 范数。"""

    name = "zscore"

    def fit(self, train: np.ndarray, train_label: Optional[np.ndarray] = None) -> "ZScoreDetector":
        self._fit_stats(np.asarray(train, dtype=np.float32))
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        features = self._features(x)
        return self._reduce(np.linalg.norm(features, axis=1), x)


@register_model
class AdaptiveZScoreDetector(_WindowedDetector):
    """在线重算归一化统计量的 z-score。

    测试时用**测试流自身**的均值和标准差做标准化。这是对抗漂移的
    最低成本手段，也是本项目最重要的「必须打败的基线」。
    """

    name = "adaptive_zscore"

    def fit(self, train: np.ndarray, train_label: Optional[np.ndarray] = None) -> "AdaptiveZScoreDetector":
        # 仍然记录训练统计量，供消融对比「用训练统计量」的结果
        self._fit_stats(np.asarray(train, dtype=np.float32))
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        mu = x.mean(axis=0)
        sd = x.std(axis=0)
        sd = np.where(sd < self.eps, 1.0, sd)
        z = (x - mu[None, :]) / sd[None, :]
        windows, _ = make_windows(z.astype(np.float32), self.window, self.stride)
        features = windows.reshape(len(windows), -1)
        return self._reduce(np.linalg.norm(features, axis=1), x)


class _PyODDetector(_WindowedDetector):
    """pyod 基线的公共外壳。"""

    pyod_cls_name: str = ""

    def __init__(self, window: int = 1, stride: int = 1, **params):
        super().__init__(window=window, stride=stride, **params)
        self.pyod_params = dict(params)

    def _make(self):
        try:
            import pyod.models  # noqa: F401
        except ImportError as exc:  # pragma: no cover - 环境相关
            raise ImportError(
                f"检测器 {self.name} 需要 pyod，请执行 `pip install pyod`。"
            ) from exc
        module = __import__("pyod.models", fromlist=["__dict__"])
        mapping = {
            "IForest": "iforest",
            "OCSVM": "ocsvm",
            "LOF": "lof",
            "KNN": "knn",
            "ECOD": "ecod",
            "COPOD": "copod",
            "MCD": "mcd",
        }
        submodule = __import__(f"pyod.models.{mapping[self.pyod_cls_name]}", fromlist=["__dict__"])
        return getattr(submodule, self.pyod_cls_name)(**self.pyod_params)

    def fit(self, train: np.ndarray, train_label: Optional[np.ndarray] = None) -> "_PyODDetector":
        train = np.asarray(train, dtype=np.float32)
        self._fit_stats(train)
        self.model_ = self._make()
        self.model_.fit(self._features(train))
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        raw = self.model_.decision_function(self._features(x))
        return self._reduce(np.asarray(raw, dtype=np.float64), x)


@register_model
class IForestDetector(_PyODDetector):
    """隔离森林（pyod）。"""

    name = "iforest"
    pyod_cls_name = "IForest"


@register_model
class OCSVMDetector(_PyODDetector):
    """单类 SVM（pyod）。"""

    name = "ocsvm"
    pyod_cls_name = "OCSVM"


@register_model
class LOFDetector(_PyODDetector):
    """局部离群因子（pyod）。"""

    name = "lof"
    pyod_cls_name = "LOF"


@register_model
class KNNDetector(_PyODDetector):
    """k 近邻距离（pyod）。"""

    name = "knn"
    pyod_cls_name = "KNN"


@register_model
class ECODDetector(_PyODDetector):
    """ECOD 经验累积分布离群检测（pyod）。"""

    name = "ecod"
    pyod_cls_name = "ECOD"


@register_model
class COPODDetector(_PyODDetector):
    """COPOD 尾概率离群检测（pyod）。"""

    name = "copod"
    pyod_cls_name = "COPOD"