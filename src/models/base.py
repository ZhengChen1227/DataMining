"""检测器统一接口。

约定
----
- 主实验为**无监督**设定：`fit` 只吃训练段的观测值，`train_label` 一律忽略，
  但接口保留它，便于做半监督消融。
- `score(x)` 返回逐时刻的异常分数，**越大越异常**，形状 (T,)。
  所有基线必须遵守这个方向约定，否则结果的符号会出错。
- 检测器不得访问测试段的标签，也不得读取 `series.meta["drift"]`，
  那是评测方的信息。跑实验时由评测流程保证这一点。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

import numpy as np

REGISTRY: Dict[str, Type["Detector"]] = {}


def register_model(cls: Type["Detector"]) -> Type["Detector"]:
    if cls.name in REGISTRY and REGISTRY[cls.name] is not cls:
        raise ValueError(f"检测器名称重复: {cls.name}")
    REGISTRY[cls.name] = cls
    return cls


class Detector(ABC):
    name: str = "base"
    #: 是否需要 fit（纯在线方法可以置 False）
    requires_fit: bool = True
    #: 是否为需要 torch 的深度模型
    deep: bool = False

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    def fit(self, train: np.ndarray, train_label: Optional[np.ndarray] = None) -> "Detector":
        """在训练段上拟合。返回 self 便于链式调用。"""

    @abstractmethod
    def score(self, x: np.ndarray) -> np.ndarray:
        """返回 (T,) 的异常分数，越大越异常。"""

    def fit_score(
        self,
        train: np.ndarray,
        test: np.ndarray,
        train_label: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self.fit(train, train_label).score(test)

    def describe(self) -> Dict[str, Any]:
        return {"model": self.name, **self.params}

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}({self.params})"


def build_model(cfg: Any) -> Detector:
    """从配置构造检测器。

    支持 `model: {name: iforest, params: {...}}` 与扁平写法。
    """
    if isinstance(cfg, str):
        name, params = cfg, {}
    elif isinstance(cfg, dict):
        name = cfg.get("name")
        params = dict(cfg.get("params", {}) or {})
        for key, value in cfg.items():
            if key not in ("name", "params"):
                params.setdefault(key, value)
    else:  # pragma: no cover
        raise TypeError(f"无法从 {type(cfg)} 构造检测器")

    if not name:
        raise ValueError("配置缺少 model.name")
    cls = REGISTRY.get(str(name).lower())
    if cls is None:
        raise ValueError(f"未知检测器: {name!r}。已注册: {sorted(REGISTRY)}")
    return cls(**params)


def list_models() -> Dict[str, str]:
    return {
        name: (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else ""
        for name, cls in sorted(REGISTRY.items())
    }