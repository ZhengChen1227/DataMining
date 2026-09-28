"""漂移协议基类与注册表。

设计原则
--------
1. **只动训练段或只动测试段，不同时动**。同时动会无法归因。
2. **不改变维度、不改变标签**。测试段的异常标签必须原样保留，
   否则指标不可比。
3. **确定性**。同样的 seed + 同样的数据必须得到同样的漂移结果。
4. **可描述**。每个协议都要能被 `describe()` 写进结果表，
   因为论文里的漂移划分本身就是要交代清楚的方法论细节。

所有协议都把自身描述写进 `series.meta["drift"]`，便于结果聚合时分组。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Type

import numpy as np

from ..datasets import Bundle

REGISTRY: Dict[str, Type["DriftProtocol"]] = {}


def register(cls: Type["DriftProtocol"]) -> Type["DriftProtocol"]:
    if cls.name in REGISTRY and REGISTRY[cls.name] is not cls:
        raise ValueError(f"漂移协议名称重复: {cls.name}")
    REGISTRY[cls.name] = cls
    return cls


class DriftProtocol(ABC):
    """把所有漂移协议统一到一个 `apply(bundle) -> bundle` 接口。"""

    name: str = "base"
    #: 供文档与结果表使用的自然语言说明
    description: str = ""

    def __init__(self, seed: int = 0, **params: Any) -> None:
        self.seed = int(seed)
        self.params = params

    @abstractmethod
    def apply(self, bundle: Bundle) -> Bundle:
        """返回施加漂移后的新 Bundle（不修改输入）。"""

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "seed": self.seed, **self.params}

    def _tag(self, bundle: Bundle) -> Bundle:
        info = self.describe()
        info["description"] = self.description
        for s in bundle.series:
            s.meta["drift"] = dict(info)
        bundle.meta["drift"] = dict(info)
        return bundle

    def __repr__(self) -> str:  # pragma: no cover - 便于调试
        return f"{type(self).__name__}({self.describe()})"


@register
class NoDrift(DriftProtocol):
    """同分布对照，论文里必须有这一行。"""

    name = "none"
    description = "不做任何漂移，训练段与测试段同分布。"

    def apply(self, bundle: Bundle) -> Bundle:
        return self._tag(bundle.copy())


def build_drift(cfg: Any, seed: int = 42) -> DriftProtocol:
    """从配置构造漂移协议。

    支持两种写法::

        drift: {name: temporal, params: {n_segments: 5}}
        drift: {name: temporal, n_segments: 5}     # 扁平写法同样可用
    """
    if cfg is None:
        return NoDrift(seed=seed)
    if isinstance(cfg, str):
        name, params = cfg, {}
    elif isinstance(cfg, dict):
        name = cfg.get("name", "none") or "none"
        params = dict(cfg.get("params", {}) or {})
        for key, value in cfg.items():
            if key not in ("name", "params"):
                params.setdefault(key, value)
    else:  # pragma: no cover - 防御性分支
        raise TypeError(f"无法从 {type(cfg)} 构造漂移协议")

    cls = REGISTRY.get(str(name).lower())
    if cls is None:
        raise ValueError(
            f"未知漂移协议: {name!r}。已注册: {sorted(REGISTRY)}"
        )
    params.pop("seed", None)
    return cls(seed=seed, **params)


def list_protocols() -> Dict[str, str]:
    return {name: cls.description for name, cls in REGISTRY.items()}


def distribution_gap(a: np.ndarray, b: np.ndarray) -> float:
    """两个数组之间的逐通道标准化均值差异（选切分点用的廉价统计量）。

    这是 MMD 的粗糙替代：只比较一阶矩，计算量 O(T*C)，够用且不引入依赖。
    """
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    sd = np.sqrt((a.var(axis=0) + b.var(axis=0)) / 2.0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return float(np.mean(np.abs(mu_a - mu_b) / sd))