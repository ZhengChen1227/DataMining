"""检测器注册表入口。

导入本包即完成全部检测器注册（通过各模块的装饰器副作用）。
"""

from __future__ import annotations

from . import classic, deep, ours  # noqa: F401  触发注册
from .base import REGISTRY, Detector, build_model, list_models, register_model

__all__ = [
    "REGISTRY",
    "Detector",
    "build_model",
    "list_models",
    "register_model",
]