"""数据集注册与构建入口。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Bundle, Series, describe_labels
from .synthetic import make_synthetic
from .transforms import Normalizer, normalize
from .tsb_ad import list_subsets, load_tsb_ad

__all__ = [
    "Bundle",
    "Series",
    "describe_labels",
    "make_synthetic",
    "Normalizer",
    "normalize",
    "list_subsets",
    "load_tsb_ad",
    "build_dataset",
    "build_pipeline",
]

_ALIASES = {
    "synthetic": "synthetic",
    "tsb_ad": "tsb_ad",
    "tsbad": "tsb_ad",
    "tsb-ad": "tsb_ad",
}


def build_dataset(cfg: Any) -> Bundle:
    """根据配置构造数据集。

    配置示例::

        data:
          name: tsb_ad
          root: data/raw/tsb-ad
          subsets: [SMAP, MSL]
          limit: 10
    """
    get = cfg.get if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    raw_name = get("name", "synthetic")
    key = _ALIASES.get(str(raw_name).lower())
    if key is None:
        raise ValueError(f"未知数据集: {raw_name}，可选 {sorted(set(_ALIASES.values()))}")

    if key == "synthetic":
        params: Dict[str, Any] = dict(get("params", {}) or {})
        params.setdefault("seed", get("seed", 0))
        return make_synthetic(**params)

    root = Path(get("root", "data/raw/tsb-ad"))
    return load_tsb_ad(
        root=root,
        subsets=get("subsets"),
        limit=get("limit"),
        min_anomaly_ratio=get("min_anomaly_ratio", 0.0),
    )


def build_pipeline(cfg: Any, seed: int = 42) -> Bundle:
    """完整数据管线：加载 -> 归一化 -> 施加漂移。

    顺序不可交换，原因见 `src/datasets/transforms.py` 顶部说明。
    """
    from ..drift import build_drift  # 局部导入，避免循环依赖

    get = cfg.get if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    data_cfg = get("data", {}) or {}
    drift_cfg = get("drift", {}) or {}

    bundle = build_dataset(data_cfg)
    mode = (data_cfg.get("normalize", "zscore") if isinstance(data_cfg, dict) else "zscore")
    bundle = normalize(bundle, mode=mode)
    bundle = build_drift(drift_cfg, seed=seed).apply(bundle)
    return bundle


def list_available(root: Optional[str] = None) -> List[str]:
    base = {"synthetic": []}
    if root:
        base["tsb_ad"] = list_subsets(root)
    return [f"{k} ({len(v)})" for k, v in base.items()]