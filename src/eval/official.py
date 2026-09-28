"""TSB-AD 官方评测器适配。

为什么必须有这个文件
--------------------
时序异常检测的指标实现差异极大（buffer 定义、range 归一化、
区间长度的加权方式），不同论文的 VUS-PR 常常不可直接比较。
TSB-AD（NeurIPS 2024 D&B）给出的评测器是目前最被认可的基准实现，
投稿必须用它。

现状与待办
----------
本文件先提供两条路径：

1. `official_metrics(...)`：在你的环境里能找到官方评测器时直接调用；
2. 找不到时抛出带安装指引的异常，**绝不静默回退到近似实现**，
   避免把内部数字误当成论文数字。

接入步骤：
    git clone https://github.com/TheDatumOrg/TSB-AD
    pip install -e TSB-AD            # 或按官方 README 安装依赖
然后把 `TSB_AD_IMPORT` / `TSB_AD_ENTRY` 调整成官方实际暴露的符号名。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

import numpy as np

#: 官方评测器所在的模块路径，按需修改
TSB_AD_IMPORT = os.environ.get("TSBAD_IMPORT", "tsb_ad.metrics")
#: 官方暴露的评测入口名，按需修改
TSB_AD_ENTRY = os.environ.get("TSBAD_ENTRY", "get_metrics")

INSTALL_HINT = (
    "未找到 TSB-AD 官方评测器。请先安装：\n"
    "    git clone https://github.com/TheDatumOrg/TSB-AD\n"
    "    pip install -e TSB-AD\n"
    "然后确认 src/eval/official.py 里的 TSB_AD_IMPORT / TSB_AD_ENTRY 与官方一致。"
)


class OfficialEvaluatorUnavailable(RuntimeError):
    """找不到官方评测器时抛出，避免误用近似指标。"""


def is_available() -> bool:
    try:
        _resolve()
    except OfficialEvaluatorUnavailable:
        return False
    return True


def _resolve():
    import importlib

    try:
        module = importlib.import_module(TSB_AD_IMPORT)
    except ImportError as exc:
        raise OfficialEvaluatorUnavailable(INSTALL_HINT) from exc
    entry = getattr(module, TSB_AD_ENTRY, None)
    if entry is None:
        raise OfficialEvaluatorUnavailable(
            f"{TSB_AD_IMPORT} 中找不到入口 {TSB_AD_ENTRY!r}。" + INSTALL_HINT
        )
    return entry


def official_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    metrics: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> Dict[str, float]:
    """调用官方评测器计算指标。接口名以官方仓库为准，必要时在此处做形状适配。"""
    entry = _resolve()
    raw = entry(
        np.asarray(labels),
        np.asarray(scores),
        list(metrics) if metrics else None,
        **kwargs,
    )
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items() if np.isscalar(v)}
    raise TypeError(f"官方评测器返回了非字典类型: {type(raw)}")