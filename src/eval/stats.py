"""显著性检验：判断两个方法之间的提升是否只是噪声。

时序异常检测的数据集间方差极大，「平均提升 0.02」经常不显著。
报告 mean ± std 之外，配对检验是低成本高可信度的加分项。
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """配对 Wilcoxon 符号秩检验。a 为我们的方法，b 为基线。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 5:
        return {"n": int(len(a)), "statistic": float("nan"), "p_value": float("nan"), "note": "样本太少"}

    try:
        from scipy.stats import wilcoxon
    except ImportError:  # pragma: no cover
        return {"n": int(len(a)), "statistic": float("nan"), "p_value": float("nan"), "note": "缺少 scipy"}

    diff = a - b
    if np.allclose(diff, 0.0):
        return {"n": int(len(a)), "statistic": 0.0, "p_value": 1.0, "note": "无差异"}
    result = wilcoxon(a, b)
    return {
        "n": int(len(a)),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "mean_delta": float(diff.mean()),
        "wins": int((diff > 0).sum()),
        "losses": int((diff < 0).sum()),
        "note": "p < 0.05 视为显著" if result.pvalue < 0.05 else "不显著",
    }


def paired_by_series(
    rows: Sequence[Dict],
    metric: str,
    ours: str,
    baseline: str,
    series_key: str = "series",
) -> Dict[str, float]:
    """按序列名配对后做检验。"""
    ours_map: Dict[str, float] = {}
    base_map: Dict[str, float] = {}
    for row in rows:
        name = str(row.get(series_key))
        if str(row.get("model")) == ours:
            ours_map[name] = row.get(metric, float("nan"))
        elif str(row.get("model")) == baseline:
            base_map[name] = row.get(metric, float("nan"))

    shared: List[str] = sorted(set(ours_map) & set(base_map))
    return paired_wilcoxon([ours_map[k] for k in shared], [base_map[k] for k in shared])