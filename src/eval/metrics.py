"""评测指标。

必读警告
--------
本文件里的 `vus_pr` / `range_*` 是**开发期用的近似实现**，用于快速比较方法、
画学习曲线。TSB-AD 官方评测器对 VUS 的定义还额外包含 range-recall 维度，
数值会有差异。

**投稿论文里的数字必须来自官方评测器**（见 `src/eval/official.py`），
本文件的结果只能在 README 里标注为「内部比较」。

指标选择的立场
--------------
- 主指标：`vus_pr`、`auc_pr`。原因：异常点占比通常 <5%，AUC-ROC 会严重高估性能。
- `best_f1`：方便和早期文献对齐，但它是「最优阈值」下的结果，属于上界，慎用。
- `point_adjust_f1`：**只作为对照列**。它会把「区间内检出一个点」当作整个区间
  检出，对长区间极其宽容，是被 TSB-AD 明确批评的指标。报告它并说明其偏乐观，
  反而是展示你懂行的地方。
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np

DEFAULT_METRICS: Sequence[str] = ("auc_roc", "auc_pr", "best_f1", "vus_pr")
DEFAULT_BUFFERS: Sequence[int] = (0, 1, 5, 10, 50, 100)

__all__ = [
    "DEFAULT_METRICS",
    "DEFAULT_BUFFERS",
    "auc_roc",
    "auc_pr",
    "best_f1",
    "point_adjust_f1",
    "range_auc_pr",
    "vus_pr",
    "compute_metrics",
    "aggregate_metrics",
    "anomaly_ranges",
]


def _check(labels: np.ndarray, scores: np.ndarray):
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if len(labels) != len(scores):
        raise ValueError(f"标签长度 {len(labels)} 与分数长度 {len(scores)} 不一致")
    return (labels > 0).astype(np.int8), scores


def _rankdata(values: np.ndarray) -> np.ndarray:
    """平均秩（处理并列），等价于 scipy.stats.rankdata 的默认行为。"""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)

    sorted_values = values[order]
    start = 0
    while start < len(sorted_values):
        end = start
        while end + 1 < len(sorted_values) and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        if end > start:
            ranks[order[start : end + 1]] = (start + end + 2) / 2.0
        start = end + 1
    return ranks


def auc_roc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC 曲线下面积（Mann-Whitney U 统计量，自动处理并列）。"""
    labels, scores = _check(labels, scores)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(scores)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auc_pr(labels: np.ndarray, scores: np.ndarray) -> float:
    """平均精度 (Average Precision)，即 PR 曲线下面积。"""
    labels, scores = _check(labels, scores)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order].astype(np.float64)
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    precision = tp / np.maximum(tp + fp, 1e-12)
    return float(precision[y == 1].sum() / n_pos)


def best_f1(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """扫描所有阈值取最大 F1。

    判定约定：`score >= threshold` 为正类。返回的 threshold 是第 (k+1) 大的分数。
    """
    labels, scores = _check(labels, scores)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return {"f1": float("nan"), "precision": float("nan"), "recall": float("nan"), "threshold": float("nan")}

    order = np.argsort(-scores, kind="mergesort")
    y = labels[order].astype(np.float64)
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    fn = n_pos - tp
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / n_pos
    denom = precision + recall
    f1 = np.where(denom > 0, 2.0 * precision * recall / np.maximum(denom, 1e-12), 0.0)

    k = int(np.argmax(f1))
    return {
        "f1": float(f1[k]),
        "precision": float(precision[k]),
        "recall": float(recall[k]),
        "threshold": float(scores[order][k]),
        "_tp": float(tp[k]),
        "_fp": float(fp[k]),
        "_fn": float(fn[k]),
    }


def anomaly_ranges(labels: np.ndarray):
    """返回所有异常区间的 (start, end_inclusive) 列表。"""
    labels = (np.asarray(labels).ravel() > 0).astype(np.int8)
    idx = np.flatnonzero(labels)
    if idx.size == 0:
        return []
    splits = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[splits + 1]))
    ends = np.concatenate((idx[splits], [idx[-1]]))
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def point_adjust_f1(labels: np.ndarray, scores: np.ndarray, threshold: float | None = None) -> Dict[str, float]:
    """point-adjust F1（对照用，偏乐观）。

    区间内只要有一个点被判为正类，就把该区间整体视为已检出。
    """
    labels, scores = _check(labels, scores)
    if threshold is None:
        threshold = best_f1(labels, scores)["threshold"]
    pred = scores >= threshold
    adjusted = pred.copy()
    for start, end in anomaly_ranges(labels):
        if pred[start : end + 1].any():
            adjusted[start : end + 1] = True

    tp = float(np.sum(adjusted & (labels == 1)))
    fp = float(np.sum(adjusted & (labels == 0)))
    fn = float(np.sum((~adjusted) & (labels == 1)))
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "point_adjust_f1": f1,
        "point_adjust_precision": precision,
        "point_adjust_recall": recall,
        "threshold": float(threshold),
    }


def _buffer_labels(labels: np.ndarray, buffer: int) -> np.ndarray:
    labels = (np.asarray(labels).ravel() > 0).astype(np.int8)
    if buffer <= 0:
        return labels.copy()
    out = labels.copy()
    for start, end in anomaly_ranges(labels):
        out[max(0, start - buffer) : min(len(out), end + 1 + buffer)] = 1
    return out


def range_auc_pr(labels: np.ndarray, scores: np.ndarray, buffer: int) -> float:
    """把异常区间向两侧各扩展 buffer 个时刻后计算 AUC-PR。"""
    return auc_pr(_buffer_labels(labels, buffer), scores)


def vus_pr(
    labels: np.ndarray,
    scores: np.ndarray,
    buffers: Iterable[int] = DEFAULT_BUFFERS,
) -> Dict[str, float]:
    """VUS-PR 的开发期近似：在 buffer 维度上对 AUC-PR 取平均。

    官方实现对同一 buffer 序列还会引入 range-recall 维度的积分，
    因此这里的数值**不等于**论文里要报告的 VUS-PR。用它做方法间的
    相对比较是合理的，报绝对数值不行。
    """
    labels = np.asarray(labels).ravel()
    buffers = [int(b) for b in buffers]
    values = [range_auc_pr(labels, scores, b) for b in buffers]
    values = [v for v in values if not np.isnan(v)]
    if not values:
        return {"vus_pr": float("nan")}
    result = {"vus_pr": float(np.mean(values))}
    for buffer, value in zip(buffers, values):
        result[f"auc_pr@{buffer}"] = float(value)
    return result


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    names: Iterable[str] = DEFAULT_METRICS,
    buffers: Iterable[int] = DEFAULT_BUFFERS,
) -> Dict[str, float]:
    """按名称计算指标集合。"""
    labels, scores = _check(labels, scores)
    out: Dict[str, float] = {
        "n_test": int(len(labels)),
        "n_anomaly": int(labels.sum()),
        "anomaly_ratio": float(labels.mean()) if len(labels) else 0.0,
    }
    if labels.sum() == 0:
        # 没有异常就无法计算任何指标，返回 NaN 并在聚合时跳过
        for name in names:
            out[name] = float("nan")
        return out

    for name in names:
        key = str(name).lower()
        if key == "auc_roc":
            out["auc_roc"] = auc_roc(labels, scores)
        elif key == "auc_pr":
            out["auc_pr"] = auc_pr(labels, scores)
        elif key == "best_f1":
            detail = best_f1(labels, scores)
            out["best_f1"] = detail["f1"]
            out["best_precision"] = detail["precision"]
            out["best_recall"] = detail["recall"]
        elif key == "point_adjust_f1":
            out.update(point_adjust_f1(labels, scores))
        elif key == "vus_pr":
            out.update(vus_pr(labels, scores, buffers))
        else:
            raise ValueError(f"未知指标: {name}")
    return out


def aggregate_metrics(
    rows: Sequence[Dict[str, float]],
    names: Iterable[str],
    key: str = "series",
) -> Dict[str, Dict[str, float]]:
    """把逐序列指标聚合成 mean / std / count。"""
    result: Dict[str, Dict[str, float]] = {}
    for name in names:
        values = np.array(
            [r[name] for r in rows if name in r and not np.isnan(r[name])], dtype=np.float64
        )
        result[name] = {
            "mean": float(values.mean()) if values.size else float("nan"),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "count": int(values.size),
        }
    return result