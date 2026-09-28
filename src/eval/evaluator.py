"""评测流程：训练 -> 打分 -> 算指标 -> 聚合 -> 出版式表格。

刻意把「分数」和「指标」分开保存：
分数是原始产物，改指标定义时不需要重跑实验，这在反复调指标口径的过程中很关键。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..datasets import Bundle, build_pipeline
from ..models import build_model
from ..utils.logging import get_logger
from ..utils.seed import set_seed
from .metrics import DEFAULT_BUFFERS, DEFAULT_METRICS, aggregate_metrics, compute_metrics

LOG = get_logger("eval")

ScoreWriter = Callable[[str, np.ndarray], None]


def run_experiment(
    cfg: Any,
    bundle: Optional[Bundle] = None,
    score_writer: Optional[ScoreWriter] = None,
) -> List[Dict[str, Any]]:
    """跑一次完整实验，返回逐序列的结果行。"""
    def get(key: str, default: Any = None) -> Any:
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    seed = int(get("seed", 42) or 42)
    set_seed(seed)

    eval_cfg = get("eval", {}) or {}
    metrics = eval_cfg.get("metrics", DEFAULT_METRICS) if isinstance(eval_cfg, dict) else DEFAULT_METRICS
    buffers = eval_cfg.get("buffer_sizes", DEFAULT_BUFFERS) if isinstance(eval_cfg, dict) else DEFAULT_BUFFERS

    bundle = bundle if bundle is not None else build_pipeline(cfg, seed=seed)
    LOG.info("数据就绪：%d 条序列（%s）", len(bundle), bundle.name)

    rows: List[Dict[str, Any]] = []
    for series in bundle:
        detector = build_model(get("model"))
        scores = detector.fit_score(series.train, series.test).astype(np.float64)

        if not np.all(np.isfinite(scores)):
            LOG.warning("%s: 分数中存在 NaN/Inf，已替换为最小值", series.name)
            finite = scores[np.isfinite(scores)]
            floor = float(finite.min()) if finite.size else 0.0
            scores = np.nan_to_num(scores, nan=floor, posinf=floor, neginf=floor)

        if score_writer is not None:
            score_writer(series.name, scores)

        row: Dict[str, Any] = {
            "dataset": bundle.name,
            "series": series.name,
            "n_channels": series.n_channels,
            "drift": (series.meta.get("drift") or {}).get("name", "none"),
            "model": detector.name,
        }
        row.update(detector.describe())
        row.update(compute_metrics(series.test_label, scores, metrics, buffers))
        rows.append(row)
        LOG.info(
            "%s | %s | VUS-PR(近似)=%s AUC-PR=%s",
            series.name,
            detector.name,
            _fmt(row.get("vus_pr")),
            _fmt(row.get("auc_pr")),
        )
    return rows


def _fmt(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    return "nan" if np.isnan(value) else f"{value:.4f}"


def summarize(
    rows: Sequence[Dict[str, Any]],
    metrics: Iterable[str] = DEFAULT_METRICS,
    group_keys: Sequence[str] = ("drift", "model"),
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """按分组键聚合指标。"""
    buckets: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row.get(k, "none")) for k in group_keys)
        buckets.setdefault(key, []).append(row)

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for key, group in buckets.items():
        label = "/".join(key)
        out[label] = aggregate_metrics(group, metrics)
    return out


def format_table(
    rows: Sequence[Dict[str, Any]],
    metric: str = "vus_pr",
    row_key: str = "model",
    col_key: str = "drift",
    aggregate: str = "mean",
) -> str:
    """生成 Markdown 对比表，用于 README 与论文的表格草稿。"""
    row_values = sorted({str(r.get(row_key, "-")) for r in rows})
    col_values = sorted({str(r.get(col_key, "-")) for r in rows})

    header = [row_key] + col_values
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]

    for row_value in row_values:
        cells = [row_value]
        for col_value in col_values:
            subset = [
                r for r in rows
                if str(r.get(row_key)) == row_value and str(r.get(col_key)) == col_value
            ]
            values = [r[metric] for r in subset if metric in r and not np.isnan(r[metric])]
            if not values:
                cells.append("-")
            elif aggregate == "mean":
                cells.append(f"{np.mean(values):.4f}")
            elif aggregate == "count":
                cells.append(str(len(values)))
            else:
                cells.append("-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def drop_internal_columns(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """去掉以 `_` 开头的内部字段（例如 best_f1 的 tp/fp/fn），便于写进 json。"""
    cleaned = []
    for row in rows:
        cleaned.append({k: v for k, v in row.items() if not str(k).startswith("_")})
    return cleaned