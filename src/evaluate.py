"""汇总多次实验：生成分组对比表 + 显著性检验。

用法::

    python -m src.evaluate --results results --out results/summary.md \
        --metric vus_pr --ours ours --baseline iforest

约定：同一个 `(drift, model)` 组如果跑了多个种子，表格报均值，
并在 `--per-seed` 时额外列出每个种子的数值。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .eval import format_table
from .eval.stats import paired_by_series
from .utils.logging import get_logger

LOG = get_logger("evaluate")
DEFAULT_METRICS = ("auc_pr", "vus_pr", "best_f1", "auc_roc")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总实验结果")
    parser.add_argument("--results", default="results", help="结果根目录")
    parser.add_argument("--out", default="results/summary.md", help="输出 Markdown 路径")
    parser.add_argument("--metric", default="vus_pr", help="表格主指标")
    parser.add_argument("--ours", default="ours", help="我们的方法名")
    parser.add_argument("--baseline", action="append", default=[], help="对比基线，可重复")
    parser.add_argument("--drift", default=None, help="只汇总指定漂移协议")
    parser.add_argument("--per-seed", action="store_true", help="额外列出每个种子的数值")
    return parser.parse_args(argv)


def load_rows(results_root: Path, drift: Optional[str] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metrics_file in sorted(results_root.glob("*/metrics.json")):
        try:
            payload = json.loads(metrics_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOG.warning("跳过无法解析的文件: %s", metrics_file)
            continue
        run_id = payload.get("run_id", metrics_file.parent.name)
        for row in payload.get("rows", []):
            if drift is not None and str(row.get("drift")) != drift:
                continue
            row = dict(row)
            row["run_id"] = run_id
            rows.append(row)
    return rows


def build_markdown(
    rows: Sequence[Dict[str, Any]],
    metric: str,
    ours: str,
    baselines: Sequence[str],
    per_seed: bool,
) -> str:
    lines: List[str] = ["# 实验结果汇总", ""]
    lines.append(f"主指标：`{metric}`；共 {len(rows)} 条序列级结果。")
    lines.append("")
    lines.append("> 数值来自内部近似实现，投稿前需用 TSB-AD 官方评测器复核")
    lines.append("> （见 `docs/drift_protocol.md`）。")
    lines.append("")

    for name, doc in (("模型 x 漂移", None),):
        lines.append(f"## {name}")
        lines.append("")
        lines.append(format_table(rows, metric=metric, row_key="model", col_key="drift"))
        lines.append("")

    lines.append("## 相对同分布基线的下降幅度")
    lines.append("")
    lines.append(_drop_table(rows, metric))
    lines.append("")

    if baselines:
        lines.append("## 显著性检验（配对 Wilcoxon，按序列配对）")
        lines.append("")
        header = ["baseline", "n", "mean_delta", "wins", "losses", "p_value", "结论"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for baseline in baselines:
            result = paired_by_series(rows, metric, ours, baseline)
            lines.append(
                "| "
                + " | ".join(
                    [
                        baseline,
                        str(result.get("n", "-")),
                        _fmt(result.get("mean_delta")),
                        str(result.get("wins", "-")),
                        str(result.get("losses", "-")),
                        _fmt(result.get("p_value")),
                        str(result.get("note", "-")),
                    ]
                )
                + " |"
            )
        lines.append("")

    if per_seed:
        lines.append("## 逐种子明细")
        lines.append("")
        lines.append("| run_id | drift | model | series | " + metric + " |")
        lines.append("| " + " | ".join(["---"] * 5) + " |")
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("run_id")),
                        str(row.get("drift")),
                        str(row.get("model")),
                        str(row.get("series")),
                        _fmt(row.get(metric)),
                    ]
                )
                + " |"
            )
        lines.append("")

    return "\n".join(lines)


def _drop_table(rows: Sequence[Dict[str, Any]], metric: str) -> str:
    """相对同分布 (drift=none) 的性能下降，负值表示漂移下变差。"""
    none_baseline: Dict[str, float] = {}
    buckets: Dict[tuple, List[float]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("model")), str(row.get("drift")))
        value = row.get(metric)
        if value is None or np.isnan(float(value)):
            continue
        buckets[key].append(float(value))

    for (model, drift), values in buckets.items():
        if drift == "none":
            none_baseline[model] = float(np.mean(values))

    header = ["model", "drift", metric, "同分布基线", "绝对下降"]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for (model, drift), values in sorted(buckets.items()):
        if drift == "none":
            continue
        base = none_baseline.get(model)
        current = float(np.mean(values))
        delta = "-" if base is None else f"{current - base:+.4f}"
        lines.append(
            "| "
            + " | ".join([model, drift, f"{current:.4f}", _fmt(base), delta])
            + " |"
        )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if np.isnan(value):
        return "nan"
    return f"{value:.4f}" if abs(value) >= 1e-4 else f"{value:.2e}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    results_root = Path(args.results)
    rows = load_rows(results_root, args.drift)
    if not rows:
        LOG.error("在 %s 下没有找到任何结果，先跑 `python -m src.train`", results_root)
        return 1

    markdown = build_markdown(rows, args.metric, args.ours, args.baseline, args.per_seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8", newline="\n")
    LOG.info("已写入 %s（%d 条结果）", out_path, len(rows))
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())