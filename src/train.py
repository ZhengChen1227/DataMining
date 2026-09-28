"""单次实验入口。

用法::

    # 冒烟测试（不需要任何真实数据）
    python -m src.train -c configs/base.yaml -c configs/model/zscore.yaml \
        --drift none -o data.name=synthetic

    # 完整实验
    python -m src.train -c configs/base.yaml -c configs/model/iforest.yaml --drift temporal

    # 查看可用组件
    python -m src.train --list
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .drift import list_protocols
from .eval import drop_internal_columns, format_table, run_experiment, summarize
from .models import list_models
from .utils.config import dump_config, load_config
from .utils.logging import get_logger

LOG = get_logger("train")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TSAD-Drift 单次实验",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config", action="append", default=[],
        help="YAML 配置文件，可重复指定，后面的覆盖前面的",
    )
    parser.add_argument("--drift", default=None, help="覆盖漂移协议名")
    parser.add_argument("--model", default=None, help="覆盖检测器名")
    parser.add_argument("--seed", type=int, default=None, help="覆盖随机种子")
    parser.add_argument("--tag", default=None, help="自定义 run_id（默认自动生成）")
    parser.add_argument("--out", default="results", help="结果根目录")
    parser.add_argument("--no-scores", action="store_true", help="不保存逐序列分数")
    parser.add_argument("-o", "--override", action="append", default=[], help="形如 key=value 的覆盖")
    parser.add_argument("--list", action="store_true", help="列出全部可用组件后退出")
    return parser.parse_args(argv)


def print_registry() -> None:
    print("可用检测器：")
    for name, doc in list_models().items():
        print(f"  {name:<16} {doc}")
    print("\n可用漂移协议：")
    for name, doc in list_protocols().items():
        print(f"  {name:<16} {doc}")


def build_overrides(args: argparse.Namespace) -> List[str]:
    overrides = list(args.override)
    if args.drift is not None:
        overrides.append(f"drift.name={args.drift}")
    if args.model is not None:
        overrides.append(f"model.name={args.model}")
    if args.seed is not None:
        overrides.append(f"seed={args.seed}")
    return overrides


def make_run_id(cfg, tag: Optional[str], out_root: Path) -> str:
    if tag:
        return tag
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model = cfg.get_path("model.name", "model")
    drift = cfg.get_path("drift.name", "none")
    seed = cfg.get("seed", 42)
    run_id = f"{stamp}_{model}_{drift}_s{seed}"
    if (out_root / run_id).exists():
        run_id = f"{run_id}_{np.random.default_rng().integers(1000, 9999)}"
    return run_id


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.list:
        print_registry()
        return 0

    configs = args.config or ["configs/base.yaml"]
    cfg = load_config(*configs, overrides=build_overrides(args))

    out_root = Path(args.out)
    run_id = make_run_id(cfg, args.tag, out_root)
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    dump_config(cfg, run_dir / "config.yaml")

    score_dir = run_dir / "scores"
    score_dir.mkdir(exist_ok=True)
    writer = None
    if not args.no_scores:
        def writer(name: str, scores: np.ndarray) -> None:
            np.save(score_dir / f"{name}.npy", scores)

    LOG.info("run_id = %s", run_id)
    rows = run_experiment(cfg, score_writer=writer)
    rows = drop_internal_columns(rows)

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {"run_id": run_id, "config": cfg.to_dict(), "rows": rows},
            fh,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    summary = summarize(rows)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)

    LOG.info("结果已写入 %s", run_dir)
    print()
    print(f"== {run_id} 逐序列结果 ==")
    header = ["series", "n_channels", "anomaly_ratio", "auc_pr", "vus_pr", "best_f1"]
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    str(row.get("series", "-")),
                    str(row.get("n_channels", "-")),
                    f"{row.get('anomaly_ratio', float('nan')):.4f}",
                    _fmt(row.get("auc_pr")),
                    _fmt(row.get("vus_pr")),
                    _fmt(row.get("best_f1")),
                ]
            )
            + " |"
        )
    return 0


def _fmt(value) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    return "nan" if np.isnan(value) else f"{value:.4f}"


if __name__ == "__main__":
    sys.exit(main())