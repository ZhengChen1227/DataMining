"""从逐次结果自动生成汇总、配对差值和静态科学图。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def summarize(root):
    root = Path(root)
    records = []
    for path in sorted((root / "runs").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        records.append({**{k: v for k, v in row.items() if not isinstance(v, (dict, list))},
                        **row["stats"], "zero_cn_ap": row["zero_cn_metrics"]["ap"]})
    if not records:
        raise ValueError("没有完成的实验，不能生成结果")
    frame = pd.DataFrame(records)
    frame.to_csv(root / "all_runs.csv", index=False)
    group = ["dataset", "retention", "negative_ratio", "method", "budget"]
    values = ["ap", "roc_auc", "wall_seconds", "cpu_seconds", "total_checks", "three_hop_visits", "zero_cn_ap", "zero_cn_fraction"]
    summary = frame.groupby(group)[values].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(root / "summary.csv", index=False)
    abl = frame[frame.method == "ablp"]
    fixed = frame[frame.method == "fixed"]
    keys = ["dataset", "seed", "retention", "negative_ratio", "budget"]
    paired = abl.merge(fixed, on=keys, suffixes=("_ablp", "_fixed"))
    for metric in ("ap", "wall_seconds", "total_checks"):
        paired[f"delta_{metric}"] = paired[f"{metric}_ablp"] - paired[f"{metric}_fixed"]
    paired[keys + [f"delta_{m}" for m in ("ap", "wall_seconds", "total_checks")]].to_csv(root / "paired_differences.csv", index=False)
    figs = root / "figures"
    figs.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    for dataset in frame.dataset.unique():
        base = summary[(summary.dataset == dataset) & (summary.retention == 1) & (summary.negative_ratio == 1)]
        for metric, label, name in (("total_checks", "Measured adjacency checks (CN + 3-hop)", "cost"),
                                    ("wall_seconds", "Scoring wall time (seconds)", "time")):
            fig, ax = plt.subplots(figsize=(7, 4.3))
            for method, rows in base.groupby("method", sort=True):
                rows = rows.sort_values("budget")
                ax.plot(rows[f"{metric}_mean"], rows.ap_mean,
                        "*-" if method == "ablp" else "o-", label=method,
                        markersize=11 if method == "ablp" else 5,
                        zorder=4 if method == "ablp" else 2)
            ax.set(xlabel=label, ylabel="Average precision", title=f"{dataset}: quality vs {name}")
            ax.set_xscale("log")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            fig.savefig(figs / f"{dataset}_{name}.png")
            plt.close(fig)
        selected = summary[(summary.dataset == dataset) & (summary.negative_ratio == 1) & ((summary.budget == 100) | summary.method.isin(["cn", "aa", "lp"]))]
        if selected.retention.nunique() > 1:
            fig, ax = plt.subplots(figsize=(7, 4.3))
            for method, rows in selected.groupby("method"):
                rows = rows.sort_values("retention")
                ax.errorbar(rows.retention, rows.ap_mean, yerr=rows.ap_std.fillna(0), marker="o", label=method, capsize=3)
            ax.set(xlabel="Fraction of training edges retained", ylabel="Average precision", title=f"{dataset}: sparsity (B=100)")
            ax.legend(fontsize=8, ncol=2)
            ax.grid(alpha=.2)
            fig.tight_layout()
            fig.savefig(figs / f"{dataset}_sparsity.png")
            plt.close(fig)
        ablation = base[base.budget == 100].sort_values("method")
        if len(ablation):
            fig, ax = plt.subplots(figsize=(7, 4.3))
            ax.bar(ablation.method, ablation.ap_mean, yerr=ablation.ap_std.fillna(0), capsize=4)
            ax.set(ylabel="Average precision", title=f"{dataset}: ablation (B=100)")
            fig.tight_layout()
            fig.savefig(figs / f"{dataset}_ablation.png")
            plt.close(fig)
    progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))
    lines = ["# 实验结果自动汇总", "", f"运行状态：{progress['status']}；完成 {progress['completed']}/{progress['planned']} 组。",
             "数字来源：runs/*.json；完整标准差及样本数见 summary.csv。单种子标准差记为 —，不作稳定性结论。", "",
             "| 数据集 | 方法 | B | AP 均值 | AP 标准差 | 种子数 | 秒 | 总检查数 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    main = summary[(summary.retention == 1) & (summary.negative_ratio == 1) & ((summary.budget == 100) | summary.method.isin(["cn", "aa", "lp"]))]
    for r in main.itertuples():
        std = '—' if pd.isna(r.ap_std) else f'{r.ap_std:.4f}'
        lines.append(f"|{r.dataset}|{r.method}|{r.budget}|{r.ap_mean:.4f}|{std}|{r.ap_count}|{r.wall_seconds_mean:.4f}|{r.total_checks_mean:.0f}|")
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"汇总 {len(frame)} 个实验：{root / 'summary.md'}")
    return frame, summary
