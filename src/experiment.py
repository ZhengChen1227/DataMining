"""可恢复的实验矩阵；验证集选 epsilon，测试集仅评估一次。"""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from .algorithms import predict
from .data import load_graph, split_graph


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def metrics(labels, scores):
    if len(np.unique(labels)) < 2:
        return {"ap": None, "roc_auc": None}
    return {"ap": float(average_precision_score(labels, scores)),
            "roc_auc": float(roc_auc_score(labels, scores))}


def source_hash():
    digest = hashlib.sha256()
    for name in ("algorithms.py", "data.py", "experiment.py"):
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(),
            "packages": {p: importlib.metadata.version(p) for p in
                         ("numpy", "scipy", "networkx", "scikit-learn", "pandas", "matplotlib", "PyYAML")},
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
            "source_sha256": source_hash()}


def scenarios(config):
    for dataset in config["datasets"]:
        for seed in config["seeds"]:
            for retention in config["retentions"]:
                for ratio in config["negative_ratios"]:
                    if ratio != 1 and retention != 1.0:
                        continue
                    yield dataset, seed, retention, ratio


def variants(config):
    yield from ((name, 0) for name in ("cn", "aa", "lp"))
    for budget in config["budgets"]:
        for method in ("fixed", "gate_uniform", "degree_only", "ablp"):
            yield method, budget


def run(config_path):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode() + source_hash().encode()).hexdigest()
    meta_path = output / "manifest.json"
    previous = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    if previous and previous["identity"] != identity:
        raise ValueError("配置或实验源码变化：请使用新 output，避免混入旧结果")
    if not previous:
        write_json(meta_path, {"identity": identity, "config": config, "environment": environment()})
        (output / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    ledger_path = output / "progress.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {"elapsed_seconds": 0}
    # 上一次若非正常结束，用最后心跳到本次启动的时间保守计费，绝不悄悄重置预算。
    elapsed = ledger["elapsed_seconds"]
    if ledger.get("active_since"):
        elapsed += max(0, time.time() - ledger["active_since"])
    start = time.perf_counter()
    deadline = start + max(0, config["max_hours"] * 3600 - elapsed)
    planned = len(list(scenarios(config))) * len(list(variants(config)))
    for folder in ("scores", "splits", "runs"):
        (output / folder).mkdir(exist_ok=True)
    completed = 0

    def checkpoint(status, active=False):
        write_json(ledger_path, {"elapsed_seconds": elapsed + time.perf_counter()-start,
                                "active_since": time.time() if active else None,
                                "status": status, "completed": completed, "planned": planned})

    checkpoint("running", True)
    try:
        for dataset, seed, retention, ratio in scenarios(config):
            if time.perf_counter() >= deadline:
                raise TimeoutError("实验时间预算已耗尽")
            graph, data_info = load_graph(dataset)
            train, parts = split_graph(graph, seed, retention, ratio)
            key = f"{dataset}_s{seed}_r{retention:g}_n{ratio}"
            split_path = output / "splits" / f"{key}.npz"
            if not split_path.exists():
                np.savez_compressed(split_path, **parts)
            write_json(output / "splits" / f"{key}.json", data_info)
            for method, budget in variants(config):
                run_id = f"{key}_{method}_b{budget}"
                result_path = output / "runs" / f"{run_id}.json"
                if result_path.exists():
                    completed += 1
                    continue
                checkpoint("running", True)
                valid = predict(train, parts["valid_pairs"], method, budget, seed, deadline)
                epsilons = config["epsilons"] if method not in ("cn", "aa") else [0.]
                validation = [{"epsilon": e, **metrics(parts["valid_labels"], valid.scores(e))} for e in epsilons]
                # 同分选较小 epsilon；测试标签永远不参与选择。
                chosen = max(validation, key=lambda row: (row["ap"], -row["epsilon"]))["epsilon"]
                test = predict(train, parts["test_pairs"], method, budget, seed, deadline)
                scores = test.scores(chosen)
                zero = test.cn == 0
                row = {"run_id": run_id, "dataset": dataset, "seed": seed, "retention": retention,
                       "negative_ratio": ratio, "method": method, "budget": budget, "epsilon": chosen,
                       "n_candidates": len(scores), "train_nodes": train.number_of_nodes(),
                       "train_edges": train.number_of_edges(), **metrics(parts["test_labels"], scores),
                       "zero_cn_fraction": float(zero.mean()), "zero_cn_count": int(zero.sum()),
                       "zero_cn_positives": int(parts["test_labels"][zero].sum()),
                       "zero_cn_metrics": metrics(parts["test_labels"][zero], scores[zero]),
                       "stats": test.stats, "validation_stats": valid.stats,
                       "validation": validation, "source_sha256": source_hash(), "data": data_info,
                       "split_file": str(split_path), "scores_file": f"scores/{run_id}.npz"}
                np.savez_compressed(output / "scores" / f"{run_id}.npz", scores=scores,
                                    cn=test.cn, paths=test.paths, allocated=test.allocated, visits=test.visits,
                                    valid_scores=valid.scores(chosen), valid_cn=valid.cn,
                                    valid_paths=valid.paths, valid_allocated=valid.allocated, valid_visits=valid.visits)
                write_json(result_path, row)
                completed += 1
                checkpoint("running", True)
                print(f"[{completed}/{planned}] {run_id} AP={row['ap']:.4f} time={test.stats['wall_seconds']:.3f}s", flush=True)
        checkpoint("complete")
        return 0
    except TimeoutError:
        checkpoint("budget_exhausted")
        print("已保存完成的实验；总时间预算耗尽，未完成项目不会伪装为完成。", flush=True)
        return 2
    except BaseException:
        checkpoint("interrupted")
        raise
