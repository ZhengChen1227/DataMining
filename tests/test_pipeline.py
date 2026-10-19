import json

import numpy as np
import yaml

from src.data import load_graph, split_graph
from src.experiment import metrics, run
from src.summary import summarize


def test_disjoint_and_nested_splits():
    full, _ = load_graph("synthetic")
    train, parts = split_graph(full, 42)
    _, sparse = split_graph(full, 42, .5)
    _, more = split_graph(full, 42, 1., 10)
    _, repeat = split_graph(full, 42)
    for key in parts:
        assert np.array_equal(parts[key], repeat[key])
    assert set(map(tuple, sparse["train_edges"])) <= set(map(tuple, parts["train_edges"]))
    assert set(train.nodes) == set(full.nodes)
    seen = set(map(tuple, parts["train_edges"]))
    negatives = []
    for name in ("valid", "test"):
        pairs, labels = parts[f"{name}_pairs"], parts[f"{name}_labels"]
        assert np.array_equal(pairs, sparse[f"{name}_pairs"])
        assert set(map(tuple, pairs)) <= set(map(tuple, more[f"{name}_pairs"]))
        assert not seen.intersection(map(tuple, pairs))
        seen.update(map(tuple, pairs))
        for (u, v), y in zip(pairs, labels):
            assert full.has_edge(u, v) == bool(y)
        negatives.extend(map(tuple, pairs[labels == 0]))
    assert len(set(negatives)) == len(negatives)


def test_single_class_and_tied_scores():
    assert metrics(np.array([0, 0]), [0, 1])["ap"] is None
    assert metrics(np.array([0, 1, 0, 1]), [0, 0, 0, 0])["ap"] == .5


def test_end_to_end_and_resume(tmp_path):
    config = {"datasets": ["synthetic"], "seeds": [42], "retentions": [1.],
              "negative_ratios": [1], "budgets": [10], "epsilons": [.001, .01],
              "max_hours": .1, "output": str(tmp_path / "results")}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    assert run(path) == 0
    root = tmp_path / "results"
    files = list((root / "runs").glob("*.json"))
    assert len(files) == 7
    times = {p: p.stat().st_mtime_ns for p in files}
    assert run(path) == 0
    assert times == {p: p.stat().st_mtime_ns for p in files}
    frame, summary = summarize(root)
    assert len(frame) == len(summary) == 7
    for path in files:
        row = json.loads(path.read_text(encoding="utf-8"))
        expected = max(row["validation"], key=lambda x: (x["ap"], -x["epsilon"]))["epsilon"]
        assert row["epsilon"] == expected
        assert 0 <= row["ap"] <= 1
        assert (root / row["scores_file"]).exists()


def test_zero_runtime_records_missing_work(tmp_path):
    config = {"datasets": ["synthetic"], "seeds": [42], "retentions": [1.],
              "negative_ratios": [1], "budgets": [10], "epsilons": [.01],
              "max_hours": 0, "output": str(tmp_path / "results")}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    assert run(path) == 2
    progress = json.loads((tmp_path / "results/progress.json").read_text())
    assert progress["completed"] == 0
    assert progress["status"] == "budget_exhausted"
