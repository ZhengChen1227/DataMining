"""公开数据获取和静态缺失边恢复划分。完整图不传给评分算法。"""
import gzip
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import networkx as nx
import numpy as np

URLS = {name: f"https://snap.stanford.edu/data/{name}.txt.gz"
        for name in ("ca-GrQc", "email-Eu-core")}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root="data/raw"):
    """下载小型公开压缩包；先验证 gzip 再原子替换，失败不留下假数据。"""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        path = root / f"{name}.txt.gz"
        if not path.exists():
            with urlopen(url, timeout=60) as response:
                payload = response.read(20_000_001)
            if len(payload) > 20_000_000:
                raise ValueError("下载超过预期 20 MB 上限")
            gzip.decompress(payload)
            temporary = path.with_suffix(".partial")
            temporary.write_bytes(payload)
            temporary.replace(path)
        print(name, path.stat().st_size, sha256(path), flush=True)


def load_graph(name, root="data/raw"):
    if name == "synthetic":
        graph = nx.stochastic_block_model([35, 35, 30], [[.15, .01, .02], [.01, .12, .01], [.02, .01, .15]], seed=17)
        checksum = hashlib.sha256(repr(sorted(graph.edges())).encode()).hexdigest()
    else:
        path = Path(root) / f"{name}.txt.gz"
        if not path.exists():
            raise FileNotFoundError(f"缺少 {path}，先运行 python -m src prepare")
        graph = nx.Graph()
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip() and not line.startswith("#"):
                    u, v = map(int, line.split()[:2])
                    graph.add_nodes_from((u, v))
                    if u != v:
                        graph.add_edge(u, v)
        checksum = sha256(path)
    return graph, {"dataset": name, "sha256": checksum, "nodes": graph.number_of_nodes(),
                   "edges": graph.number_of_edges(), "source": URLS.get(name, "synthetic seed=17"),
                   "processing": "undirected, unweighted, no self-loops, unique edges; retain all nodes"}


def split_graph(graph, seed, retention=1., negative_ratio=1):
    """划分一次观测图并嵌套稀疏化训练边。

    验证/测试正边互斥；负例排除完整观测图的所有边。所有比例使用相同
    负例池的前缀，所以 1:1 候选是 1:10 候选的子集。测试标签不用于调参。
    """
    if not 0 < retention <= 1 or negative_ratio not in (1, 10):
        raise ValueError("retention 必须在 (0,1]，negative_ratio 必须为 1 或 10")
    nodes = sorted(graph.nodes())
    edges = np.array(sorted((min(u, v), max(u, v)) for u, v in graph.edges()), dtype=np.int64)
    rng = np.random.default_rng(seed)
    edges = edges[rng.permutation(len(edges))]
    n_train, n_valid = int(.8 * len(edges)), int(.1 * len(edges))
    train_edges = edges[:int(n_train * retention)]
    valid_pos, test_pos = edges[n_train:n_train+n_valid], edges[n_train+n_valid:]
    required = 10 * (len(valid_pos) + len(test_pos))
    if required > len(nodes) * (len(nodes)-1) // 2 - len(edges):
        raise ValueError("图太稠密，无法构造互斥的 1:10 负例池")
    observed = set(map(tuple, edges.tolist()))
    negatives, seen = [], set()
    while len(negatives) < required:
        a, b = rng.choice(nodes, 2, replace=False)
        pair = (min(int(a), int(b)), max(int(a), int(b)))
        if pair not in observed and pair not in seen:
            seen.add(pair)
            negatives.append(pair)
    negative_array = np.array(negatives, dtype=np.int64)
    valid_neg = negative_array[:10*len(valid_pos)][:negative_ratio*len(valid_pos)]
    test_neg = negative_array[10*len(valid_pos):][:negative_ratio*len(test_pos)]
    train = nx.Graph()
    train.add_nodes_from(nodes)
    train.add_edges_from(train_edges)
    parts = {"nodes": np.array(nodes), "train_edges": train_edges}
    for name, pos, neg in (("valid", valid_pos, valid_neg), ("test", test_pos, test_neg)):
        pairs = np.concatenate([pos, neg])
        labels = np.concatenate([np.ones(len(pos), dtype=np.int8), np.zeros(len(neg), dtype=np.int8)])
        permutation = np.random.default_rng(seed + (1000 if name == "valid" else 2000)).permutation(len(pairs))
        parts[f"{name}_pairs"], parts[f"{name}_labels"] = pairs[permutation], labels[permutation]
    return train, parts
