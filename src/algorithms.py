"""仅依赖训练图的评分算法；标签只允许进入评测层。"""
from dataclasses import dataclass
from math import log
from time import perf_counter, process_time

import numpy as np

METHODS = ("cn", "aa", "lp", "fixed", "gate_uniform", "degree_only", "ablp")


@dataclass
class Prediction:
    cn: np.ndarray
    paths: np.ndarray
    base: np.ndarray
    allocated: np.ndarray
    visits: np.ndarray
    stats: dict

    def scores(self, epsilon=0.01):
        return self.base + epsilon * self.paths


def allocate(capacities, weights, total, pairs):
    """带容量约束的比例分配，返回整数预算。

    饱和候选先领满，剩余预算重新按权重分配。最终不足一次的余量采用
    最大余数法；同余数按节点对排序，避免输入顺序改变预算。
    零权重候选不参与，total 表示整批候选的三跳访问上限。
    """
    cap = np.asarray(capacities, dtype=np.int64)
    weights = np.asarray(weights, dtype=float)
    if total < 0 or np.any(cap < 0) or np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("预算、容量和权重必须非负且有限")
    result = np.zeros(len(cap), dtype=np.int64)
    active = np.flatnonzero((cap > 0) & (weights > 0))
    remaining = min(int(total), int(cap[active].sum()))
    while len(active) and remaining:
        share = remaining * weights[active] / weights[active].sum()
        full = share >= cap[active]
        if full.any():
            selected = active[full]
            result[selected] = cap[selected]
            remaining -= int(cap[selected].sum())
            active = active[~full]
            continue
        floor = np.floor(share).astype(np.int64)
        result[active] = floor
        remainder = remaining - int(floor.sum())
        order = sorted(range(len(active)), key=lambda j: (
            -float(share[j] - floor[j]), *map(int, pairs[active[j]])
        ))
        for j in order[:remainder]:
            result[active[j]] += 1
        break
    return result


def predict(graph, pairs, method, budget=100, seed=42, deadline=None):
    """输入训练图与非边候选，返回基础分数、三跳命中和操作计数。

    seed 保留在统一接口中；当前算法完全确定性，不使用标签或随机排序。
    wall/cpu 时间包含邻接表准备、CN、预算分配和路径搜索，不含磁盘写入。
    """
    if method not in METHODS or not isinstance(budget, (int, np.integer)) or budget < 0:
        raise ValueError("未知方法或非法预算")
    wall, cpu = perf_counter(), process_time()
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    adjacency = {u: set(graph.neighbors(u)) for u in graph.nodes}
    degree = {u: len(n) for u, n in adjacency.items()}
    ordered = {u: sorted(n, key=lambda v: (degree[v], v)) for u, n in adjacency.items()}
    m = len(pairs)
    cn = np.zeros(m, dtype=np.int64)
    aa = np.zeros(m)
    capacity = np.zeros(m, dtype=np.int64)
    cn_checks = 0
    for i, (u, v) in enumerate(pairs):
        if deadline is not None and i % 256 == 0 and perf_counter() >= deadline:
            raise TimeoutError("实验时间预算已耗尽")
        if u == v or u not in adjacency or v not in adjacency or v in adjacency[u]:
            raise ValueError("候选必须是训练图中两个不同节点之间的非边")
        left, right = (u, v) if degree[u] <= degree[v] else (v, u)
        cn_checks += degree[left]
        for a in ordered[left]:
            if a in adjacency[right]:
                cn[i] += 1
                if method == "aa":
                    aa[i] += 1 / log(degree[a])
        capacity[i] = degree[u] * degree[v]
    alloc_start = perf_counter()
    if method in ("cn", "aa"):
        allocated = np.zeros(m, dtype=np.int64)
    elif method == "lp":
        allocated = capacity.copy()
    elif method == "fixed":
        allocated = np.minimum(capacity, budget)
    else:
        weights = np.ones(m) if method == "gate_uniform" else np.sqrt(capacity)
        if method in ("gate_uniform", "ablp"):
            weights[cn > 0] = 0
        allocated = allocate(capacity, weights, m * budget, pairs)
    allocation_seconds = perf_counter() - alloc_start
    paths, visits = np.zeros(m, dtype=np.int64), np.zeros(m, dtype=np.int64)
    for i, (u, v) in enumerate(pairs):
        limit = int(allocated[i])
        if not limit:
            continue
        # 规范化候选方向，使交换 u、v 后仍按相同的顺序访问。
        u, v = min(u, v), max(u, v)
        hits, used = 0, 0
        for a in ordered[u]:
            neighbors_a = adjacency[a]
            for b in ordered[v]:
                if used >= limit:
                    break
                if deadline is not None and used % 4096 == 0 and perf_counter() >= deadline:
                    raise TimeoutError("实验时间预算已耗尽")
                used += 1
                hits += b in neighbors_a
            if used >= limit:
                break
        paths[i], visits[i] = hits, used
    return Prediction(cn, paths, aa if method == "aa" else cn.astype(float), allocated, visits, {
        "cn_checks": cn_checks,
        "three_hop_visits": int(visits.sum()),
        "total_checks": cn_checks + int(visits.sum()),
        "budget_limit": None if method == "lp" else (m * budget if method not in ("cn", "aa") else 0),
        "allocation_seconds": allocation_seconds,
        "wall_seconds": perf_counter() - wall,
        "cpu_seconds": process_time() - cpu,
    })
