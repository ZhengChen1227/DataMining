"""适配器模板（可直接运行）。

本文件实现一个不依赖任何第三方库的 k 近邻距离检测器，作用有两个：

1. 证明 `ExternalDetector` 的接入契约是可以端到端跑通的；
2. 作为模板，把内部逻辑换成官方仓库的调用即可，签名与返回值不用改。

接入契约
--------
    fit_score(train: np.ndarray, test: np.ndarray, **params) -> np.ndarray

- `train` / `test` 形状均为 (T, C) 的 float32；
- 返回 (T_test,) 的异常分数，**越大越异常**；
- 函数内部可以自由训练、可以调 API、可以加载权重，本仓库不关心；
- 长度必须与 `test` 一致，否则 `ExternalDetector` 会直接报错（这是刻意的）。
"""

from __future__ import annotations

import numpy as np

from src.models.window import make_windows, window_scores_to_timesteps


def fit_score(
    train: np.ndarray,
    test: np.ndarray,
    window: int = 8,
    stride: int = 2,
    n_neighbors: int = 4,
    **kwargs,
) -> np.ndarray:
    """用「到训练集最近窗口的距离」作为异常分数。"""
    train = np.asarray(train, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)

    # 只用训练段统计量标准化，与仓库其它部分的约定保持一致
    center = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    train_z = (train - center) / scale
    test_z = (test - center) / scale

    train_windows, _ = make_windows(train_z, window, stride)
    test_windows, starts = make_windows(test_z, window, stride)
    flat_train = train_windows.reshape(len(train_windows), -1)
    flat_test = test_windows.reshape(len(test_windows), -1)

    # 距离矩阵规模为 (N_test, N_train)，长序列请换成分块或 faiss
    distances = np.linalg.norm(flat_test[:, None, :] - flat_train[None, :, :], axis=2)
    k = min(max(1, n_neighbors), distances.shape[1])
    window_scores = np.sort(distances, axis=1)[:, :k].mean(axis=1)

    return window_scores_to_timesteps(window_scores, starts, window, len(test))