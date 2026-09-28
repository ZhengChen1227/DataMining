"""滑窗工具：所有需要窗口的检测器共用。

窗口张量约定为 (N, W, C)：N 个窗口、每个窗口 W 个时刻、C 个通道。
逐窗口分数通过 `window_scores_to_timesteps` 归约回逐时刻分数，
采用「窗口覆盖到的时刻取平均」，stride > 1 时不会留下空洞。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def make_windows(x: np.ndarray, window: int, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (窗口张量 (N, W, C), 每个窗口的起始下标 (N,))。

    序列短于窗口时在尾部复制最后一个时刻，保证至少产出一个窗口。
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if window <= 1:
        return x[:, None, :], np.arange(len(x))

    if len(x) < window:
        pad = np.repeat(x[-1:], window - len(x), axis=0)
        x = np.concatenate([x, pad], axis=0)

    total = len(x)
    starts = np.arange(0, total - window + 1, max(1, stride))
    if starts[-1] != total - window:
        starts = np.append(starts, total - window)

    view = np.lib.stride_tricks.sliding_window_view(x, window, axis=0)  # (T-W+1, C, W)
    windows = view[starts].transpose(0, 2, 1)  # (N, W, C)
    return np.ascontiguousarray(windows, dtype=np.float32), starts.astype(np.int64)


def window_scores_to_timesteps(
    scores: np.ndarray,
    starts: np.ndarray,
    window: int,
    total: int,
) -> np.ndarray:
    """把逐窗口分数摊回逐时刻分数。"""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    acc = np.zeros(total, dtype=np.float64)
    cnt = np.zeros(total, dtype=np.float64)

    if window <= 1:
        n = min(total, len(scores))
        return scores[:n].astype(np.float32)

    for value, start in zip(scores, starts):
        end = min(int(start) + window, total)
        acc[start:end] += value
        cnt[start:end] += 1
    cnt = np.where(cnt == 0.0, 1.0, cnt)
    return (acc / cnt).astype(np.float32)