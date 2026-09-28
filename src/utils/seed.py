"""随机种子统一入口。

注意：这里不开启 `torch.use_deterministic_algorithms(True)`，
它在部分 CUDA 算子上会直接抛错且显著变慢。论文级复现请配合
`CUBLAS_WORKSPACE_CONFIG=:4096:8` 与固定 cuDNN benchmark 开关。
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True