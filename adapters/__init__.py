"""第三方方法的薄适配层。

每个官方实现写一个模块，暴露统一签名的 `fit_score`，再由
`src.models.deep.ExternalDetector` 调用。这样做的理由：

- 不把第三方源码复制进本仓库（避免许可证问题、避免重实现带来的复现争议）；
- 官方超参保持原样，结果表里记录的是「调用了哪个入口、传了什么参数」；
- 换方法只需要加一个文件，评测流程一行都不用改。
"""

from __future__ import annotations

import numpy as np


def bad_length(train, test, **kwargs):  # pragma: no cover - 仅用于测试契约校验
    """故意返回错误长度的入口，用于测试 ExternalDetector 的契约检查。"""
    return np.zeros(max(1, len(test) - 7))