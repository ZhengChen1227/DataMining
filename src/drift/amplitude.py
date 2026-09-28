"""幅度维漂移：传感器标定漂移。

对测试段逐通道施加 `x' = scale * x + offset`，其中
`scale ~ N(1, scale_std)`、`offset ~ N(0, offset_std)`。

这条协议专门打击依赖「固定阈值 / 固定边缘分布」的方法，
包括逐点 z-score 基线和所有在训练集上拟合归一化统计量的深度模型。

注意：`scale` 是相对量而非全局量。全局统一缩放会被最常见的
z-score 归一化吸收，绝对不能作为漂移实验使用——这正是
`docs/drift_protocol.md` 里要求「先归一化、后漂移」的原因。
"""

from __future__ import annotations

import numpy as np

from ..datasets import Bundle
from .protocol import DriftProtocol, register


@register
class AmplitudeDrift(DriftProtocol):
    name = "amplitude"
    description = (
        "对测试段逐通道施加随机的尺度与偏移漂移，模拟传感器标定漂移。"
    )

    def __init__(
        self,
        scale_std: float = 0.3,
        offset_std: float = 0.5,
        per_channel: bool = True,
        seed: int = 0,
        **params,
    ) -> None:
        super().__init__(seed=seed, **params)
        if scale_std < 0 or offset_std < 0:
            raise ValueError("scale_std / offset_std 不能为负")
        self.scale_std = float(scale_std)
        self.offset_std = float(offset_std)
        self.per_channel = bool(per_channel)

    def apply(self, bundle: Bundle) -> Bundle:
        out = bundle.copy()
        for i, s in enumerate(out.series):
            rng = np.random.default_rng(self.seed + 2000 + i)
            n_channels = s.n_channels
            size = n_channels if self.per_channel else 1
            scale = 1.0 + rng.normal(scale=self.scale_std, size=size)
            offset = rng.normal(scale=self.offset_std, size=size)
            scale = np.where(scale <= 1e-3, 1e-3, scale)  # 避免出现负尺度

            s.meta["drift_detail"] = {
                "scale": [round(float(v), 4) for v in np.ravel(scale)],
                "offset": [round(float(v), 4) for v in np.ravel(offset)],
            }
            s.test = (s.test * scale[None, :] + offset[None, :]).astype(np.float32)
        return self._tag(out)