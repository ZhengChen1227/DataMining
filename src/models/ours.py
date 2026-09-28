"""我们的方法：通道图 + 环境不变性 + 测试时校准。

三个组件都可以单独开关，配置里用 `variant` 或直接给布尔开关控制，
这样才能做干净的消融：

| 组件 | 开关 | 作用 |
| --- | --- | --- |
| 可学习稀疏通道图 | `topk > 0` | 只依赖少量通道关系，降低通道置换 / 增减的伤害 |
| 环境不变性 (V-REx) | `env_penalty > 0` 且 `n_env > 1` | 让表示跨工况稳定 |
| 测试时校准 (TTA) | `tta: true` | 无标签地重估归一化统计量并轻量微调 |

`tta` 内部又分两级，报告时必须拆开：
`tta_recenter`（只用测试流重估均值方差，等价于 adaptive_zscore 的深度版本）
与 `tta_finetune`（再额外做自监督微调）。审稿人一定会问「收益到底来自哪一级」。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..utils.logging import get_logger
from .base import register_model
from .deep import _TorchDetector
from .window import make_windows, window_scores_to_timesteps

LOG = get_logger("models.ours")


def build_graph_autoencoder(
    n_channels: int,
    window: int,
    hidden: int,
    latent: int,
    topk: int,
    dropout: float,
    gamma_init: float,
    init_std: float,
):
    import torch
    import torch.nn as nn

    class ChannelGraph(nn.Module):
        """可学习的通道邻接矩阵，top-k 稀疏化后行归一化。"""

        def __init__(self) -> None:
            super().__init__()
            # 随机小初始化：zeros 会让邻接矩阵对称，从而对通道置换完全免疫，
            # 那不是我们想测的模型，也会让 channel 协议的对照失去意义。
            self.adj = nn.Parameter(torch.randn(n_channels, n_channels) * init_std)
            self.topk = int(min(max(1, topk), n_channels))

        def matrix(self):
            weight = torch.softmax(self.adj, dim=-1)
            if self.topk < n_channels:
                # 用索引而非阈值做稀疏化：softmax 输出可能大量并列，
                # 阈值法在并列时会选出远多于 topk 的元素。
                topk_idx = weight.topk(self.topk, dim=-1).indices
                mask = torch.zeros_like(weight).scatter_(-1, topk_idx, 1.0)
                weight = weight * mask
            return weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    class GraphAutoEncoder(nn.Module):
        """逐通道编码 -> 通道图消息传递 -> 逐通道解码。"""

        def __init__(self) -> None:
            super().__init__()
            self.window = window
            self.input_proj = nn.Linear(window, hidden)
            self.graph = ChannelGraph()
            self.message = nn.Linear(hidden, hidden)
            self.norm = nn.LayerNorm(hidden)
            self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
            self.head = nn.Linear(hidden, latent)
            self.decoder = nn.Linear(latent, window)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):  # x: (B, W, C)
            import torch.nn.functional as F

            seq = x.transpose(1, 2)  # (B, C, W)
            h = F.relu(self.input_proj(seq))  # (B, C, H)
            mixed = F.relu(self.message(torch.matmul(self.graph.matrix(), h)))
            h = self.norm(h + self.gamma * mixed)
            z = self.head(self.dropout(h))  # (B, C, L)
            recon = self.decoder(z)  # (B, C, W)
            return recon.transpose(1, 2)

        def adjacency(self) -> np.ndarray:  # 画通道关系图用
            with torch.no_grad():
                return self.graph.matrix().detach().cpu().numpy()

    return GraphAutoEncoder()


@register_model
class GraphInvariantDetector(_TorchDetector):
    """通道图 + 环境不变性 + 测试时校准。"""

    name = "ours"

    def __init__(
        self,
        topk: int = 3,
        gamma_init: float = 0.1,
        init_std: float = 0.01,
        sparsity: float = 1e-3,
        tta: bool = False,
        tta_finetune: bool = False,
        tta_steps: int = 30,
        tta_lr: float = 1e-4,
        tta_scope: str = "norm",
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self.topk = int(topk)
        self.gamma_init = float(gamma_init)
        self.init_std = float(init_std)
        self.sparsity = float(sparsity)
        self.tta = bool(tta)
        self.tta_finetune = bool(tta_finetune)
        self.tta_steps = int(tta_steps)
        self.tta_lr = float(tta_lr)
        self.tta_scope = str(tta_scope)

    # --- 构建与损失 -----------------------------------------------------
    def _build_model(self, n_channels: int):
        return build_graph_autoencoder(
            n_channels=n_channels,
            window=self.window,
            hidden=self.hidden,
            latent=self.latent,
            topk=self.topk,
            dropout=self.dropout,
            gamma_init=self.gamma_init,
            init_std=self.init_std,
        )

    def _regularizers(self, model) -> Dict[str, Any]:
        if self.sparsity <= 0:
            return {}
        return {"graph_sparsity": self.sparsity * model.graph.matrix().abs().mean()}

    # --- 测试时校准 -----------------------------------------------------
    def _recenter(self, x: np.ndarray) -> None:
        """用测试流自身的统计量替换训练统计量（无标签）。"""
        mu = x.mean(axis=0)
        sd = x.std(axis=0)
        self.mu_ = mu.astype(np.float32)
        self.sd_ = np.where(sd < self.eps, 1.0, sd).astype(np.float32)

    def _finetune(self, x: np.ndarray) -> None:
        import torch

        z = self._standardize(x)
        windows, _ = make_windows(z, self.window, self.stride)
        tensor = torch.from_numpy(np.ascontiguousarray(windows)).to(self.device_)

        params = [
            p
            for name, p in self.model_.named_parameters()
            if self._tta_selected(name)
        ]
        if not params:
            return

        opt = torch.optim.Adam(params, lr=self.tta_lr)
        self.model_.train()
        for _ in range(self.tta_steps):
            size = min(self.batch_size, len(tensor))
            idx = torch.randint(0, len(tensor), (size,), device=self.device_)
            loss = self._losses(self.model_, tensor[idx])["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        self.model_.eval()

    def _tta_selected(self, name: str) -> bool:
        if self.tta_scope == "all":
            return True
        if self.tta_scope == "norm":
            return "norm" in name
        if self.tta_scope == "norm+head":
            return ("norm" in name) or name.startswith("head")
        raise ValueError(f"未知 tta_scope: {self.tta_scope}")

    def score(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.tta:
            # 两级校准严格分开，便于消融归因
            self._recenter(x)
            if self.tta_finetune:
                self._finetune(x)
        z = self._standardize(x)
        windows, starts = make_windows(z, self.window, self.stride)
        errors = self._window_errors(windows)
        return window_scores_to_timesteps(errors, starts, self.window, len(x))

    # --- 分析接口 -------------------------------------------------------
    def adjacency(self) -> Optional[np.ndarray]:
        """返回学到的通道邻接矩阵，用于论文里的机制分析图。"""
        if self.model_ is None:
            return None
        return self.model_.adjacency()

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "topk": self.topk,
                "gamma_init": self.gamma_init,
                "init_std": self.init_std,
                "sparsity": self.sparsity,
                "tta": self.tta,
                "tta_finetune": self.tta_finetune,
                "tta_steps": self.tta_steps,
                "tta_lr": self.tta_lr,
                "tta_scope": self.tta_scope,
            }
        )
        return info