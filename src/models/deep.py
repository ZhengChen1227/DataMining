"""深度基线。

- `ConvAutoEncoderDetector`：完全可跑的 1D 卷积自编码器，
  只依赖 torch，作为深度方法的 backbone 对照。
- `ExternalDetector`：把第三方官方实现（Anomaly Transformer、
  DCdetector、TimesNet、GPT4TS / Lag-Llama 等）包装进来的统一适配器。
  不要在本仓库里重新实现它们，直接调官方代码，避免复现争议。

外部方法的接入契约见 `ExternalDetector` 的文档字符串。
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional

import numpy as np

from ..utils.logging import get_logger
from ..utils.seed import set_seed
from .base import Detector, register_model
from .classic import _WindowedDetector
from .window import make_windows

LOG = get_logger("models.deep")


def select_device(device: str = "auto"):
    import torch

    if device and device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _TorchDetector(_WindowedDetector):
    """深度检测器的公共训练循环。

    支持多环境训练（`n_env > 1`）：把训练窗口按时间切成 E 个连续环境，
    每个优化步从每个环境采一个 minibatch，得到 E 个 risk，再按
    `loss = mean(risk) + env_penalty * Var(risk)` 组合后单次反传。
    这正是 IRM / V-REx 系列的实现方式，`env_penalty=0` 时退化为普通训练。
    """

    deep = True

    def __init__(
        self,
        window: int = 64,
        stride: int = 1,
        epochs: int = 20,
        batch_size: int = 128,
        lr: float = 1e-3,
        hidden: int = 64,
        latent: int = 32,
        dropout: float = 0.0,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
        device: str = "auto",
        seed: int = 0,
        n_env: int = 1,
        env_penalty: float = 0.0,
        verbose: bool = False,
        **params: Any,
    ) -> None:
        super().__init__(window=window, stride=stride, **params)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.hidden = int(hidden)
        self.latent = int(latent)
        self.dropout = float(dropout)
        self.weight_decay = float(weight_decay)
        self.grad_clip = float(grad_clip)
        self.device = device
        self.seed = int(seed)
        self.n_env = int(n_env)
        self.env_penalty = float(env_penalty)
        self.verbose = bool(verbose)
        self.device_ = select_device(device)
        self.model_ = None

    # --- 子类需要实现的钩子 ---------------------------------------------
    def _build_model(self, n_channels: int):
        raise NotImplementedError

    def _losses(self, model, batch) -> Dict[str, Any]:
        import torch

        recon = model(batch)
        return {"loss": torch.nn.functional.mse_loss(recon, batch)}

    def _regularizers(self, model) -> Dict[str, Any]:
        return {}

    # --- 训练 -----------------------------------------------------------
    def _split_envs(self, windows: np.ndarray) -> List[Any]:
        import torch

        n = len(windows)
        k = max(1, min(self.n_env, n))
        bounds = np.linspace(0, n, k + 1).astype(int)
        envs = []
        for i in range(k):
            seg = windows[bounds[i] : bounds[i + 1]]
            if len(seg):
                envs.append(torch.from_numpy(np.ascontiguousarray(seg)).to(self.device_))
        return envs or [torch.from_numpy(np.ascontiguousarray(windows)).to(self.device_)]

    def fit(self, train: np.ndarray, train_label: Optional[np.ndarray] = None) -> "_TorchDetector":
        import torch

        set_seed(self.seed)
        train = np.asarray(train, dtype=np.float32)
        self._fit_stats(train)
        z = self._standardize(train)
        windows, _ = make_windows(z, self.window, self.stride)

        n_channels = windows.shape[-1]
        self.model_ = self._build_model(n_channels).to(self.device_)
        opt = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        envs = self._split_envs(windows)
        steps_per_epoch = max(
            1, sum(max(1, len(env) // self.batch_size) for env in envs)
        )

        self.model_.train()
        for epoch in range(self.epochs):
            running = 0.0
            for _ in range(steps_per_epoch):
                risks = []
                for env in envs:
                    size = min(self.batch_size, len(env))
                    idx = torch.randint(0, len(env), (size,), device=self.device_)
                    risks.append(self._losses(self.model_, env[idx])["loss"])

                stacked = torch.stack(risks)
                loss = stacked.mean()
                if self.env_penalty > 0 and len(risks) > 1:
                    loss = loss + self.env_penalty * stacked.var(unbiased=False)
                for value in self._regularizers(self.model_).values():
                    loss = loss + value

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model_.parameters(), self.grad_clip)
                opt.step()
                running += float(loss.detach())

            if self.verbose and (epoch + 1) % max(1, self.epochs // 5) == 0:
                LOG.info(
                    "epoch %d/%d loss=%.6f", epoch + 1, self.epochs, running / steps_per_epoch
                )

        self.model_.eval()
        return self

    # --- 打分 -----------------------------------------------------------
    def _window_errors(self, windows: np.ndarray) -> np.ndarray:
        import torch

        if len(windows) == 0:
            return np.zeros(0, dtype=np.float64)
        self.model_.eval()
        tensor = torch.from_numpy(np.ascontiguousarray(windows)).to(self.device_)
        out: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(tensor), 1024):
                chunk = tensor[i : i + 1024]
                recon = self.model_(chunk)
                err = ((recon - chunk) ** 2).mean(dim=(1, 2))
                out.append(err.detach().cpu().numpy().astype(np.float64))
        return np.concatenate(out)

    def score(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        z = self._standardize(x)
        windows, starts = make_windows(z, self.window, self.stride)
        errors = self._window_errors(windows)
        return _reduce_errors(errors, starts, self.window, len(x))

    def describe(self) -> Dict[str, Any]:
        return {
            "model": self.name,
            "window": self.window,
            "stride": self.stride,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "hidden": self.hidden,
            "latent": self.latent,
            "dropout": self.dropout,
            "n_env": self.n_env,
            "env_penalty": self.env_penalty,
            "seed": self.seed,
            "device": str(self.device_),
            **self.params,
        }


def _reduce_errors(errors: np.ndarray, starts: np.ndarray, window: int, total: int) -> np.ndarray:
    from .window import window_scores_to_timesteps

    return window_scores_to_timesteps(errors, starts, window, total)


def build_conv_autoencoder(n_channels: int, window: int, hidden: int, latent: int, dropout: float):
    import torch.nn as nn

    class ConvAutoEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.window = window
            self.enc = nn.Sequential(
                nn.Conv1d(n_channels, hidden, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden, latent, kernel_size=5, padding=2, stride=2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.dec = nn.Sequential(
                nn.ConvTranspose1d(latent, hidden, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden, n_channels, kernel_size=5, padding=2),
            )

        def forward(self, x):  # x: (B, W, C) -> (B, W, C)
            import torch.nn.functional as F

            seq = x.transpose(1, 2)  # (B, C, W)
            recon = self.dec(self.enc(seq))
            if recon.shape[-1] < self.window:
                recon = F.pad(recon, (0, self.window - recon.shape[-1]))
            elif recon.shape[-1] > self.window:
                recon = recon[..., : self.window]
            return recon.transpose(1, 2)

    return ConvAutoEncoder()


@register_model
class ConvAutoEncoderDetector(_TorchDetector):
    """1D 卷积自编码器，用重构误差作为异常分数。"""

    name = "convae"

    def _build_model(self, n_channels: int):
        return build_conv_autoencoder(
            n_channels, self.window, self.hidden, self.latent, self.dropout
        )


@register_model
class ExternalDetector(Detector):
    """包装第三方官方实现的适配器。

    接入契约：被包装模块必须提供一个可调用对象

        入口函数(train: np.ndarray, test: np.ndarray, **params) -> np.ndarray

    其中 `train`/`test` 都是 (T, C) 数组，返回值是 (T_test,) 的异常分数
    （越大越异常）。推荐在自己的 `adapters/` 目录里为每个官方仓库写一个
    这样的薄函数，再用本适配器统一调用，好处是：

    - 官方代码超参保持原样，复现争议最小；
    - 本仓库不需要把第三方源码复制进来（避免许可证问题）；
    - 结果表里 `params` 会完整记录调用了哪个入口、传了什么参数。
    """

    name = "external"
    requires_fit = False

    def __init__(
        self,
        import_path: str = "",
        entrypoint: str = "fit_score",
        repo: str = "",
        params: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> None:
        super().__init__(
            import_path=import_path, entrypoint=entrypoint, repo=repo, **(params or {}), **extra
        )
        self.import_path = import_path
        self.entrypoint = entrypoint
        self.repo = repo
        self.call_params = dict(params or {})

    def _resolve(self):
        if not self.import_path:
            raise ValueError(
                "ExternalDetector 需要 import_path，例如 adapters.anomaly_transformer"
            )
        module = importlib.import_module(self.import_path)
        fn = getattr(module, self.entrypoint, None)
        if fn is None:
            raise AttributeError(
                f"{self.import_path} 中找不到入口 {self.entrypoint!r}"
            )
        return fn

    def fit(self, train: np.ndarray, train_label: Optional[np.ndarray] = None) -> "ExternalDetector":
        self._resolve()
        return self

    def score(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - 由 fit_score 覆盖
        raise NotImplementedError("ExternalDetector 请使用 fit_score(train, test)")

    def fit_score(
        self,
        train: np.ndarray,
        test: np.ndarray,
        train_label: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        fn = self._resolve()
        scores = np.asarray(fn(np.asarray(train, np.float32), np.asarray(test, np.float32), **self.call_params))
        scores = scores.ravel().astype(np.float64)
        if len(scores) != len(test):
            raise ValueError(
                f"外部方法返回长度 {len(scores)} 与测试段长度 {len(test)} 不一致"
            )
        return scores

    def describe(self) -> Dict[str, Any]:
        return {
            "model": self.name,
            "import_path": self.import_path,
            "entrypoint": self.entrypoint,
            "repo": self.repo,
            **self.call_params,
        }