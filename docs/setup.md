# 环境搭建（RTX 4060 / 8GB 显存）

## 结论先行

用 **WSL2 + Ubuntu** 跑实验，不要在原生 Windows 上装 PyG / pyod 生态。
4060 的 8GB 显存对本项目的所有模型都绰绰有余：`convae` 和 `ours` 的
参数量都在 10 万量级，显存占用 < 1GB，瓶颈在显存之外。

## 1. WSL2

```powershell
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2
```

装好后在 Ubuntu 里确认能看到 GPU：

```bash
nvidia-smi          # 应该显示 RTX 4060 与驱动版本
```

> 不要自己在 WSL 里装显卡驱动，用 Windows 侧的驱动即可。

## 2. Python 环境

```bash
sudo apt update && sudo apt install -y python3.10-venv python3-pip git
cd ~ && git clone <你的仓库地址> tsad-drift && cd tsad-drift
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
```

PyTorch 按官方命令装 CUDA 版本（不要直接 `pip install torch`，会装到 CPU 版）：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望输出：2.x.x True
```

## 3. 冒烟测试

```bash
python -m src.train --list
bash -c "python -m src.train -c configs/base.yaml -c configs/model/zscore.yaml \
  --drift none -o data.name=synthetic --tag smoke"
```

看到逐序列的指标表格就说明环境没问题。

## 4. 8GB 显存下的常用手段

按优先级排序，够用就停：

1. **降 batch size**：本项目的模型对 batch size 不敏感，64 甚至 32 都能收敛。
2. **混合精度**：需要时在训练循环里加 `torch.autocast("cuda")` 与 `GradScaler`。
3. **梯度裁剪**：已经在 `_TorchDetector` 里默认开启（`grad_clip=1.0`）。
4. **减少窗口数**：调大 `stride`（默认 4），窗口数直接按比例下降。
5. **降低 hidden / latent**：`hidden=32, latent=16` 仍然能复现主要结论。

不建议通过削减数据集来省显存，那会改变结论的可比性。

## 5. 常见坑

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `torch.cuda.is_available()` 为 False | 装了 CPU 版 torch | 按上面 cu121 的索引重装 |
| 显存不足 | 窗口数过多或 batch 过大 | 调大 `stride`，调小 `batch_size` |
| 结果不可复现 | 未固定种子 / DataLoader 多进程 | 本项目已 `set_seed`；不要开 `num_workers>0` |
| 指标与参考文献对不上 | 用了本仓库的近似指标 | 用 `src/eval/official.py` 接官方评测器 |