# DataMining · 数据挖掘小组研究

本仓库为数据挖掘课程的方向研究仓库，当前内容为「方向一：分布漂移下的多变量时序异常检测」。

> 原仓库自述：数据挖掘领域小组研究

---

# TSAD-Drift

面向分布漂移的多变量时序异常检测：通道关系建模与测试时校准。

> 课程项目脚手架。目标是把「一个可复现的基准 + 一套漂移评测协议 + 一个方法原型」
> 打包成能直接扩写成 KDD / WWW / ICDM / AAAI 投稿的工程。

## 1. 研究问题

主流时序异常检测方法默认 **训练集与测试集同分布、传感器通道集合固定**。
真实系统里这两条都不成立：工况切换、传感器标定漂移、通道增减都会发生。
本项目要回答三个问题：

1. **现象**：现有方法在什么条件下会崩溃？崩溃有多严重？（动机实验，工作的命脉）
2. **机制**：崩溃的根源是表示不稳定，还是归一化统计量失效，还是通道依赖建模过强？
3. **方法**：如何用最小的改动恢复性能？我们探索三条路线（可任选其一或组合）：
   - 可学习的稀疏通道图，降低对通道数量 / 顺序变化的敏感度；
   - 环境不变性约束（V-REx / IRM 风格），让表示跨工况稳定；
   - 测试时自适应（TTA），在无标签测试流上做轻量校准。

## 2. 数据来源

| 数据 | 来源 | 说明 |
| --- | --- | --- |
| TSB-AD | https://github.com/TheDatumOrg/TSB-AD | NeurIPS 2024 D&B，40+ 数据集统一格式，官方提供划分与评测器 |
| SMAP / MSL | NASA 遥测，TSB-AD 与 TODS 均有打包 | 经典多变量基准 |
| SWaT | iTrust 水处理试验台 | 工控场景 |
| SMD | Server Machine Dataset | 服务器指标 |
| 合成数据 | 本仓库 `src/datasets/synthetic.py` | 无需下载即可跑通全流程，用于 CI 与冒烟测试 |

数据许可证与获取方式见 `data/README.md`。**所有实验必须使用官方评测器复核**，
原因见 `docs/drift_protocol.md`。

## 3. 快速开始

```bash
pip install -r requirements.txt

# 冒烟测试：合成数据 + 合成漂移，无需下载任何真实数据
python -m src.train --config configs/base.yaml --config configs/model/zscore.yaml --drift none -o data.name=synthetic

# 真实数据（先在 data/README.md 里下载 TSB-AD）
python -m src.train --config configs/base.yaml --config configs/model/iforest.yaml --drift temporal
```

跑完一次实验会生成：

- `results/<run_id>/metrics.json`：逐 series 的指标
- `results/<run_id>/scores/`：逐 series 的异常分数（画图 / 显著性检验用）
- `results/<run_id>/config.yaml`：完整快照，保证可复现

全部结果汇总成分组对比表：

```bash
python -m src.evaluate --results results --out results/summary.md
```

## 4. 实验矩阵

`漂移协议 x 方法 x 数据集` 是这三层正交设计：

```bash
# 一键跑完整矩阵（耗时较长）
pwsh scripts/run_all.ps1
```

| 漂移协议 | 配置 | 模拟的真实场景 |
| --- | --- | --- |
| `none` | `configs/base.yaml` | 同分布基线，用于对照 |
| `temporal` | `configs/drift/temporal.yaml` | 工况切换 |
| `channel` | `configs/drift/channel.yaml` | 传感器失效 / 通道顺序变化 |
| `amplitude` | `configs/drift/amplitude.yaml` | 传感器标定漂移 |

## 5. 仓库结构

```
tsad-drift/
├── configs/                  # 实验配置（base + 模型 + 漂移，深合并）
├── data/                     # 数据下载脚本与说明（原始数据不入库）
├── docs/                     # 选题书、漂移协议、实验计划
├── results/                  # 运行产物（只保留 summary.md 与图）
├── scripts/                  # 一键复现脚本
├── src/
│   ├── datasets/             # 数据抽象与加载器
│   ├── drift/                # 漂移协议（本项目的核心贡献之一）
│   ├── models/               # 检测器：经典 / 深度 / 我们的方法
│   ├── eval/                 # 指标、评测流程、官方评测器适配
│   ├── utils/                # 配置、随机种子、日志
│   ├── train.py              # 单次实验入口
│   └── evaluate.py           # 结果汇总与显著性检验
└── tests/                    # 指标正确性与端到端冒烟测试
```

## 6. 主结果表（待填）

| Dataset | IForest | OCSVM | ConvAE | Anomaly Transformer | **Ours** |
| --- | --- | --- | --- | --- | --- |
| SMAP | | | | | |
| MSL | | | | | |
| SWaT | | | | | |
| SMD | | | | | |
| *平均* | | | | | |

指标用 `VUS-PR` 与 `AUC-PR`；`point-adjust F1` 只作为对照列出现，
并需在论文中说明其争议。

## 7. 复现约定

- 固定随机种子 (`seed: 42`)，所有结果记录在 `results/<run_id>/config.yaml`。
- 归一化只使用训练段统计量，且顺序为 `加载 -> 归一化 -> 施加漂移`，
  这样漂移不会被归一化吸收（细节见 `docs/drift_protocol.md`）。
- 每个数据集至少跑 3 个种子，报告均值 ± 标准差。

## 8. 引用

```bibtex
@inproceedings{tsbad2024,
  title={The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark},
  booktitle={NeurIPS Datasets and Benchmarks Track},
  year={2024}
}
```