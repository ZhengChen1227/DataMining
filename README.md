# ABLP 链接预测课程项目

**研究题目：面向稀疏网络的计算预算感知型自适应链接预测。**

本项目研究：共同邻居不足时，按节点度数分配有限的三跳搜索预算，是否比固定预算更划算。
ABLP 是课程实验的工作名称，创新假设是预算分配策略，不是首次使用三跳路径。
论文结论依据实际实验，允许方法无优势或在部分条件下失败。

**当前只交付阶段一，服务下周的阶段成果汇报。** 完整论文与系统实验安排在后续阶段，
见 [八周计划](docs/roadmap.md) 和 [阶段一汇报材料](docs/stage1_report.md)。

## 快速运行（Windows PowerShell）

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
powershell -ExecutionPolicy Bypass -File scripts/run_smoke.ps1
# 两个公开小型数据集；无需 GPU、API Key 或付费服务
powershell -ExecutionPolicy Bypass -File scripts/run_stage1.ps1
```

无需激活虚拟环境，直接调用 `.venv` 内的 Python。首次安装通常需要网络；`requirements-lock.txt`
记录本次验证的完整版本，`requirements.txt` 给出最小直接依赖，后者不保证未来环境完全一致。

也可以分别执行：

```powershell
.\.venv\Scripts\python.exe -m src prepare
.\.venv\Scripts\python.exe -m src smoke
.\.venv\Scripts\python.exe -m src run --config configs/stage1.yaml
.\.venv\Scripts\python.exe -m src summarize --results results/stage1
```

重复运行同一配置会跳过已完成方法；配置或核心源码改变时必须使用新输出目录，避免旧结果混入。
阶段一单种子小矩阵的上限为 3 分钟；后续正式矩阵预设总预算 8 小时，目前暂停。
预算耗尽返回退出码 2，其他异常为非零退出码。中断恢复采用保守时间计费，
非正常退出可能将离线间隔也计入预算；需要调整上限时先保留旧结果并明确说明，不静默重置。

## 算法和输入输出

统一入口 `src.algorithms.predict(graph, pairs, method, budget, seed, deadline)`：

- 输入仅有训练图与候选非边，禁止传入标签；节点编号为整数，候选采用 `u < v` 的规范方向。
- 返回 CN 基础项、发现的三跳路径数、逐候选分配预算、实际访问量和计时。
- `Prediction.scores(epsilon)` 形成最终分数；AA 使用自身基础项，不添加三跳项。

| 名称 | 三跳预算策略 |
|---|---|
| `cn` / `aa` | 不扩展三跳 |
| `lp` | 完整枚举，CN + epsilon × 路径数 |
| `fixed` | 每个候选最多 B 次 |
| `gate_uniform` | 仅 CN=0 的候选共享总预算，均分权重 |
| `degree_only` | 所有候选按 sqrt(d(u)d(v)) 加权 |
| `ablp` | 仅 CN=0 的候选按 sqrt(d(u)d(v)) 加权 |

每批候选的总三跳预算为 M×B。ABLP 属于**批量候选评分**方法；改变候选批次或负例数量也会改变预算分配，
不应把它当成与候选集合无关的单节点对相似度。详细定义见 [实验协议](docs/protocol.md)。

## 结果与交付

- [阶段一主表](results/stage1/summary.md)：训练保留比例 100%、正负例 1:1、B=100，只有一个种子。
- `results/stage1/summary.csv`：阶段一两个预算条件的实际数值；单种子不报告标准差。
- `results/stage1/paired_differences.csv`：同种子 ABLP 相对 fixed 的差值。
- `results/stage1/figures/`：预算—效果及消融图；阶段一没有稀疏度实验，不作趋势解读。
- [阶段汇报](docs/stage1_report.md)：当前完成情况、实测结果、失败条件与下周讲稿。
- [教学式上手说明](docs/setup.md)、[创新点问答](docs/defense.md)、[实验协议](docs/protocol.md)。

原始压缩数据、逐候选分数及划分文件保留在本地但默认不提交 Git，可用同一配置重新生成。
逐次 JSON、汇总与图表是可审计的实验产物，不手工填写“提升”结果。

## 项目结构

```text
src/          图处理、评分、实验、汇总和命令行
configs/      正式矩阵与合成冒烟配置
tests/        手算、矩阵参照、预算边界、防泄漏与端到端测试
scripts/      PowerShell 复现入口及报告生成工具
docs/         协议、环境与答辩材料
data/raw/     下载的公开数据（不入 Git）
results/      配置、环境、结果、分数、划分、图表
reports/      后续正式论文的预留位置（阶段一不生成）
```

## 项目版本

默认分支 [main](https://github.com/ZhengChen1227/DataMining) 使用新版 ABLP 链接预测项目。
当前目录仅包含链接预测相关实现、文档与实验结果。

## 后续完整实验

`configs/course.yaml` 与 `scripts/run_all.ps1` 已准备好，但本轮不继续运行。
`results/course/progress.json` 记录暂停状态；此前已完成的部分保留供后续复核，
不能把该目录当成已经完成的最终论文实验。阶段一资料与它分开保存。

## 参考来源

- Zhou, Lü, Zhang (2009), Predicting Missing Links via Local Information. https://arxiv.org/abs/0901.0553
- Lü, Jin, Zhou (2009), Similarity Index Based on Local Paths for Link Prediction. https://doi.org/10.1103/PhysRevE.80.046122
- SNAP 数据来源与处理记录见 [data/README.md](data/README.md)。
