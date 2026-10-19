# 阶段一验证记录

验证日期：2026 年 10 月 19 日。实验与测试均使用项目 `.venv` 中的 Python 3.12。

| 检查 | 命令或证据 | 实际结果 |
|---|---|---|
| 自动化测试 | `.venv\Scripts\python.exe -m pytest -q` | 32 passed，最终验证耗时 4.68 秒 |
| 合成冒烟 | `.venv\Scripts\python.exe -m src smoke` | 22/22 组完成，包括零三跳预算和训练边稀疏化 |
| 阶段一真实实验 | `.venv\Scripts\python.exe -m src run --config configs/stage1.yaml` | 22/22 组完成，完整管线约 11.92 秒 |
| 输出汇总 | `.venv\Scripts\python.exe -m src summarize --results results/stage1` | 从实际逐次 JSON 生成 CSV、主表和图 |
| 代码差异 | `git diff --check` | 无空白错误；PowerShell 文件只有 Git 的 LF/CRLF 提示 |
| 图表检查 | 两个网络的效果—成本图 | 已打开检查，标签可读；ABLP 使用星形标记避免重叠曲线被遮住 |
| 原始版本 | `git show archive/tsad-before-ablp:README.md` | 标签保留旧时序异常检测项目 |

测试覆盖：CN/AA 手算，完整三跳与邻接矩阵三次方核对，零预算、容量上限、再分配、
孤立节点、候选顺序稳定性、候选合法性、截止时间、划分与负例互斥、固定种子可复现、
标签不进入评分接口、单类别子组、并列分数、端到端输出和断点续跑。

阶段一没有验证多种子稳定性、统计显著性、实际部署可用性或全局原创性。
耗时是当前桌面环境的观测值，不是独占 CPU 的严格性能基准。
正式矩阵曾运行到 606/760 组后按用户要求停止，状态为 paused_by_user；保留结果，不继续执行。
Word/PDF 论文及最终汇报未生成，属于后续阶段。
