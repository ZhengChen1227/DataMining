# 从零运行项目

虚拟环境是本项目自己的 Python 工具箱：安装依赖只影响 `.venv`，不会替换系统 Python。
在仓库根目录打开 PowerShell，按 README 的四条命令运行即可。不要把 `.venv` 上传到 Git。

建议阅读顺序：README → docs/protocol.md → src/algorithms.py → src/data.py → src/experiment.py。
先在四节点链 0—1—2—3 上理解为什么 (0,3) 的 CN 为零而三跳路径数为一，再看预算分配。

`python -m src` 表示把 src 当 Python 模块执行；不要直接双击源码文件。
出现 ModuleNotFoundError 时先确认调用的是 `.venv\Scripts\python.exe`，并从仓库根目录运行。
下载失败可重新执行 prepare；已下载且存在的压缩包不会重复下载。数据 SHA256 在实验 JSON 中记录。

改算法后先跑 pytest，再修改输出目录跑新实验。不能把旧结果文件复制到新目录冒充新实现结果。
已有运行结果与源码哈希绑定，核心代码变化会拒绝继续旧目录。

阶段一运行 scripts/run_stage1.ps1，只有一个随机种子，不能据此声称稳定提升。
后续正式实验不需要 GPU，也不需要在 WSL 中运行。最大时间预算覆盖整个正式矩阵，而不是每个配置各 8 小时。
实验完成后先读 summary.md，再对照 paired_differences.csv 查看每个种子的差值，不只看平均数。

阶段一交付 Markdown 汇报材料。正式论文 Word/PDF 留到后续阶段，避免在方法尚未确认时过早排版。
