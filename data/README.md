# 数据来源与使用范围

| 名称 | 官方页面 | 原始文件 |
|---|---|---|
| ca-GrQc | https://snap.stanford.edu/data/ca-GrQc.html | https://snap.stanford.edu/data/ca-GrQc.txt.gz |
| email-Eu-core | https://snap.stanford.edu/data/email-Eu-core.html | https://snap.stanford.edu/data/email-Eu-core.txt.gz |

下载入口：`python -m src prepare`。两个 gzip 总计约 185 KB，写入 data/raw。
原始包不提交 Git，不附加自创许可证；使用和再分发须遵循原始网站及对应数据来源条款。
引用数据时保留官方数据页面给出的出处。

ca-GrQc 表示科研合作关系。email-Eu-core 原始为有向邮件网络，本实验取无向投影，
因此处理后的边数与原始页面统计可以不同。原始自环仍让节点进入节点集合，但不会进入边集合。
数据不包含训练标签：正负例由静态随机留边协议构造，见 docs/protocol.md。
