# 数据说明

本目录只放**下载脚本与格式说明**，原始数据不入库（见 `.gitignore` 的 `data/raw/`）。

## 目录结构

脚本会把所有数据源统一成下面这一种结构，加载器 `src/datasets/tsb_ad.py`
只认这个结构：

```
data/raw/tsb-ad/
  <subset_name>/             # 一个子集 = 一条多变量序列
    train.npy                # (T_train, C) float32
    test.npy                 # (T_test,  C) float32
    test_label.npy           # (T_test,)  0/1 int8
    meta.json                # 可选：{"name":..., "source":..., "period":...}
```

命名不匹配时的兼容规则（详见 `src/datasets/tsb_ad.py` 的 `*_PATTERNS`）：
`train.npy` / `*_train.npy` / `train.csv` 都能识别，
标签列名支持 `Label` / `label` / `is_anomaly` / `anomaly` / `target`。

## 数据源

| 数据 | 来源 | 许可证 / 说明 |
| --- | --- | --- |
| TSB-AD | https://github.com/TheDatumOrg/TSB-AD | NeurIPS 2024 D&B 基准，40+ 数据集统一格式，**推荐作为主基准** |
| SMAP / MSL | NASA 遥测数据，TSB-AD 与 TODS 均已打包 | 公开研究用途 |
| SWaT | https://itrust.sutd.edu.sg/testbeds/secure-water-treatment-swat/ | 需向 iTrust 申请，**不能直接分发给他人** |
| SMD | https://github.com/NetManAIOps/OmniAnomaly | 服务器指标，随论文开源 |

> 注意：SWaT 需要签署使用协议，只能自己申请下载，**不要把它提交到公开仓库**。
> 这也是 `.gitignore` 屏蔽 `data/raw/` 的原因之一。

## 下载

```bash
# 1) 克隆官方仓库并尝试自动转换
python data/download_tsb_ad.py --root data/raw/tsb-ad

# 2) 只克隆不转换（自己看官方目录结构）
python data/download_tsb_ad.py --no-convert

# 3) 从已有的本地目录转换（例如你手动下载到了别处）
python data/download_tsb_ad.py --source /path/to/TSB-AD --root data/raw/tsb-ad
```

脚本找不到可以自动识别的文件时**不会假装成功**，而是列出它找到了什么、
期望的命名是什么，你按上面的结构手动放好再跑一次即可。

## 校验

```bash
python -c "
from src.datasets.tsb_ad import list_subsets
from src.datasets import build_dataset
print(list_subsets('data/raw/tsb-ad'))
b = build_dataset({'name': 'tsb_ad', 'root': 'data/raw/tsb-ad', 'limit': 3})
print(b.summary())
"
```

看到每条序列的通道数、训练/测试长度和异常占比就说明数据没问题。
**如果异常占比是 0，说明标签没读对**，先回去检查标签文件，不要继续跑实验。