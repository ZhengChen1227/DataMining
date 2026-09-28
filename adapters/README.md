# 第三方方法适配层

## 为什么单独放一层

复现第三方方法最大的争议是「你到底改了多少」。本层的原则是：
**官方代码原样调用，只写一个薄函数做接口转换**。

## 接入一个方法的完整流程

以 Anomaly Transformer 为例：

1. 克隆官方仓库到仓库外（不要复制进本仓库）：

   ```bash
   git clone https://github.com/thuml/Anomaly-Transformer ../third_party/Anomaly-Transformer
   ```

2. 复制模板：

   ```bash
   cp adapters/example.py adapters/anomaly_transformer.py
   ```

3. 在 `adapters/anomaly_transformer.py` 里把 `fit_score` 的实现换成
   「在 `train` 上训练官方模型 → 在 `test` 上推理 → 归一化为越大越异常」。
   常见注意点：
   - 官方实现通常要求先做逐通道标准化，用**训练段**统计量；
   - 部分实现输出的分数越小越异常，需要取负号；
   - 输出长度必须等于 `len(test)`，多余的重叠窗口请自行归约。

4. 配置里引用它：

   ```yaml
   model:
     name: external
     import_path: adapters.anomaly_transformer
     entrypoint: fit_score
     repo: https://github.com/thuml/Anomaly-Transformer
     params:
       window: 100
       stride: 1
   ```

5. 跑一次并核对：`describe()` 会把 `repo` 与 `params` 写进 `metrics.json`，
   结果表里能直接追溯到你调用了哪个版本的代码。

## 已接/待接清单

| 方法 | 官方仓库 | 状态 |
| --- | --- | --- |
| Anomaly Transformer | https://github.com/thuml/Anomaly-Transformer | 待接 |
| DCdetector | https://github.com/DAMO-DI-ML/KDD2023-DCdetector | 待接 |
| TimesNet | https://github.com/thuml/Time-Series-Library | 待接 |
| GPT4TS | https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All | 待接 |
| kNN 距离（模板示范） | 本仓库自实现 | 已完成 |

建议至少接入两个深度方法，否则「我们的方法更好」这句话缺乏说服力。