# SAGD-BLS 当前方法冻结记录

记录日期：2026-05-25  
冻结版本：`SAGD-BLS-v2-cycle`  
主任务：电池单循环电压预测工业测试  
当前阶段目标：先固定一个严格属于 BLS 变体的方法，并在 BattNN 风格实验中形成可复现、可解释、可向导师汇报的结果。

定位说明：本文件冻结的是电池工业验证适配器和当前实验结果，不是最终通用 PDE 求解器定义。通用 PDE 求解器的研究主线见 `SAGD_BLS_general_pde_solver_positioning.md`。

## 1. 冻结结论

当前冻结方法为 **SAGD-BLS-v2-cycle**，即 State-Augmented Gradient-Descent Broad Learning System。

冻结后的主张是：

- SAGD-BLS 是一个 BLS 变体，而不是多模块拼接框架。
- 它保留固定随机映射层、固定随机增强层、单个线性输出层。
- 它不使用 BLS 常见的最小二乘或伪逆求解输出权重，而是用 Adam 梯度下降训练输出权重 `beta`。
- 当前工业测试以 BattNN 单循环电池预测为主，不在本阶段展开 PDE benchmark。
- SimData 是最可靠的主验收结果；Lab/NASA 作为 BattNN 处理数据集支持结果；XJTU 是新增 raw `.mat` 工业风格测试，采用同协议 BattNN tuned adapter 对比。

当前主要结果：

| 数据集 | BattNN 对照 | SAGD-BLS | MAE 降低 |
| --- | ---: | ---: | ---: |
| SimData | 0.020713 ± 0.000982 | 0.005621 ± 0.001605 | 72.9% |
| LabData local B1-B6 | 0.039661 ± 0.004084 | 0.023488 ± 0.004999 | 40.8% |
| NASAData | 0.039656 ± 0.002477 | 0.029400 ± 0.010949 | 25.9% |
| XJTU Batch-5 | 0.104974 ± 0.002318 | 0.026911 ± 0.002916 | 74.4% |
| XJTU Batch-1 | 0.161970 ± 0.012103 | 0.106851 ± 0.056163 | 34.0% |

注意：XJTU 的 BattNN 对照不是 BattNN 论文原始 benchmark，而是在当前 XJTU 抽取协议下经过 compact tuning 的 BattNN adapter。

## 2. 方法名称和定位

方法名称：

`SAGD-BLS`: State-Augmented Gradient-Descent Broad Learning System

中文表述：

状态增强梯度下降宽度学习系统。

方法定位：

SAGD-BLS 是面向电池单循环电压预测的 BLS 变体。当前研究计划中，导师要求研究宽度学习变体，并最终希望用于 PDE 求解。当前阶段不直接展开 PDE 测试，而是先用 BattNN 单循环电池预测作为工业测试闭环，证明该 BLS 变体在小样本、变长预测场景下有竞争力。

为什么是 BLS 变体：

- 输入经过状态增强后进入 BLS 特征空间；
- mapping nodes 是固定随机节点；
- enhancement nodes 是固定随机节点；
- 输出层是一个线性层；
- 训练参数只有输出权重 `beta`；
- 没有 RNN、CNN、Transformer、MLP 堆叠、物理递推模块或模型集成；
- 方法变化集中在 BLS 输入坐标扩展和输出权重优化方式上。

## 3. 模型结构

给定一条电流序列：

`u = [u_0, u_1, ..., u_{T-1}]`

SAGD-BLS 构造状态增强输入：

`X_state ∈ R^{T × d}`

然后经过固定随机映射层：

`Z = activation_map(X_scaled W_map + b_map)`

再经过固定随机增强层：

`H = activation_enhance(Z W_enhance + b_enhance)`

最终设计矩阵为：

`D = [1, X_scaled, Z, H]`

输出电压预测为：

`V_pred = D_scaled beta`

其中：

- `W_map`、`b_map`、`W_enhance`、`b_enhance` 初始化后固定；
- 只有 `beta` 使用 Adam 更新；
- 训练和预测时会对输入、design matrix 和目标电压进行标准化；
- 预测输出再反标准化回真实电压尺度。

当前冻结配置：

| 配置项 | 值 |
| --- | --- |
| `n_map` | 100 |
| `n_enhance` | 100 |
| `map_activation` | `tanh` |
| `enhance_activation` | `sigmoid` |
| `epochs` | 20000 |
| `learning_rate` | 0.01 |
| `l2` | 1e-3 |
| `smooth_l1_delta` | 1.0 |
| `train_length` | 60 |
| `n_train` | 30 |
| seeds | `[1, 2, 3, 4, 5]` |

说明：早期计划默认使用 `sigmoid/sigmoid`，但实验消融显示 `tanh/sigmoid` 是当前更优配置，因此冻结版采用 `tanh/sigmoid`。

## 4. 状态增强特征

基础状态特征只由电流序列构造，不读取测试电压。

对每个时间点 `t`，构造以下特征：

- `current / 8`
- `t / 60`
- `cumsum(current) / 360`
- `diff(current) / 6`
- RC-like 指数滤波电流状态

RC-like 滤波状态使用以下 decay：

`[0.99, 0.98, 0.95, 0.90, 0.80, 0.65, 0.50]`

递推形式：

`s_t = decay * s_{t-1} + (1 - decay) * current_t`

这些特征的作用：

- 当前值提供即时负载；
- 时间坐标提供单循环位置；
- 累积电流近似放电进度；
- 电流差分描述突变；
- 多时间尺度 RC-like 滤波状态描述电流历史。

这些都是输入坐标扩展，不是可训练模块。

## 5. XJTU v2 Cycle Context

XJTU raw `.mat` 数据中 Batch-1 是恒流工况，电流形状几乎不携带跨电池老化差异。current-only SAGD-BLS 在 Batch-1 上失败：

`MAE = 0.195916`

因此 v2 加入低维循环上下文：

- `cycle / 500`
- `duration_min / 60`

冻结版 XJTU 主结果使用 `cycle context`。

这个 context 的边界：

- 它是输入坐标的一部分；
- 它不引入可训练前端网络；
- 它不使用完整测试电压；
- `duration_min` 来自完整当前循环的电流/时间长度，适用于“给定完整单循环电流，预测完整电压”的当前任务协议；
- 如果未来任务变成在线早期预测，则需要重新定义可用特征边界。

保留但不作为主版本的消融：

`cycle_voltage`

该消融使用同一循环最开始少量电压点，因此改变了预测协议。当前结果中它没有超过纯 `cycle context`，所以冻结版不采用它作为主方法。

## 6. 训练目标

训练目标是标准化电压上的加权 SmoothL1 + L2 正则。

电压标准化：

`y_scaled = (y - mean(y_train)) / std(y_train)`

时间权重：

`weights = linspace(train_length / 5, 1, train_length)`

然后按均值归一化：

`weights = weights / mean(weights)`

这会让早期时间点权重更大，和 BattNN 原实现中的 weighted loss 方向一致。

SmoothL1：

- `delta = 1.0`
- 小残差区域近似 L2；
- 大残差区域近似 L1；
- 对异常点比纯 L2 更稳。

最终目标：

`mean(weights * smooth_l1(D beta - y_scaled)) + 0.5 * l2 * ||beta||^2`

优化器：

- Adam
- `beta1 = 0.9`
- `beta2 = 0.999`
- `eps = 1e-8`
- `lr = 0.01`
- epoch 数为 20000

重要约束：

不使用最小二乘、不使用伪逆、不用 ridge closed-form。输出权重完全通过 Adam 梯度下降训练。

## 7. 数据和实验协议

### 7.1 SimData

数据路径：

- 训练：`BattNN/data/SimData/current=[2, 8] len=[60, 200] train`
- 测试：`BattNN/data/SimData/current=[2, 8] len=[60, 200] test`

协议：

- `random.seed(2022)`
- 训练样本数 `n_train = 30`
- 训练长度 `train_length = 60`
- 测试完整变长循环
- seeds `[1, 2, 3, 4, 5]`
- 主指标 MAE，同时报告 MAPE/RMSE

主结果采用 `tanh/sigmoid`。

额外公平性 audit：

按 BattNN loader 的五次抽样语义重新运行 SAGD-BLS，平均 MAE 为：

`0.009403 ± 0.006407`

该结果比固定划分的 `0.005621` 更保守，但仍低于 BattNN 本地复现的 `0.020713`。

### 7.2 LabData

数据路径：

`BattNN/data/LabData`

说明：

本地只有 B1-B6 `.npy` 文件，BattNN 原始记录包含 B1-B8。因此 LabData 结果只能作为本地支持性结果，不能完全等价于论文完整设置。

本地有一次 BattNN LabData run 未收敛，MAE 高于 0.3。当前表中使用 MAE < 0.1 的干净 BattNN 均值。

### 7.3 NASAData

数据路径：

`BattNN/data/NASA11`

使用 BattNN 处理后的 NASAData，包含：

- `Dis_RW3.mat`
- `Dis_RW4.mat`
- `Dis_RW5.mat`
- `Dis_RW6.mat`

NASA BattNN 本地复现与仓库记录接近，因此是比较可靠的支持性结果。

### 7.4 XJTU Batch-5

数据路径：

`XJTU battery dataset/Batch-5`

协议：

- 读取原始 `.mat`
- 提取负电流随机工况放电片段
- 将放电电流转成正值
- 多段放电片段按时间拼接
- 每 0.5 分钟重采样
- leave-one-battery-out
- 电池：`RW_battery-1` 到 `RW_battery-8`
- 每个 held-out battery 使用其余电池中 30 条长度足够的 cycle 训练
- 训练长度为 60
- 测试 held-out battery 的完整 cycle

冻结版主结果：

`SAGD-BLS + cycle context`

### 7.5 XJTU Batch-1

数据路径：

`XJTU battery dataset/Batch-1`

协议同 Batch-5，但工况是 2C 恒流放电：

- 电池：`2C_battery-1` 到 `2C_battery-8`
- leave-one-battery-out
- 每 0.5 分钟重采样
- 训练 30 条 cycle
- 训练长度 60
- 测试完整 cycle

Batch-1 是 current-only SAGD-BLS 的失败案例，也是引入 `cycle context` 的主要动机。

## 8. BattNN 对照设置

### 8.1 SimData BattNN

BattNN 仓库原始记录：

`BattNN/results/batch size and seq len/Simdata_BattNN_batch_size.txt`

batch size 30、seq len 60 的原始记录均值约：

`MAE = 0.020243`

本地复现：

`MAE = 0.020713 ± 0.000982`

两者接近，因此 SimData BattNN 复现可信度高。

### 8.2 Lab/NASA BattNN

使用原 BattNN 模型与处理后数据集，本地重新运行。

结果文件：

`results/battnn_lab_nasa_reproduced.csv`

LabData 因本地数据不完整和一次未收敛，报告中要说明限制。

### 8.3 XJTU BattNN Tuned Adapter

XJTU 不是 BattNN 原始论文 benchmark，因此这里构造同协议 adapter。

为增强公平性，补做 compact tuning：

物理参数 profile：

- Lab profile
- NASA profile
- Sim profile
- paper default profile

学习率：

- `0.01`
- `0.02`

固定：

- `weight_decay = 5e-4`
- `n_train = 30`
- `train_length = 60`
- search epochs = 800
- final epochs = 2000
- final seeds = 5

验证策略：

- 每个 batch 先用一个验证电池选择最优 profile 和学习率；
- 选出最优配置后，再做完整 leave-one-battery-out 五次实验。

调参结果：

| 数据集 | 最优 profile | 最优 lr | tuned BattNN MAE |
| --- | --- | ---: | ---: |
| XJTU Batch-5 | Lab | 0.02 | 0.104974 ± 0.002318 |
| XJTU Batch-1 | Lab | 0.01 | 0.161970 ± 0.012103 |

结论：

调参后 BattNN adapter 只小幅改善，仍未接近 SAGD-BLS。

## 9. 当前结果汇总

### 9.1 总体结果

| 数据集 | BattNN | SAGD-BLS | MAE 降低 |
| --- | ---: | ---: | ---: |
| SimData | 0.020713 ± 0.000982 | 0.005621 ± 0.001605 | 72.9% |
| LabData local B1-B6 | 0.039661 ± 0.004084 | 0.023488 ± 0.004999 | 40.8% |
| NASAData | 0.039656 ± 0.002477 | 0.029400 ± 0.010949 | 25.9% |
| XJTU Batch-5 | 0.104974 ± 0.002318 | 0.026911 ± 0.002916 | 74.4% |
| XJTU Batch-1 | 0.161970 ± 0.012103 | 0.106851 ± 0.056163 | 34.0% |

### 9.2 XJTU Context 消融

| 数据集 | SAGD-BLS 变体 | MAE | 相对 tuned BattNN 的 MAE 降低 |
| --- | --- | ---: | ---: |
| Batch-5 | current-only | 0.035881 | 65.9% |
| Batch-5 | cycle context | 0.026911 | 74.4% |
| Batch-5 | cycle + early-voltage context | 0.028964 | 72.4% |
| Batch-1 | current-only | 0.195916 | -21.0% |
| Batch-1 | cycle context | 0.106851 | 34.0% |
| Batch-1 | cycle + early-voltage context | 0.127662 | 21.2% |

解释：

- Batch-5 的随机工况电流动态信息丰富，current-only 已经较强，cycle context 进一步提升。
- Batch-1 是恒流工况，仅靠电流形状无法区分跨电池老化状态，current-only 失败。
- `cycle context` 给 BLS 输入空间提供低维老化/容量上下文，使 Batch-1 反超 BattNN tuned adapter。
- `cycle_voltage` 没有超过 `cycle context`，因此不作为主方法。

## 10. 代码文件

核心模型：

`sagd_bls_battery.py`

主要接口：

- `SAGDBLS.fit(currents, voltages, train_length=60, contexts=None)`
- `SAGDBLS.predict(current_sequence, context=None)`
- `SAGDBLS.evaluate(test_iter)`

实验脚本：

- `run_sagd_bls_sim.py`
- `run_sagd_bls_battnn_datasets.py`
- `run_sagd_bls_xjtu.py`
- `rerun_battnn_sim.py`
- `rerun_battnn_battnn_datasets.py`
- `rerun_battnn_xjtu.py`
- `tune_battnn_xjtu.py`
- `audit_simdata_fairness.py`
- `make_research_artifacts.py`

主要结果目录：

`results/`

## 11. 复现实验命令

### 11.1 SAGD-BLS SimData

```powershell
.venv\Scripts\python.exe run_sagd_bls_sim.py --seeds 1 2 3 4 5 --activation tanh sigmoid --n-map 100 --n-enhance 100
```

### 11.2 SimData BattNN-style 抽样 audit

```powershell
.venv\Scripts\python.exe audit_simdata_fairness.py
```

### 11.3 SAGD-BLS XJTU Batch-5

```powershell
.venv\Scripts\python.exe run_sagd_bls_xjtu.py --batch Batch-5 --results-csv results\sagd_bls_xjtu_batch5_v2_cycle.csv --seeds 1 2 3 4 5 --activation tanh sigmoid --n-map 100 --n-enhance 100 --epochs 20000 --resample-minutes 0.5 --train-length 60 --n-train 30 --context cycle
```

### 11.4 SAGD-BLS XJTU Batch-1

```powershell
.venv\Scripts\python.exe run_sagd_bls_xjtu.py --batch Batch-1 --results-csv results\sagd_bls_xjtu_batch1_v2_cycle.csv --seeds 1 2 3 4 5 --activation tanh sigmoid --n-map 100 --n-enhance 100 --epochs 20000 --resample-minutes 0.5 --train-length 60 --n-train 30 --context cycle
```

### 11.5 BattNN XJTU compact tuning

Batch-5：

```powershell
.venv\Scripts\python.exe tune_battnn_xjtu.py --batches Batch-5 --profiles lab nasa sim paper --lrs 0.01 0.02 --weight-decays 0.0005 --search-experiments 1 --final-experiments 3 --epochs-search 800 --epochs-final 2000 --validation-count 1 --patience 60
```

Batch-1：

```powershell
.venv\Scripts\python.exe tune_battnn_xjtu.py --batches Batch-1 --profiles lab nasa sim paper --lrs 0.01 0.02 --weight-decays 0.0005 --search-experiments 1 --final-experiments 3 --epochs-search 800 --epochs-final 2000 --validation-count 1 --patience 60
```

最终五种子重跑使用单候选：

```powershell
.venv\Scripts\python.exe tune_battnn_xjtu.py --batches Batch-5 --profiles lab --lrs 0.02 --weight-decays 0.0005 --search-experiments 1 --final-experiments 5 --epochs-search 800 --epochs-final 2000 --validation-count 1 --patience 60
```

```powershell
.venv\Scripts\python.exe tune_battnn_xjtu.py --batches Batch-1 --profiles lab --lrs 0.01 --weight-decays 0.0005 --search-experiments 1 --final-experiments 5 --epochs-search 800 --epochs-final 2000 --validation-count 1 --patience 60
```

### 11.6 生成汇总表和图

```powershell
.venv\Scripts\python.exe make_research_artifacts.py
```

## 12. 结果文件清单

核心结果：

- `results/overall_battnn_vs_sagd_summary.md`
- `results/overall_battnn_vs_sagd_summary.csv`
- `results/sagd_bls_sim_results.csv`
- `results/sagd_bls_sim_battnn_sampling_audit.csv`
- `results/battnn_sim_reproduced.csv`
- `results/sagd_bls_battnn_datasets.csv`
- `results/battnn_lab_nasa_reproduced.csv`
- `results/battnn_vs_sagd_lab_nasa_summary.csv`
- `results/sagd_bls_xjtu_batch5_v2_cycle.csv`
- `results/sagd_bls_xjtu_batch1_v2_cycle.csv`
- `results/battnn_xjtu_batch5_tuned.csv`
- `results/battnn_xjtu_batch1_tuned.csv`
- `results/battnn_xjtu_tuned_summary.csv`
- `results/battnn_xjtu_compact_tuning_search_summary.csv`
- `results/xjtu_context_variant_summary.csv`
- `results/xjtu_batch5_battnn_vs_sagd_summary.csv`
- `results/xjtu_batch1_battnn_vs_sagd_summary.csv`

代表性曲线图：

- `results/figures/sagd_sim_representative.png`
- `results/figures/sagd_nasa_rw3_representative.png`
- `results/figures/sagd_xjtu_batch5_rw1_representative.png`
- `results/figures/sagd_xjtu_batch1_2c1_representative.png`

## 13. 可信度分级

### 高可信度

SimData。

理由：

- BattNN 本地复现与仓库原始记录接近；
- SAGD-BLS 在固定划分和 BattNN-style 抽样 audit 中都优于 BattNN；
- 数据协议与 BattNN 原始设置最接近。

推荐表述：

“在 BattNN SimData benchmark 上，SAGD-BLS 明显优于 BattNN，本地复现和抽样 audit 均支持该结论。”

### 中等可信度

NASAData。

理由：

- BattNN 本地复现接近仓库记录；
- 数据来自 BattNN 处理后数据；
- SAGD-BLS 有优势，但标准差较大。

推荐表述：

“NASAData 上 SAGD-BLS 取得支持性优势，但仍需更多种子或更严格统计检验。”

### 支持性结果

LabData。

理由：

- 本地缺少 B7/B8；
- 有一次 BattNN run 未收敛；
- 当前使用干净均值报告。

推荐表述：

“LabData 本地 B1-B6 上 SAGD-BLS 优于干净 BattNN 复现均值，但该结果受本地数据完整性限制。”

### 新增工业测试

XJTU Batch-1 和 Batch-5。

理由：

- XJTU 不是 BattNN 论文原始 benchmark；
- 这里是 raw `.mat` 抽取后的同协议 adapter 对比；
- 已补做 compact tuning，但还不是 exhaustive tuning。

推荐表述：

“在当前 XJTU raw `.mat` 抽取协议下，SAGD-BLS-v2-cycle 优于经过 compact tuning 的 BattNN adapter。”

## 14. 当前局限

方法局限：

- XJTU `cycle context` 依赖循环序号和完整循环时长；如果未来做在线早期预测，需要重新设定可用输入。
- `cycle context` 对电池老化有帮助，但也意味着方法不仅使用电流波形，还使用循环级元信息。
- 当前没有在 PDE 问题上测试，因此还不能声称已证明 PDE 求解能力。
- 当前没有进行统计显著性检验，只报告均值和标准差。

对照局限：

- XJTU BattNN adapter 已 compact tuning，但还不是完全穷尽调参。
- LabData 本地数据不完整。
- BattNN 的原始设计带有物理递推归纳偏置，迁移到 XJTU raw `.mat` 协议时可能需要更深入改造。

实现局限：

- SAGD-BLS 目前是 NumPy 实现，未做 GPU 加速。
- Adam epoch 固定 20000，虽然当前运行可接受，但后续可以加入早停。
- 当前结果依赖固定的数据抽取逻辑，XJTU 抽取规则变化会影响结果。

## 15. 后续建议

短期建议：

- 暂时不要继续改 SAGD-BLS 结构，先把当前 `SAGD-BLS-v2-cycle` 固定为阶段版本。
- 基于本记录整理一页方法图和一页实验总表。
- 把 SimData 作为主结果，XJTU 作为新增工业测试，Lab/NASA 作为支持性结果。

中期建议：

- 如果导师关注公平性，可继续扩大 BattNN XJTU 调参空间，包括 `weight_decay`、`x0`、`Rp/Rs/Csp/Cs` 的局部搜索，以及 per-split inner validation。
- 做 SAGD-BLS 自身的超参数敏感性分析，例如 `n_map/n_enhance/l2/activation/context`。
- 增加统计检验，例如 paired test 或 bootstrap confidence interval。

长期建议：

- 将同一个 SAGD-BLS 机制迁移到一个 PDE toy problem。
- PDE 阶段仍应保持 BLS 变体属性：固定宽度特征 + 单线性输出层 + 梯度下降训练输出权重。
- 电池任务作为工业测试，不替代 PDE 任务，只作为方法有效性的应用支撑。

## 16. 当前冻结口径

推荐对外汇报口径：

“我们提出了 SAGD-BLS，即一种状态增强梯度下降宽度学习系统。它保留 BLS 的固定随机映射层、固定随机增强层和单线性输出层，只将传统最小二乘输出权重求解替换为 Adam 梯度下降，以支持加权 SmoothL1 和显式正则。在 BattNN 单循环电池预测实验中，SAGD-BLS 在 SimData 上显著优于 BattNN 本地复现；在 NASA/Lab 支持实验和 XJTU raw `.mat` 同协议 adapter 测试中也取得优势。XJTU 上经过 compact tuning 的 BattNN adapter 仍未追上 SAGD-BLS-v2-cycle。当前结论限定在电池单循环工业测试，PDE 求解将在下一阶段验证。”
