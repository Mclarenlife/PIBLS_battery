# SAGD-BLS 研究说明

详细冻结版记录见：`SAGD_BLS_method_freeze_record.md`。当前阶段以其中的 `SAGD-BLS-v2-cycle` 作为固定方法版本。

## 方法定位

SAGD-BLS 全称为 State-Augmented Gradient-Descent Broad Learning System，即“状态增强梯度下降宽度学习系统”。本阶段把它作为一个严格属于 BLS 范式的变体，用于 BattNN 单循环电池电压预测任务。

模型保留 BLS 的核心结构：

- 固定随机映射节点；
- 固定随机增强节点；
- 单个线性输出层；
- 不使用 RNN、CNN、MLP 堆叠、BattNN 电路递推模块或模型集成；
- 只优化输出层权重 `beta`。

相对原版 BLS，主要方法变化是：不用最小二乘或伪逆直接求解输出权重，而是用 Adam 梯度下降训练输出权重。这样可以引入加权 SmoothL1 损失和显式 L2 正则，同时仍然保持“只有单个线性输出层可训练”的 BLS 变体属性。

## 状态增强输入

SAGD-BLS 的主设置使用电流序列和少量循环级上下文构造 BLS 输入。预测时不把完整电压序列作为特征输入。

默认状态坐标包括：

- `current / 8`；
- `t / 60`；
- `cumsum(current) / 360`；
- `diff(current) / 6`；
- RC-like 指数滤波电流状态，衰减系数为 `[0.99, 0.98, 0.95, 0.90, 0.80, 0.65, 0.50]`。

这些坐标使模型可以对变长单循环进行外推预测。它们属于 BLS 输入空间的预处理/特征坐标扩展，不是额外的神经网络模块。

针对 XJTU 原始 `.mat` 中的恒流数据，v2 版本加入了第二层低维状态增强：

- `cycle / 500`；
- 放电持续时间分钟数除以 60。

这仍然只是 BLS 输入坐标扩展，没有新增可训练前端层，也没有引入 BattNN 的物理递推结构。

另外保留一个可选校准消融 `cycle_voltage`：它会使用同一循环最开始的少量电压点。由于这改变了预测协议，所以只作为消融实验单独报告。当前 XJTU 主结果采用更干净的 `cycle` context，不使用早期电压点。

## 基准设置

第一阶段的原始验收目标是 BattNN SimData：

- 训练样本数为 30；
- 训练长度为 60；
- 测试完整变长循环；
- 随机种子为 `[1, 2, 3, 4, 5]`；
- 主要验收指标为五次随机种子平均 MAE。

BattNN 的原始记录来自仓库文件：

`BattNN/results/batch size and seq len/Simdata_BattNN_batch_size.txt`

该记录中的 BattNN MAE 约为 `0.02024`。本地复现后的 BattNN SimData 平均 MAE 为 `0.020713`。

## 运行方式

SimData 主实验：

```powershell
python run_sagd_bls_sim.py --seeds 1 2 3 4 5 --activation sigmoid sigmoid --n-map 100 --n-enhance 100
```

脚本会生成：

`results/sagd_bls_sim_results.csv`

默认还会运行激活函数消融：

- sigmoid/sigmoid；
- tanh/sigmoid；
- sin/sigmoid；
- softplus/sigmoid。

如需只快速运行主配置，可加入：

```powershell
--skip-ablation
```

XJTU v2 主实验使用 `cycle` context，例如：

```powershell
python run_sagd_bls_xjtu.py --batch Batch-1 --context cycle --seeds 1 2 3 4 5 --activation tanh sigmoid
```

## 当前研究范围

本阶段仍然聚焦电池单循环工业测试，而不是 PDE benchmark。当前目标是先建立一个干净、单模型、严格属于 BLS 变体的方法，使其在 BattNN SimData、BattNN 处理后数据集，以及原始 XJTU `.mat` 检查中超过 BattNN 风格基线。PDE 求解可以作为下一阶段再展开。

## 当前结果

BattNN 风格电池基准已经在本地完成运行。

| 数据集 | BattNN 基线 | SAGD-BLS | MAE 降低 |
| --- | ---: | ---: | ---: |
| SimData | 0.020713 本地复现 | 0.005621 | 72.9% |
| LabData | 0.039661 本地干净复现 | 0.023488 | 40.8% |
| NASAData | 0.039656 本地复现 | 0.029400 | 25.9% |
| XJTU Batch-5 | 0.104974 BattNN tuned adapter | 0.026911 | 74.4% |
| XJTU Batch-1 | 0.161970 BattNN tuned adapter | 0.106851 | 34.0% |

说明：

- SimData 的 SAGD-BLS 数值采用当前最好的激活函数组合 `tanh/sigmoid`，结果来自 `results/sagd_bls_sim_results.csv`。
- 本地 LabData 文件夹只有 B1-B6 `.npy` 文件；BattNN 原始记录包含 B1-B8。因此本地复现对比采用 B1-B6。
- 本地有一次 BattNN LabData 运行未收敛，MAE 高于 0.3。表中报告的是 MAE < 0.1 的干净 BattNN 均值；完整结果保存在 `results/battnn_lab_nasa_reproduced.csv`。
- NASAData 本地复现结果与 BattNN 记录较接近，因此可作为环境和复现实验稳定性的检查。
- XJTU Batch-5 使用原始 `.mat` 文件，提取负电流随机工况放电片段，将放电电流转为正值，每 0.5 分钟重采样，并在 RW_battery-1 到 RW_battery-8 上做 leave-one-battery-out 测试。这是新增的工业风格测试，不是 BattNN 论文原始 benchmark。BattNN 通过匹配 adapter 运行，并补做了物理参数 profile 与学习率调参。
- XJTU 表中采用 v2 `cycle` context。它修复了早期 current-only 特征图在 Batch-1 恒流数据上输给 BattNN 的问题。Batch-1 current-only MAE 为 `0.195916`；只加入 cycle 和 duration context 后降到 `0.106851`。
- 可选的 `cycle_voltage` 校准在当前实验中没有超过 `cycle` context，因此 `cycle` 是更干净的主版本。
- BattNN XJTU compact tuning 覆盖 Lab/NASA/Sim/论文默认四组物理 profile 和 `0.01/0.02` 两个学习率。Batch-5 最优为 Lab profile、`lr=0.02`；Batch-1 最优为 Lab profile、`lr=0.01`。调参后 BattNN 只小幅改善，仍低于 SAGD-BLS。

## XJTU Context 消融

| 数据集 | SAGD-BLS 变体 | MAE | 相对 BattNN 的 MAE 降低 |
| --- | --- | ---: | ---: |
| Batch-5 | current-only | 0.035881 | 65.9% |
| Batch-5 | cycle context | 0.026911 | 74.4% |
| Batch-5 | cycle + early-voltage context | 0.028964 | 72.4% |
| Batch-1 | current-only | 0.195916 | -21.0% |
| Batch-1 | cycle context | 0.106851 | 34.0% |
| Batch-1 | cycle + early-voltage context | 0.127662 | 21.2% |

这个消融说明：对于随机工况 Batch-5，电流动态本身已经提供较强信息；加入 cycle context 后还能进一步提升。对于恒流 Batch-1，仅靠电流形状几乎无法区分跨电池老化状态，因此 current-only 会失败；加入循环序号和放电时长后，BLS 输入空间获得了必要的低维老化/容量上下文，性能反超 BattNN adapter。

## 可信度与复现风险

当前“超过 BattNN 很多”的结论需要分层看待。

SimData 结论可信度最高。原因是 BattNN 的本地复现结果 `0.020713` 与仓库原始记录中 batch size 30、seq len 60 的五次结果均值约 `0.020243` 非常接近，说明 BattNN 环境和配置基本对齐。SAGD-BLS 固定训练划分下 MAE 为 `0.005621`。为了排除“固定训练集更有利”的可能，又补做了一个 audit：按 BattNN loader 的五次抽样语义重新运行 SAGD-BLS，平均 MAE 为 `0.009403 ± 0.006407`。这个数值不如固定划分漂亮，但仍明显低于 BattNN 的 `0.020` 左右。因此 SimData 上的优势大概率是真实存在的，不过建议报告时同时给出固定划分结果和 BattNN-style 抽样 audit 结果。

LabData 和 NASAData 可信度中等。NASAData 的 BattNN 本地复现与仓库记录接近，比较相对可靠；LabData 本地文件只有 B1-B6，而原始记录包含 B1-B8，并且本地有一次 BattNN run 未收敛。因此 LabData 可以作为支持性结果，但不应作为最核心论据。

XJTU Batch-1 和 Batch-5 仍应称为“同协议 adapter 对比”，不能直接称为超过 BattNN 原论文 XJTU 结果，因为 XJTU 原始 `.mat` 不是 BattNN 论文原始 benchmark。当前已补做一轮 BattNN adapter 调参：四组物理 profile、两个学习率、验证电池选择最优配置，再用五次种子做 leave-one-battery-out 测试。调参后 Batch-5 MAE 为 `0.104974`，Batch-1 MAE 为 `0.161970`，仍明显高于 SAGD-BLS 的 `0.026911` 和 `0.106851`。因此可以更稳妥地说：在当前 XJTU 抽取协议下，SAGD-BLS 优于经过 compact tuning 的 BattNN adapter。

还需要注意协议边界：SAGD-BLS 主版本不使用完整测试电压作为输入；XJTU v2 主结果使用 `cycle` context，即循环序号和放电时长。`cycle_voltage` 会使用早期电压点，因此只作为校准消融报告，不作为主版本。

## 为什么仍然是 BLS 变体

SAGD-BLS 没有把多个模型拼成框架，也没有引入可训练深层特征提取器。它的计算流程仍然是：

1. 对原始电流和少量上下文构造状态增强输入；
2. 输入经过固定随机映射节点；
3. 映射结果经过固定随机增强节点；
4. 拼接得到宽度特征；
5. 用一个线性输出层预测电压；
6. 只用梯度下降优化线性输出层权重。

因此，方法创新点集中在 BLS 的输入状态扩展、输出层训练目标和梯度下降求解方式上，而不是堆叠外部神经网络模块。

## 结果文件

- `results/sagd_bls_sim_results.csv`
- `results/sagd_bls_sim_battnn_sampling_audit.csv`
- `results/battnn_sim_reproduced.csv`
- `results/sagd_bls_battnn_datasets.csv`
- `results/battnn_lab_nasa_reproduced.csv`
- `results/battnn_vs_sagd_lab_nasa_summary.csv`
- `results/sagd_bls_xjtu_batch5.csv`
- `results/battnn_xjtu_batch5.csv`
- `results/battnn_xjtu_batch5_tuned.csv`
- `results/xjtu_batch5_battnn_vs_sagd_summary.csv`
- `results/sagd_bls_xjtu_batch1.csv`
- `results/sagd_bls_xjtu_batch1_v2_cycle.csv`
- `results/sagd_bls_xjtu_batch1_v2_cycle_voltage.csv`
- `results/battnn_xjtu_batch1.csv`
- `results/battnn_xjtu_batch1_tuned.csv`
- `results/battnn_xjtu_compact_tuning_search_summary.csv`
- `results/xjtu_batch1_battnn_vs_sagd_summary.csv`
- `results/sagd_bls_xjtu_batch5_v2_cycle.csv`
- `results/sagd_bls_xjtu_batch5_v2_cycle_voltage.csv`
- `results/xjtu_context_variant_summary.csv`
- `results/overall_battnn_vs_sagd_summary.csv`
- `results/overall_battnn_vs_sagd_summary.md`
- `results/figures/sagd_sim_representative.png`
- `results/figures/sagd_nasa_rw3_representative.png`
- `results/figures/sagd_xjtu_batch5_rw1_representative.png`
- `results/figures/sagd_xjtu_batch1_2c1_representative.png`

## 下一步

原始 XJTU Batch-5 和 Batch-1 检查已经完成。下一步建议：

- 固定 `cycle` context 作为 XJTU v2 主版本，`cycle_voltage` 仅作为消融保留；
- 如需更强审稿口径，可继续扩大 BattNN adapter 的搜索空间，例如加入更多 `weight_decay`、`x0` 和 per-split inner validation；
- 添加一个紧凑的方法流程图和可直接放入报告的实验表；
- 之后再决定是否把同一个 SAGD-BLS 机制扩展到 PDE toy problem。
