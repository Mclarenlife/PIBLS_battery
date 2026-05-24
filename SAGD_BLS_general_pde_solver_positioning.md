# SAGD-BLS 通用 PDE 求解器定位记录

记录日期：2026-05-25  
当前目标：把 SAGD-BLS 从“电池实验模型”重新定位为“偏通用的 PDE/算子求解 BLS 变体”，并把电池单循环预测作为工业验证任务，而不是方法本体。

## 1. 核心判断

当前电池实验结果很好，但如果直接把现有 `current/cumsum/diff/RC-state/cycle` 特征写成方法主体，会显得方法是为电池任务专门设计的。这不符合导师给出的原始研究方向：研究一个宽度学习 BLS 变体，用于 PDE 求解，再用电池单循环预测作为工业测试。

因此，后续研究叙述必须做一次抽象：

- **方法本体**：通用的梯度下降宽度学习求解器。
- **PDE 任务**：通过坐标输入、物理残差、边界/初值条件训练输出权重。
- **电池任务**：作为工业验证，把电流时序和循环上下文看作外部 forcing/history 输入，用监督损失验证该 BLS 变体的小样本泛化能力。

换句话说，电池实验不是 SAGD-BLS 的定义，而是 SAGD-BLS 的一个应用适配器。

## 2. 推荐方法名称

建议把研究主方法写成：

**GD-BLS Solver** 或 **SAGD-BLS Solver**

更完整的论文式名称：

**State-Augmented Gradient-Descent Broad Learning System for PDE and Operator Approximation**

中文：

**面向 PDE 与算子逼近的状态增强梯度下降宽度学习系统**

其中：

- `State-Augmented` 不专指电池状态，而是指对基础坐标、边界条件、物理参数、外部 forcing、历史状态等信息的输入坐标扩展；
- `Gradient-Descent` 表示输出权重不用伪逆/最小二乘，而通过梯度下降优化；
- `Broad Learning System` 表示固定随机宽度特征 + 单线性输出层的结构约束。

## 3. 通用方法本体

设通用输入为：

`z ∈ R^d`

在 PDE 中，`z` 可以是：

- 空间坐标 `x`
- 时间坐标 `t`
- PDE 参数 `μ`
- 材料系数或源项参数
- 边界标记或区域标记

在电池任务中，`z` 可以是：

- 时间坐标
- 当前电流
- 累积电流
- 电流差分
- 电流历史滤波状态
- 循环级上下文

通用 BLS 近似器为：

`u_hat(z) = beta^T Phi(z)`

其中 `Phi(z)` 是固定随机宽度特征：

`Phi(z) = [1, z_scaled, H_map(z), H_enhance(z)]`

映射层：

`H_map(z) = sigma_map(z_scaled W_map + b_map)`

增强层：

`H_enhance(z) = sigma_enhance(H_map W_enhance + b_enhance)`

训练参数只有：

`beta`

这就是需要固定的通用方法本体。

## 4. 为什么它适合 PDE

PDE 求解通常需要近似未知函数：

`u(x, t)`

或参数化算子：

`u(x, t; μ)`

SAGD-BLS 可以直接把 `u` 表示成固定随机基函数的线性组合：

`u_hat(x, t; μ) = beta^T Phi(x, t, μ)`

因为随机层固定，所以对坐标的导数可以转化为对固定特征的导数：

`∂u_hat/∂x = beta^T ∂Phi/∂x`

`∂²u_hat/∂x² = beta^T ∂²Phi/∂x²`

这样可以构造 PDE 残差：

`R(z) = N[u_hat](z) - f(z)`

训练目标可以写成：

`L = λ_r L_residual + λ_b L_boundary + λ_i L_initial + λ_d L_data + λ_reg ||beta||²`

其中：

- `L_residual`：内部 collocation 点 PDE 残差；
- `L_boundary`：边界条件损失；
- `L_initial`：初值条件损失；
- `L_data`：可选观测数据损失；
- `L_reg`：输出权重正则。

此时梯度下降仍然只更新 `beta`。这保留了 BLS 的结构边界，同时避免了伪逆无法自然处理 PDE 残差、非均匀权重、多项损失的问题。

## 5. PDE 适配器与电池适配器的关系

需要把系统分成三层：

### 5.1 通用 BLS 核心

负责：

- 初始化固定随机 mapping nodes；
- 初始化固定随机 enhancement nodes；
- 构造 `Phi(z)`；
- 标准化输入和设计矩阵；
- 用 Adam 优化输出权重 `beta`；
- 提供预测接口。

这一层不应该包含电池、电流、cycle、RC filter 等概念。

### 5.2 PDE 适配器

负责：

- 生成 collocation 点；
- 生成边界点和初值点；
- 定义 PDE residual；
- 计算或近似特征导数；
- 组合残差损失、边界损失、初值损失；
- 调用通用 BLS 核心训练 `beta`。

示例 PDE：

- Poisson 方程；
- Heat 方程；
- Burgers 方程；
- Allen-Cahn 方程；
- Helmholtz 方程。

第一阶段不需要全面覆盖各种 PDE，但至少应做一个 toy PDE 来证明“这不是纯电池模型”。

### 5.3 电池工业验证适配器

负责：

- 把电流序列转成状态增强输入；
- 把 cycle/duration 转成低维上下文；
- 使用监督电压损失训练；
- 跑 BattNN SimData、Lab/NASA、XJTU；
- 报告工业任务性能。

这一层可以保留当前效果好的电池特征，但要明确写成“任务适配器”，不是主方法本体。

## 6. 当前电池特征如何重新解释

当前电池特征不应被解释为“方法核心”，而应解释为：

**面向动态系统外部 forcing 的输入状态化策略。**

对应关系：

| 电池特征 | 通用解释 | PDE/算子中的类比 |
| --- | --- | --- |
| `current` | 外部 forcing 当前值 | 源项、输入载荷、边界驱动 |
| `t` | 坐标 | 时间坐标 |
| `cumsum(current)` | forcing 历史积分 | 累积通量、历史作用量 |
| `diff(current)` | forcing 局部变化率 | 输入梯度、载荷变化 |
| RC filtered states | 多尺度历史记忆 | 多尺度核特征、历史卷积近似 |
| `cycle` | 工况/老化参数 | 参数化 PDE 中的 `μ` |
| `duration` | 轨迹尺度信息 | 时间域长度或终止边界 |

这样写之后，电池实验就不是“专门手工调特征”，而是“通用状态增强思想在工业时序系统上的一个实例”。

## 7. 当前方法需要调整的叙述

不要这样写：

> 本文提出一种用于电池单循环预测的 SAGD-BLS，输入包括电流、累积电流、RC 滤波状态和循环序号。

应该这样写：

> 本文提出一种状态增强梯度下降宽度学习系统。该方法用固定随机宽度特征近似未知函数或算子，并用梯度下降训练单个线性输出层，从而可以统一处理监督数据损失与物理残差损失。对于 PDE，状态增强输入由坐标、参数、边界/初值信息构成；对于电池工业验证，状态增强输入由电流 forcing 及其多尺度历史坐标构成。

## 8. 推荐技术路线

### 阶段 A：冻结电池结果

当前已经完成。

冻结内容：

- `SAGD-BLS-v2-cycle`
- SimData 主结果；
- Lab/NASA 支持结果；
- XJTU Batch-1/Batch-5 tuned BattNN adapter 对比；
- 中文冻结记录。

### 阶段 B：抽离通用核心

建议新增：

- `sagd_bls_core.py`
- `pde_sagd_bls.py`

其中 `sagd_bls_core.py` 只包含通用 BLS 特征和 Adam 输出层训练，不出现 battery 字样。

电池代码后续可以改成：

- `battery_feature_adapter.py`
- `run_sagd_bls_battery_*.py`

### 阶段 C：做一个最小 PDE sanity check

不需要马上“测试各种 PDE”，但至少需要一个 PDE toy problem，否则“偏通用 PDE 求解器”的说法没有支点。

推荐第一个 PDE：

**1D Poisson 方程**

例如：

`-u''(x) = π² sin(πx), x ∈ [0, 1]`

边界：

`u(0) = 0, u(1) = 0`

解析解：

`u(x) = sin(πx)`

原因：

- 一维；
- 有解析解；
- 边界简单；
- 只需要二阶导；
- 很适合验证固定随机特征 + beta 梯度下降是否能做 PDE residual training。

### 阶段 D：再做一个时间 PDE 或非线性 PDE

可选：

- Heat equation；
- Burgers equation；
- Allen-Cahn equation。

这一步用于证明方法不是只会做静态 ODE/PDE。

### 阶段 E：电池作为工业验证

在论文/报告结构中，电池实验放在：

**Industrial Validation / Real-world Dynamic System Test**

而不是放在主方法定义里。

## 9. 建议论文或汇报结构

推荐结构：

1. 背景：PDE 求解和工业动态系统都需要小样本函数/算子逼近。
2. 原始 BLS 局限：伪逆输出层难以自然适配物理残差、多项损失和加权优化。
3. 方法：SAGD-BLS，用固定随机宽度特征 + Adam 输出层训练。
4. 通用公式：`u_hat(z)=beta^T Phi(z)`。
5. PDE 损失：残差、边界、初值、数据损失。
6. 电池适配器：把电流 forcing 状态化，作为工业验证任务。
7. 实验一：PDE toy sanity check。
8. 实验二：BattNN SimData。
9. 实验三：Lab/NASA/XJTU 工业测试。
10. 消融：激活函数、cycle context、BattNN tuning。
11. 局限与下一步。

## 10. 当前结果如何重新命名

当前文件 `SAGD_BLS_method_freeze_record.md` 记录的是：

**Battery industrial validation record**

而不是最终完整 PDE 方法记录。

建议后续新增：

`SAGD_BLS_pde_solver_record.md`

该文件记录 PDE 求解器公式、PDE toy problem、残差训练和边界处理。

## 11. 关键风险

最大风险：

如果继续只优化电池实验，方法会越来越像电池专用经验模型。

规避方式：

- 抽离通用 BLS 核心；
- 至少完成一个 PDE toy problem；
- 把电池特征写成 industrial adapter；
- 把 `cycle context` 写成参数化输入，而不是电池专有 trick；
- 不把 XJTU 结果作为“PDE 求解能力”的证据，只作为工业泛化能力证据。

## 12. 当前建议

当前不要再继续改电池特征。下一步应该是：

1. 固定 `SAGD-BLS-v2-cycle` 电池结果；
2. 抽象通用核心；
3. 实现一个最小 PDE residual training demo；
4. 再把电池实验作为应用验证接回主线。

这样研究逻辑会更稳：

**PDE/算子求解器是主体，电池实验是工业应用证明。**

## 13. 新方法脚本

已新增独立方法脚本：

`sagd_bls_pde_solver.py`

该脚本与当前电池实验脚本分离，不包含 `current/cycle/RC-state` 等电池专用概念。它的目标是研究通用 PDE 残差训练版 SAGD-BLS。

当前脚本包含：

- 通用 1D 固定随机宽度特征；
- mapping nodes；
- enhancement nodes；
- 单个线性输出层 `beta`；
- 对输入坐标的一阶和二阶解析导数；
- PDE residual loss；
- boundary loss；
- Adam 训练输出权重；
- 1D Poisson sanity check。

当前 Poisson demo：

`-u''(x) = pi^2 sin(pi x), x in [0, 1]`

`u(0)=u(1)=0`

解析解：

`u(x)=sin(pi x)`

运行命令：

```powershell
.venv\Scripts\python.exe sagd_bls_pde_solver.py --demo poisson --seed 1 --n-map 120 --n-enhance 120 --activation tanh tanh --epochs 80000 --lr 0.0003 --n-interior 128 --n-eval 1000
```

当前 tuned sanity check 结果：

| 指标 | 数值 |
| --- | ---: |
| MAE | 3.430333e-04 |
| RMSE | 3.822643e-04 |
| MAXAE | 6.040770e-04 |
| residual_RMSE | 7.901179e-03 |
| boundary_MAXAE | 6.040770e-04 |

结果文件：

`results/sagd_bls_pde_poisson_demo.csv`

这一步只证明新方法脚本已经从电池特征工程转向 PDE residual training。后续还需要继续做 Heat/Burgers 等时间或非线性 PDE，才能把“通用 PDE 求解器”的主张做扎实。

## 14. PIBLS/PINN Baseline 测试

已新增 baseline 比较脚本：

`run_pde_solver_baselines.py`

该脚本在同一个 1D Poisson 问题上比较三种方法：

- `SAGD-BLS-PDE`：当前通用 PDE 求解器研究版本，固定随机宽度特征，Adam 训练单个输出层 `beta`；
- `PIBLS-pinv`：与现有 `PIBLS.py` 思路一致的 1D physics-informed BLS baseline，用伪逆一次求解输出权重；
- `PINN`：tanh MLP，用 PyTorch autograd 计算二阶导，并用 Adam 训练网络参数。

正式五种子命令：

```powershell
.venv\Scripts\python.exe run_pde_solver_baselines.py --seeds 1 2 3 4 5 --sagd-epochs 80000 --sagd-lr 0.0003 --pinn-epochs 3000 --results-csv results\pde_poisson_baseline_results.csv --summary-csv results\pde_poisson_baseline_summary.csv
```

五种子结果：

| 方法 | MAE | RMSE | residual_RMSE | boundary_MAXAE | 平均运行时间 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SAGD-BLS-PDE | 8.307034e-05 ± 1.31e-04 | 9.352520e-05 ± 1.45e-04 | 9.584249e-03 ± 1.47e-03 | 1.486046e-04 ± 2.30e-04 | 2.158s |
| PIBLS-pinv | 3.926562e-04 ± 3.75e-04 | 4.624333e-04 ± 4.22e-04 | 1.762328e-02 ± 1.30e-02 | 8.259945e-04 ± 8.11e-04 | 0.037s |
| PINN | 2.601692e-04 ± 4.67e-04 | 2.665535e-04 ± 4.65e-04 | 1.731177e-02 ± 4.43e-03 | 2.442223e-04 ± 4.82e-04 | 10.041s |

结果文件：

- `results/pde_poisson_baseline_results.csv`
- `results/pde_poisson_baseline_summary.csv`

当前结论：

在这个最小 1D Poisson sanity check 上，调好学习率后的 SAGD-BLS-PDE 平均误差低于 PIBLS-pinv 和 3000 epoch PINN，并且比 PINN 更快。但这只是第一个 toy PDE，不能单独支撑“通用 PDE 求解器”结论。下一步应继续测试 Heat/Burgers 等时间或非线性 PDE。

## 15. Forced Burgers 非线性 PDE 测试

已新增 Burgers 实验脚本：

`run_burgers_experiment.py`

测试问题为带解析解的 forced Burgers：

`u_t + u u_x - nu u_xx = f(x,t), x in [-1,1], t in [0,1]`

解析解：

`u(x,t) = -sin(pi x) exp(-t)`

边界和初值：

- `u(-1,t)=0`
- `u(1,t)=0`
- `u(x,0)=-sin(pi x)`

粘性系数：

`nu = 0.01 / pi`

选择 forced Burgers 的原因：

- 保留 Burgers 的非线性项 `u u_x`；
- 有解析解，可直接评估 MAE/RMSE/MAXAE；
- 可以同时计算 PDE residual、初值误差和边界误差；
- 适合作为从线性 Poisson 走向非线性 PDE 的第一步。

比较方法：

- `SAGD-BLS-Burgers`：固定随机宽度特征，Adam 只训练输出 `beta`，直接优化非线性 Burgers residual + IC/BC；
- `PIBLS-linearized-pinv`：PIBLS 风格伪逆基线，只解线性化残差 `u_t - nu u_xx = f`，因为原始 pinv-PIBLS 对 `u u_x` 这种非线性 beta residual 不能直接一次伪逆求解；
- `PINN-Burgers`：tanh MLP，用 PyTorch autograd 训练完整非线性 residual。

运行命令：

```powershell
.venv\Scripts\python.exe run_burgers_experiment.py --seeds 1 2 3 --results-csv results\burgers_baseline_results.csv --summary-csv results\burgers_baseline_summary.csv
```

三种子结果：

| 方法 | MAE | RMSE | residual_RMSE | IC MAXAE | BC MAXAE | 平均运行时间 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SAGD-BLS-Burgers | 4.073106e-03 ± 9.14e-04 | 5.094749e-03 ± 1.15e-03 | 4.054669e-02 ± 4.36e-03 | 5.450117e-03 ± 1.30e-03 | 5.713737e-03 ± 1.68e-03 | 192.888s |
| PINN-Burgers | 1.044180e-02 ± 4.11e-03 | 1.290501e-02 ± 4.62e-03 | 5.371159e-02 ± 8.47e-03 | 2.214875e-02 ± 1.76e-02 | 2.141384e-02 ± 1.78e-02 | 77.869s |
| PIBLS-linearized-pinv | 2.687259e-01 ± 2.29e-04 | 3.243791e-01 ± 2.11e-04 | 1.162437e+00 ± 1.37e-03 | 1.877545e-04 ± 9.90e-05 | 3.976104e-04 ± 2.47e-04 | 0.971s |

结果文件：

- `results/burgers_baseline_results.csv`
- `results/burgers_baseline_summary.csv`

当前 Burgers 结论：

在这个 forced Burgers 非线性 PDE 上，SAGD-BLS-Burgers 的误差低于当前 PINN-Burgers baseline，并显著优于只能处理线性化残差的 PIBLS-pinv baseline。缺点是当前 NumPy 全量 Adam 实现较慢，平均运行时间高于 PINN。下一步可以从小批量 collocation、L-BFGS/Adam 混合优化、特征缩放和更稳的初始化入手优化效率。

## 16. Standard Unforced Burgers 单 seed 验证

为回答“当前 Burgers 难度是否太简单”的问题，新增标准无源 Burgers 实验脚本：

`run_standard_burgers_experiment.py`

测试问题：

`u_t + u u_x - nu u_xx = 0, x in [-1,1], t in [0,1]`

初值和边界：

- `u(x,0) = -sin(pi x)`
- `u(-1,t)=0`
- `u(1,t)=0`
- `nu = 0.01 / pi`

这个问题不再使用 manufactured source，因此没有直接解析解。当前脚本用有限体积/有限差分 method-of-lines 生成参考解：对流项使用保守型 Rusanov 数值通量，扩散项使用二阶中心差分，再用 `scipy.solve_ivp` 求解时间推进。这样避免了低粘性 Burgers 中裸中心差分对对流项不稳定的问题。

单 seed 运行命令：

```powershell
.venv\Scripts\python.exe run_standard_burgers_experiment.py --seeds 1 --skip-hard-sagd --results-csv results\standard_burgers_single_seed_results.csv --summary-csv results\standard_burgers_single_seed_summary.csv
```

单 seed 结果：

| 方法 | MAE | RMSE | MAXAE | residual_RMSE | IC MAXAE | BC MAXAE | 运行时间 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PINN-standard-Burgers | 1.174917e-01 | 2.197461e-01 | 9.252207e-01 | 5.477191e-01 | 1.852849e-02 | 1.500989e-02 | 78.960s |
| SAGD-BLS-standard-Burgers | 1.398185e-01 | 2.550852e-01 | 9.872398e-01 | 5.915243e-01 | 2.295649e-02 | 2.077347e-02 | 196.567s |
| PIBLS-linearized-pinv | 2.846471e-01 | 3.470212e-01 | 8.559041e-01 | 1.074628e+00 | 1.895792e-06 | 2.757466e-06 | 1.211s |

结果文件：

- `results/standard_burgers_single_seed_results.csv`
- `results/standard_burgers_single_seed_summary.csv`

当前结论：

标准无源 Burgers 明显比 forced Burgers 更难。forced Burgers 中，SAGD-BLS 在有解析制造解和源项约束的情形下优于 PINN；但在标准无源 Burgers 单 seed 上，当前通用 SAGD-BLS 版本暂时没有超过 PINN，MAE 为 `0.1398`，PINN 为 `0.1175`。三个方法的 MAXAE 都接近 1，说明误差主要来自低粘性 Burgers 的陡峭过渡区域，而不是简单边界拟合失败。

这个结果对研究路线是有价值的：它证明 forced Burgers 不能单独作为“通用 PDE 求解能力”的强证据，后续需要在标准无源 Burgers 上继续改进。优先方向包括：

1. 引入硬约束 trial solution，使初值和边界条件解析满足，只让 BLS 输出层学习内部自由函数。
2. 对 collocation 点做自适应或残差重采样，增加陡峭过渡区域的训练密度。
3. 尝试 Adam + L-BFGS 或分阶段学习率，改善非线性残差优化。
4. 保持 BLS 结构边界不变：固定随机宽度特征 + 单线性输出层 `beta`，只改变 PDE 适配和优化策略。

## 17. Hard-ICBC SAGD-BLS 改进记录

基于第 16 节的结果，已在 `run_standard_burgers_experiment.py` 中加入硬约束版本：

`SAGD-BLS-hard-ICBC`

核心 trial solution 为：

`u_hat(x,t) = (1-t)u0(x) + t(1-x^2)v_beta(x,t)`

其中：

- `u0(x)=-sin(pi x)`
- `v_beta(x,t)=beta^T Phi(x,t)`
- `Phi(x,t)` 仍然是固定随机 mapping nodes + enhancement nodes 的 BLS 宽度特征
- 训练参数仍然只有单个线性输出层 `beta`

这个形式解析满足：

- `u_hat(x,0)=u0(x)`
- `u_hat(-1,t)=0`
- `u_hat(1,t)=0`

因此初值和边界不再通过 soft penalty 拟合，而是由 trial solution 保证。训练目标只需要优化内部 Burgers residual 和 `beta` 正则项：

`R = u_t + u u_x - nu u_xx`

调参记录：

| 方法/配置 | MAE | RMSE | residual_RMSE | IC MAXAE | BC MAXAE | 运行时间 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PINN-standard-Burgers, 5000 epoch | 1.174917e-01 | 2.197461e-01 | 5.477191e-01 | 1.852849e-02 | 1.500989e-02 | 78.960s |
| SAGD-BLS-standard-Burgers, soft IC/BC, 40000 epoch | 1.398185e-01 | 2.550852e-01 | 5.915243e-01 | 2.295649e-02 | 2.077347e-02 | 196.567s |
| SAGD-BLS-hard-ICBC, 40000 epoch, lr=5e-4 | 1.135707e-01 | 2.237811e-01 | 5.617356e-01 | 0.000000e+00 | 1.224647e-16 | 534.594s |
| SAGD-BLS-hard-ICBC, 15000 epoch, lr=1.5e-3 | 1.171445e-01 | 2.264927e-01 | 5.668843e-01 | 0.000000e+00 | 1.224647e-16 | 199.903s |

结果文件：

- `results/standard_burgers_hard_icbc_single_seed_results.csv`
- `results/standard_burgers_hard_icbc_single_seed_summary.csv`
- `results/standard_burgers_hard_icbc_lr15e4_15k_results.csv`
- `results/standard_burgers_hard_icbc_lr15e4_15k_summary.csv`

推荐配置运行命令：

```powershell
.venv\Scripts\python.exe run_standard_burgers_experiment.py --seeds 1 --skip-soft-sagd --skip-pibls --skip-pinn --hard-trial decay --hard-sagd-epochs 15000 --hard-sagd-lr 0.0015 --results-csv results\standard_burgers_hard_icbc_lr15e4_15k_results.csv --summary-csv results\standard_burgers_hard_icbc_lr15e4_15k_summary.csv
```

当前结论：

Hard-ICBC trial solution 是有效改进。最高精度配置的 MAE 从 soft 版的 `0.1398` 降到 `0.1136`，已经低于当前 PINN 单 seed 的 `0.1175`；decay trial 推荐配置 `15000 epoch + lr=1.5e-3` 的 MAE 为 `0.1171`，也略低于 PINN，同时严格满足初值和边界条件。

需要谨慎表述的是：Hard-ICBC 版本虽然在 MAE 上略优于当前 PINN，但 RMSE 和 residual_RMSE 仍略高于 PINN，运行时间也更长。因此这不是“全面碾压 PINN”，而是证明通用 BLS-PDE 方向在标准无源 Burgers 难例上已经从落后推进到可竞争。下一步应优先优化效率和陡峭区域误差。

## 18. Stationary Base Trial 消融

为避免 hard trial 中的 `(1-t)u0(x)` 给 Burgers 演化施加过强的线性衰减先验，新增 `--hard-trial stationary` 选项。对应 trial solution 为：

`u_hat(x,t) = u0(x) + t(1-x^2)v_beta(x,t)`

它同样解析满足：

- `u_hat(x,0)=u0(x)`
- `u_hat(-1,t)=0`
- `u_hat(1,t)=0`

与 decay trial 相比，stationary trial 不再假设初始波形会随时间线性衰减，BLS 只需要学习 Burgers 演化相对初值的修正项。

运行命令：

```powershell
.venv\Scripts\python.exe run_standard_burgers_experiment.py --seeds 1 --skip-soft-sagd --skip-pibls --skip-pinn --hard-trial stationary --hard-sagd-epochs 15000 --hard-sagd-lr 0.0015 --results-csv results\standard_burgers_hard_icbc_stationary_15k_results.csv --summary-csv results\standard_burgers_hard_icbc_stationary_15k_summary.csv

.venv\Scripts\python.exe run_standard_burgers_experiment.py --seeds 1 --skip-soft-sagd --skip-pibls --skip-pinn --hard-trial stationary --hard-sagd-epochs 40000 --hard-sagd-lr 0.0005 --results-csv results\standard_burgers_hard_icbc_stationary_40k_results.csv --summary-csv results\standard_burgers_hard_icbc_stationary_40k_summary.csv
```

单 seed 消融结果：

| 配置 | MAE | RMSE | residual_RMSE | IC MAXAE | BC MAXAE | 运行时间 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decay, 15000 epoch, lr=1.5e-3 | 1.171445e-01 | 2.264927e-01 | 5.668843e-01 | 0.000000e+00 | 1.224647e-16 | 199.903s |
| stationary, 15000 epoch, lr=1.5e-3 | 1.170710e-01 | 2.265148e-01 | 5.668790e-01 | 0.000000e+00 | 1.224647e-16 | 197.753s |
| decay, 40000 epoch, lr=5e-4 | 1.135707e-01 | 2.237811e-01 | 5.617356e-01 | 0.000000e+00 | 1.224647e-16 | 534.594s |
| stationary, 40000 epoch, lr=5e-4 | 1.135364e-01 | 2.235604e-01 | 5.616979e-01 | 0.000000e+00 | 1.224647e-16 | 532.118s |

结果文件：

- `results/standard_burgers_hard_icbc_stationary_15k_results.csv`
- `results/standard_burgers_hard_icbc_stationary_15k_summary.csv`
- `results/standard_burgers_hard_icbc_stationary_40k_results.csv`
- `results/standard_burgers_hard_icbc_stationary_40k_summary.csv`

当前结论：

stationary base 的方向是正确的，但提升很小。15k 配置的 MAE 从 `0.1171445` 降到 `0.1170710`，40k 配置的 MAE 从 `0.1135707` 降到 `0.1135364`。这说明线性衰减先验不是当前主要瓶颈；主要瓶颈更可能来自固定随机宽度特征对低粘性 Burgers 陡峭过渡区的表达能力。

因此，后续改进不宜继续堆外部模块，而应限定在 BLS 变体内部：

1. 特征字典本体：宽度、激活函数、随机权重尺度、输入缩放。
2. 输出层优化：仍然只优化 `beta`，可比较 Adam、L-BFGS 或二者的 beta-only 阶段化优化。
3. 硬约束表达：保留 stationary base 作为默认 hard trial，因为它更自然且略优。
