# 改进 Attention-PINN 模型架构 Spec

## Why

当前 `DualEncoderFNODecoder` 存在五处结构性短板：注意力完全没有位置信息且 K/V 被 16 倍池化抹掉细节；谱卷积占全模型 96.3% 参数且随 modes² 增长；物理损失不覆盖 Dirichlet 边界与源极等位约束，且 norm_factor=20000 将物理/梯度损失数值压低约 4 个数量级（PINN 项实际近乎失效）；差分后掩码在填充边缘引入伪梯度；batch=2 下 BatchNorm 统计失真。

## 用户六项思路的裁决

| # | 用户思路 | 裁决 | 理由与替代 |
|---|---|---|---|
| 1 | RoPE 频率可学习 | 采纳并加固 | 正确洞见：固定频率的尺度谱不适配 EMF 多尺度场。加固：log-频率参数化 + 几何级数初始化防频率塌缩。注意力中的位置信息由本项独立承担 |
| 2 | 从源出发的因果（水波纹）掩码 | 否决，且不加替代 | 静电场是**椭圆**方程（Laplace），解全局依赖所有边界，硬因果序阻断远端信息流，物理上错误；水波纹直觉只适用波动（双曲）方程。（原拟的 ALiBi/距离通道/K-V 屏蔽三件套经用户评审因复杂度剔除，本轮不实现） |
| 3 | FNO 谱卷积参数过大 | 重构而非单纯缩减 | 4.36M 绝对量不算大，真正的病是 dense modes² 的参数增长方式。改为轴向 1D 谱卷积 ×2：参数 ~8× 下降的同时 modes 16→32，频率覆盖翻倍 |
| 4 | BC 不覆盖、源极无约束 | 采纳 | loss_bc（边界/源极像素上加权数据项）+ loss_eq（源极等位 ‖∇φ‖²）。附带发现并修正 norm_factor 数值问题 |
| 5 | 有限差分 + 全图差分后掩码 | 采纳（修正方案：掩码先行） | 对离散算子网络，有限差分是自洽选择；谱导数需周期性假设，被零填充违反。核心修正是"先腐蚀掩码、再取内点差分" |
| 6 | batch=2 下 BatchNorm 噪声 | 采纳 | GroupNorm 替换 + 可选梯度累积 |

## What Changes

- `CrossAttentionBlock` 重写：新增可学习频率 2D 轴向 RoPE、pre-norm GroupNorm(1)；num_heads 4→2（head_dim 8→16，为 RoPE 提供 4 个频率对/轴）
- `SpectralConv2d`（dense）替换为 `AxialSpectralConv`（y 轴 1D 谱卷积 → x 轴 1D 谱卷积，各带通道混合），modes 16→32（CLI 可调至 64）
- 源极/几何编码器中全部 `BatchNorm2d` → `GroupNorm(8)`
- 新增损失：`loss_bc`（边界∪源极 Dirichlet 软约束）、`loss_eq`（源极区 ‖∇φ_pred‖²）；`loss_p`/`loss_g` 改用腐蚀内点掩码并在物理单位下计算
- `dataset.py`：norm_factor 改为按数据实测最大值设定；数据路径 CLI 化（输入仍为 3 通道 one-hot，不变）
- `train.py`：总损失 = loss_d + λ_phy·loss_p + λ_grad·loss_g + λ_bc·loss_bc + λ_eq·loss_eq，全部系数 CLI 可调；五路损失分项日志；可选梯度累积
- **BREAKING**：旧 checkpoint `dual_encoder_fno_model.pth` 不再兼容（保留原文件不删除）；新训练写入 `dual_encoder_fno_model_v2.pth`

## Impact

- Affected specs: 无先序 spec（本目录首个）
- Affected code: `Attentiion_PINN/model.py`（注意力块、谱卷积、归一化）、`Attentiion_PINN/dataset.py`（norm_factor、路径）、`Attentiion_PINN/train.py`（损失、CLI、日志）、`Attentiion_PINN/visualize.py`（新权重文件名）

## 数据依赖的参数决策（Task 1 实测后落定）

以下默认值需在数据验证后确认/修正，实测结果记录于 checklist：

| 参数 | 暂定默认 | 决策依据 |
|---|---|---|
| norm_factor | 实测全局电位最大值向上取整（疑 ~2100–4000，现值 20000 明显偏大） | 目标值落入 [0,1]，同时让 loss_p/loss_g 数值恢复有效量级 |
| λ_phy | 50（重定标后） | 训练初期各损失项量级匹配（checklist 校验 ≥1% 贡献） |
| λ_grad | 1 | 同上 |
| λ_bc | 5 | 边界/源极像素约为空气像素 5 倍权重 |
| λ_eq | 10 | 源极等位约束 |
| modes | 32（轴向分解后单块参数 ≈131k vs 原 dense 1.05M） | 频率覆盖翻倍且参数降 ~8× |

## ADDED Requirements

### Requirement: R1 可学习频率 2D 轴向 RoPE

注意力 block 中，Q（全分辨率坐标）与 K（池化后格心坐标，乘以 downsample_factor 还原到原像素坐标系）的每个 head 向量 SHALL 按 2D 轴向旋转位置编码旋转：head_dim 拆为 x 半轴与 y 半轴，各半轴内相邻维配对为复数，乘以 e^{i·pos·θ}。频率 θ = exp(λ) SHALL 为每 block 每轴独立可学习参数，初始化为几何级数（覆盖 ~4px 到 ~256px 空间尺度），训练中记录到 TensorBoard。

#### Scenario: 频率自适应
- **WHEN** 不同边界工况下场的空间尺度差异显著
- **THEN** s2g 与 g2s 两个 block 学出不同的频率谱（日志可查），而非共享固定频率
- **AND** 若观察到频率塌缩（多个 λ 收敛到同值），checklist 中的监控项触发，提示加正则（本轮不实现）

### Requirement: R2 轴向谱卷积

每个 FNO 块 SHALL 由 dense `SpectralConv2d` 替换为 `AxialSpectralConv`：先沿 y 轴、再沿 x 轴各做一次 1D 谱卷积（rfft → 模态截断复数乘 → irfft），两次均带完整 in-out 通道混合（中间的通道混合使复合算子不再是秩 1 的可分离核，表达力充分）。modes 默认 32（CLI 可调）。1×1 卷积 skip 与 GELU 结构保持。

#### Scenario: 参数量与频率覆盖
- **WHEN** 以 width=32, modes=32 实例化
- **THEN** 单块参数 ≈ 131k（原 dense modes=16 为 1.05M，约 8× 降幅），全模型总参数从 4.36M 降至 ≈ 0.7M，同时频率截断从 16 模态升至 32 模态

### Requirement: R3 边界条件与源极等位损失

系统 SHALL 新增两个物理约束项，且 loss_p/loss_g 在物理单位（乘回 norm_factor）下计算后按统一标尺折算：
- `loss_bc`：仅在源极(ch0)∪边界(ch2)像素上的 MSE
- `loss_eq`：源极区内一阶梯度平方均值 mean(‖∇φ_pred‖² · ch0_mask)，即导体内部 E=0（等位）约束——无需真值、无需连通域标记

#### Scenario: 物理项量级恢复
- **WHEN** 训练第一步在真实 batch 上计算五路损失
- **THEN** loss_p/loss_g/loss_bc/loss_eq 各自对总损失的贡献 ≥ 1%（日志验证），不再出现归一化导致的物理项近零

### Requirement: R4 内点腐蚀掩码

`calculate_physics_loss` 与 `calculate_gradient_loss` SHALL 先构造腐蚀掩码（5 点差分模板的中心像素，其上下左右 4 邻居均为有效区且属于被约束区域），再对差分结果施加腐蚀掩码；不再对全图差分后以原掩码二次相乘。

#### Scenario: 填充边缘伪梯度消除
- **WHEN** 对人工构造的"有效区外恒为零、内部线性"的玩具张量计算梯度损失
- **THEN** 腐蚀掩码版本损失为 0（或数值误差量级），原实现则产生非零伪梯度（单元验证）

### Requirement: R5 归一化替换与训练稳定性

源编码器 3 处、几何编码器 1 处 `BatchNorm2d` SHALL 全部替换为 `GroupNorm(8)`；`CrossAttentionBlock` 增加 pre-norm `GroupNorm(1)`（即逐像素通道 LayerNorm）。train.py SHALL 提供梯度累积开关（默认 1=关闭），累积时正确处理 AMP GradScaler 语义（多次 scale/backward，一次 step/update）。

#### Scenario: 小 batch 训练稳定
- **WHEN** batch=2 训练
- **THEN** 模型中不再含任何 BatchNorm 层（`isinstance` 扫描验证），损失曲线无 BatchNorm 统计噪声导致的锯齿

### Requirement: R6 训练管线参数化

train.py/visualize.py 的数据路径与全部超参（lr、batch_size、epochs、五个 λ、modes、width、grad_accum）SHALL 支持 CLI 参数，默认路径指向本仓库 `retangle_data/results_reduction_fixed_boundary`；五路损失分项写入 TensorBoard；新权重文件名 `dual_encoder_fno_model_v2.pth`。

#### Scenario: 沙箱可复现
- **WHEN** 在本沙箱以 CPU 模式运行 `python train.py --epochs 1 --limit-samples 4`
- **THEN** 全流程跑通无 Windows 路径依赖，损失分项日志齐全

## MODIFIED Requirements

### Requirement: 总损失构成（原 loss = loss_d + 0.5·loss_p + 0.1·loss_g）

总损失 SHALL 变为：`loss = loss_d + λ_phy·loss_p + λ_grad·loss_g + λ_bc·loss_bc + λ_eq·loss_eq`，其中 loss_d（空气区 masked MSE）保持不变，λ 默认值见"数据依赖的参数决策"表。

### Requirement: 归一化（原 norm_factor=20000）

norm_factor SHALL 改为数据实测全局最大电位的向上取整值（Task 1 落定），使归一化目标落入 [0,1] 并恢复物理损失量级。checkpoint 兼容性因此断裂，属预期。

## REMOVED Requirements

### Requirement: 因果式注意力掩码（用户提案，未实现即否决）

**Reason**: 静电场为椭圆方程，解在每一点依赖全部边界条件；硬因果序（水波纹扩散）物理上仅适用双曲/波动问题，且破坏并行性。
**Migration**: 无存量实现，无需迁移。原拟的 ALiBi 距离偏置、距离变换通道、K/V 填充屏蔽三件套替代方案，经用户评审以复杂度为由剔除，本轮不实现。

### Requirement: BatchNorm2d 的使用

**Reason**: batch=2 下批统计噪声大，验证集/训练分布漂移时评估不稳定。
**Migration**: GroupNorm(8) 参数量近似（2C vs 2C），直接替换。
