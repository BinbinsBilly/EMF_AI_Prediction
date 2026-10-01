# 改进 Attention-PINN 模型架构 Spec

## Why

当前 `DualEncoderFNODecoder` 存在五处结构性短板：注意力完全没有位置信息且 K/V 被 16 倍池化抹掉细节；谱卷积占全模型 96.3% 参数且随 modes² 增长；物理损失不覆盖 Dirichlet 边界与源极等位约束；归一化与损失尺度不当（实测电位 max=60000 而 norm_factor=20000，目标越界至 3.0；且逐像素差分天然带 1/H 与 1/H² 尺度因子，使物理/梯度损失被压低数个数量级）；差分后掩码在填充边缘引入伪梯度；batch=2 下 BatchNorm 统计失真。

## 数据实测结论（Task 1，全量 1314 对扫描 + 抽样验证）

- 电位全局 max = **60000.0**（全量扫描），min = 0
- **边界(值3)电位恒为 60000V**（30 样本并集仅 {60000.0}），**源极(值1)电位恒为 0V**（并集仅 {0.0}）——即 Dirichlet BC：源极接地、边界为高电位极；源极区严格等位（极差<1V 全部成立）
- 区域图唯一值全集 {0,1,2,3}；值 0 罕见（约 2% 样本出现），按 one-hot 全零处理（与填充一致）
- 源极连通域 1–2 个/样本，像素数 112–4876

## 用户六项思路的裁决

| # | 用户思路 | 裁决 | 理由与替代 |
|---|---|---|---|
| 1 | RoPE 频率可学习 | 采纳并加固 | 正确洞见：固定频率的尺度谱不适配 EMF 多尺度场。加固：log-频率参数化 + 几何级数初始化防频率塌缩。注意力中的位置信息（含像素坐标与到源极/边界距离）由本项独立承担 |
| 2 | 从源出发的因果（水波纹）掩码 | 否决 | 静电场是**椭圆**方程（Laplace），解全局依赖所有边界，硬因果序阻断远端信息流，物理上错误；水波纹直觉只适用波动（双曲）方程。但其"近处比远处重要"的局部性动机有效，改由 R1 的 RoPE 增设到源极/到边界距离两个位置轴来承接（距离信息融入第 1 项），不再单设章节 |
| 3 | FNO 谱卷积参数过大 | 重构而非单纯缩减 | 4.36M 绝对量不算大，真正的病是 dense modes² 的参数增长方式。改为轴向 1D 谱卷积 ×2：参数 ~8× 下降的同时 modes 16→32，频率覆盖翻倍 |
| 4 | BC 不覆盖、源极无约束 | 采纳 | loss_bc（边界∪源极 Dirichlet 软约束）+ loss_eq（源极等位 ‖∇φ‖²，实测源极恒 0V 严格等位，依据充分）。附带修正归一化与损失尺度：norm_factor 20000→60000，物理/梯度损失改用无量纲尺度（×H 或 ×H²）恢复量级 |
| 5 | 有限差分 + 全图差分后掩码 | 采纳（修正方案：掩码先行） | 对离散算子网络，有限差分是自洽选择；谱导数需周期性假设，被零填充违反。核心修正是"先腐蚀掩码、再取内点差分" |
| 6 | batch=2 下 BatchNorm 噪声 | 采纳 | GroupNorm 替换 + 可选梯度累积 |

## What Changes

- `CrossAttentionBlock` 重写：新增可学习频率 4 轴 RoPE（位置轴 = y 像素坐标、x 像素坐标、到源极距离、到边界距离，共 4 轴）、pre-norm GroupNorm(1)；num_heads 4→2（head_dim 8→16，为 4 轴 RoPE 各分配 4 维=2 频率对/轴）
- `SpectralConv2d`（dense）替换为 `AxialSpectralConv`（y 轴 1D 谱卷积 → x 轴 1D 谱卷积，各带通道混合），modes 16→32（CLI 可调至 64）
- 源极/几何编码器中全部 `BatchNorm2d` → `GroupNorm(8)`
- 新增损失：`loss_bc`（边界∪源极 Dirichlet 软约束）、`loss_eq`（源极区 ‖∇φ_pred‖²）；`loss_p`/`loss_g` 改用腐蚀内点掩码并在物理单位下计算
- `dataset.py`：除 one-hot 输入(3,H,W) 外，额外输出距离位置辅助 (2,H,W)（到源极/到边界的 EDT，供 RoPE 使用，不进编码器通道）；norm_factor 改为按数据实测最大值设定；数据路径 CLI 化
- `train.py`：总损失 = loss_d + λ_phy·loss_p + λ_grad·loss_g + λ_bc·loss_bc + λ_eq·loss_eq，全部系数 CLI 可调；五路损失分项日志；可选梯度累积
- **BREAKING**：旧 checkpoint `dual_encoder_fno_model.pth` 不再兼容（保留原文件不删除）；新训练写入 `dual_encoder_fno_model_v2.pth`

## Impact

- Affected specs: 无先序 spec（本目录首个）
- Affected code: `Attentiion_PINN/model.py`（注意力块、谱卷积、归一化）、`Attentiion_PINN/dataset.py`（norm_factor、路径）、`Attentiion_PINN/train.py`（损失、CLI、日志）、`Attentiion_PINN/visualize.py`（新权重文件名）

## 数据依赖的参数决策（已按 Task 1 实测落定）

| 参数 | 落定值 | 决策依据 |
|---|---|---|
| norm_factor | **60000**（实测全局 max，原 20000 偏小使目标越界至 3.0） | 目标落入 [0,1]，边界极归一化后恰为 1.0 |
| λ_phy | 1 | 无量纲化后各损失项 O(1) 量级（checklist 校验贡献） |
| λ_grad | 0.1 | 同上 |
| λ_bc | 2 | 边界/源极像素加权数据项 |
| λ_eq | 5 | 源极等位约束 |
| modes | 32（轴向分解后单块参数 ≈131k vs 原 dense 1.05M） | 频率覆盖翻倍且参数降 ~8× |

**损失无量纲化**：物理/梯度损失不再乘回 norm_factor，而用固定参考尺度消除 1/H、1/H² 因子——loss_p = mean((Δ²φ·H²)²)（H=256 网格高），loss_g = mean(((∇φ-∇tgt)·H)²)，loss_eq = mean((∇φ·H)²·src)。该尺度与 norm_factor 无关，λ 语义稳定。

## ADDED Requirements

### Requirement: R1 可学习频率 4 轴 RoPE（融合距离）

注意力 block 中，Q（全分辨率坐标）与 K（池化后格心坐标，乘以 downsample_factor 还原到原像素坐标系）的每个 head 向量 SHALL 按 4 轴 RoPE 旋转。4 个位置轴为：y 像素坐标、x 像素坐标、到源极的 EDT 距离、到边界的 EDT 距离（后两者由 dataset 计算并传入，在填充区 zero-pad）。head_dim=16 拆为 4 轴，每轴 4 维（2 个频率对）；各轴内相邻维配对为复数，乘以 e^{i·pos·θ}。频率 θ = exp(λ) SHALL 为每 block 每轴独立可学习参数，初始化为几何级数（覆盖 ~4px 到 ~256px 空间尺度），训练中记录到 TensorBoard。

距离位置由 dataset 用 `scipy.ndimage.distance_transform_edt` 在未填充区域图上计算（到源极=到值 1 区域的距离，到边界=到值 3 区域的距离），以像素为单位（与 y/x 坐标轴同单位制，使 4 轴可共享频率初始化尺度）、zero-pad 后作为位置辅助张量传入 attention，不参与编码器卷积通道。

#### Scenario: 频率自适应与距离先验
- **WHEN** 不同边界工况下场的空间尺度差异显著
- **THEN** s2g 与 g2s 两个 block、4 个轴学出不同的频率谱（日志可查），而非共享固定频率
- **AND** 到源极/到边界两个距离轴的引入使注意力天然偏向"近源/近边界处场变化剧烈"的局部性先验，无需硬因果序或 ALiBi 偏置
- **AND** 若观察到频率塌缩（多个 λ 收敛到同值），checklist 中的监控项触发，提示加正则（本轮不实现）

### Requirement: R2 轴向谱卷积

每个 FNO 块 SHALL 由 dense `SpectralConv2d` 替换为 `AxialSpectralConv`：先沿 y 轴、再沿 x 轴各做一次 1D 谱卷积（rfft → 模态截断复数乘 → irfft），两次均带完整 in-out 通道混合（中间的通道混合使复合算子不再是秩 1 的可分离核，表达力充分）。modes 默认 32（CLI 可调）。1×1 卷积 skip 与 GELU 结构保持。

#### Scenario: 参数量与频率覆盖
- **WHEN** 以 width=32, modes=32 实例化
- **THEN** 单块参数 ≈ 131k（原 dense modes=16 为 1.05M，约 8× 降幅），全模型总参数从 4.36M 降至 ≈ 0.7M，同时频率截断从 16 模态升至 32 模态

### Requirement: R3 边界条件与源极等位损失

系统 SHALL 新增两个物理约束项，全部损失在归一化电位上计算、以固定参考尺度无量纲化（×H 或 ×H²，H=网格高）：
- `loss_bc`：仅在源极(ch0)∪边界(ch2)像素上的 MSE（实测两区电位恒定：源极 0、边界 60000/norm_factor=1.0，软约束依据充分）
- `loss_eq`：导体内点上的一阶梯度 mean((∇φ_pred·H)² · conductor_interior)，conductor = 源极∪边界（E=0 对任何导体成立，与电位值是否已知无关）；实测源极为 ≤2px 细条带（十字腐蚀内点=0）、边界为厚导体（内点≈1350），故该项在边界内点上生效、对细源极自然静默
- `loss_p`：mean((Δ²φ_pred·H²)² · 空气内点掩码)；`loss_g`：mean(((∇φ_pred−∇φ_tgt)·H)² · 有效内点掩码)

#### Scenario: 物理项量级恢复
- **WHEN** 训练初期在真实 batch 上计算五路损失
- **THEN** loss_g/loss_bc 对总损失的贡献 ≥ 1%（日志验证）；loss_p/loss_eq 在输出仍平坦时接近 0，随预测结构形成而上升，经 λ 加权后进入总损失——不再出现尺度因子导致的物理项恒近零

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

norm_factor SHALL 改为 60000（Task 1 全量实测全局 max，原值 20000 偏小导致归一化目标越界至 3.0），使目标落入 [0,1]。物理项量级恢复由损失无量纲化（R3）承担。checkpoint 兼容性因此断裂，属预期。

## REMOVED Requirements

### Requirement: 因果式注意力掩码（用户提案，未实现即否决）

**Reason**: 静电场为椭圆方程，解在每一点依赖全部边界条件；硬因果序（水波纹扩散）物理上仅适用双曲/波动问题，且破坏并行性。
**Migration**: 无存量实现，无需迁移。其"近处比远处重要"的局部性动机由 R1 的 RoPE 增设到源极/到边界距离两个位置轴承接（距离信息融入第 1 项）。

### Requirement: BatchNorm2d 的使用

**Reason**: batch=2 下批统计噪声大，验证集/训练分布漂移时评估不稳定。
**Migration**: GroupNorm(8) 参数量近似（2C vs 2C），直接替换。
