# 阶段三数据重生成 Spec：标注方案 v2 + 内部接地电极 + 四族正式生成

## Why

原始 `retangle_data`（1314 对）存在两个不可修复级别的缺陷：

1. **0 值歧义**：`0` 既表示零填充、又在 boundary21/_50 系列中表示 0V 接地电极，语义无法区分（原始数据设计缺陷，见 [[1. 模型基础实现]] 与 [[4. 阶段二记录]] 第二节）。
2. **真值不合格**：R6 全量交叉校验裁决 FAIL（未收敛迭代解，空气区 rel L2 mean 0.128）+ 632 对 region 错标。

用户裁决（2026-10-01）：**彻底抛弃 retangle_data，不再使用**；阶段二遗留的"重真值化 + region 错标修复"随之取消。阶段三改为：先重新设计标注根除 0 值歧义，再按 [[3. 数据量讨论与不规则区域拓展]] 的三层框架与精简档数量（50 构型 × 10 变体/族），用阶段二已验证的生成管线（[[4. 阶段二记录]] 3.5 节，反向闭环 rel L2 max 2.8e-8）正式生成四族完整数据集。

## What Changes

- **标注方案 v2 正式化**：region 图值域严格为 `{1=接地电极(0V), 2=空气, 3=高压电极(60000V)}`；`0` 严禁出现在任何落盘数据中，仅由 `dataset.py` 对称填充产生、由有效性掩码区分；外框接地环与内部接地电极统一标 `1`（物理语义相同：0V Dirichlet）。
- **gen_data.py 扩展内部接地电极**：每个内部电极以概率 `--p-ground`（默认 0.25）标为接地（1），其余为 HV（3）；保证每个布局 ≥1 个 HV 电极；外框 1px 环恒为 1。
- **工况设计分层覆盖**：每族 50 个工况的电极数按 `1 + (cond % N_max)` 分层轮转（rect 1–10、circle 1–6、polygon 1–4、blob 1–3），保证电极数量维均匀覆盖，替代纯均匀随机。
- **校验加固**：新增显式"落盘 region 不含 0 值"校验关；`variant_ok` 从仅查 HV 碎片扩展为两类电极（HV + 内部接地）碎片检查。
- **正式生成**：4 族 × 50 工况 × 10 变体 = **2000 对**，输出至 `data/{family}/`（替换阶段二 16 对验证批）；生成后以 `verify_data.py` 独立反向闭环校验。
- **dataset.py / model.py / train.py / verify_data.py 零改动**：one-hot 3 通道、求解器语义（任意 SRC 区域均为 0V Dirichlet）天然兼容新标注。

## Impact

- 受影响代码：`Attentiion_PINN/gen_data.py`（修改）
- 受影响数据：`data/`（2000 对新数据替换 16 对验证批）；`retangle_data/`（弃用、不删除、不再参与任何训练）
- 受影响规格：`implement-phase2-data-eval` 的"重真值化 + region 修复"阶段三迁移项取消
- 下游影响：阶段三后续实验（学习曲线、v2 全量训练）的数据基础变为新 2000 对；[[3. 数据量讨论与不规则区域拓展]] 第六节学习曲线矩阵的 rect 族基数由 1314 对调整为 500 对，命令中 `--data-path` 换为 `data/`

## ADDED Requirements

### Requirement: 标注方案 v2（根除 0 值歧义）

系统 SHALL 采用如下 region 图标注方案，并作为生成侧强制不变式：

| 值 | 语义 | 物理约束 |
|---|---|---|
| 0 | **仅填充**（dataset.py 对称填充专用） | 严禁出现在落盘 CSV |
| 1 | 接地电极（外框环 + 内部接地电极） | Dirichlet 0V，导体内 E=0 |
| 2 | 空气求解域 | Laplace 方程 ∇²φ=0 |
| 3 | 高压电极 | Dirichlet 60000V，导体内 E=0 |

#### Scenario: 生成数据不含 0 值

- **WHEN** gen_data.py 生成任意一对数据
- **THEN** 落盘 region 图的唯一值全集 ⊆ {1, 2, 3}；若出现 0 则该对拒写盘并记录失败

#### Scenario: 内部接地电极语义统一

- **WHEN** 布局中存在内部接地电极
- **THEN** 其标注值为 1，与外框接地环同值；求解器对其施加 0V Dirichlet；校验①（BC 精确 0V）与校验②（导体 E=0）对其同样生效

#### Scenario: dataset 侧兼容性

- **WHEN** dataset.py 加载新数据并做对称零填充
- **THEN** 填充区（0）与有效区（{1,2,3}）由有效性掩码严格区分；one-hot 3 通道编码不变；训练/评估管线零改动

### Requirement: 内部接地电极生成

gen_data.py SHALL 支持内部接地电极，且类型采样可复现：

- 每个内部电极独立以概率 `p_ground`（CLI `--p-ground`，默认 0.25）标为接地（1），否则为 HV（3）
- 若某布局采样结果全为接地（无 HV），强制将其中一个电极改为 HV
- 外框 1px 环恒为接地（1），不受采样影响
- 每个电极的类型（ground/hv）记录入 `gen_report.json` 的 electrodes 元数据

#### Scenario: 类型分布可复现

- **WHEN** 以相同 `--seed` 与 `--p-ground` 运行两次
- **THEN** 各工况电极类型分配逐位一致（沿用每工况独立 RNG 流）

#### Scenario: 至少一个 HV

- **WHEN** 任一布局（含全部变体裁剪后）
- **THEN** 至少存在 1 个 ≥20px 的 HV 连通域；否则整布局重采样（沿用 MAX_LAYOUT_TRIES=200）

#### Scenario: 不产生隔离空气域

- **WHEN** 内部接地电极加入布局
- **THEN** 校验③（n_invalid=0 且 isolated_air=0）对每对数据仍然成立；失败拒写盘

### Requirement: 工况分层设计

每族 50 个工况 SHALL 按电极数量分层轮转：`n_target = 1 + (cond % N_max)`（rect N_max=10、circle 6、polygon 4、blob 3），放置仍为最佳努力（放不下则少放，记录实际数量）；变体高度沿用既有规则（变体 0 = 256，其余 9 个自高度池 [246,…,166] 无放回抽取，variants=10 时恰好用尽高度池）。

#### Scenario: 电极数量维均匀覆盖

- **WHEN** 生成某族 50 工况
- **THEN** 每个目标电极数 1..N_max 被分配到的工况数 ∈ {⌊50/N_max⌋, ⌈50/N_max⌉}（gen_report.json 可核查）

### Requirement: 校验体系加固（六关）

在既有五关（① BC 精确 mem+roundtrip、② 导体 E=0、③ 求解残差 <1e-10、④ 存储 Laplace <0.05V、⑤ 细网格抽检 rel L2 <0.02）之上 SHALL 新增：

- **⑥ 标注合规**：落盘前断言 `set(unique(region)) ⊆ {1,2,3}`（显式独立校验关，不再仅是①的推论）
- **variant_ok 扩展**：裁剪后 HV 连通域与内部接地连通域均 ≥20px（外框环豁免），否则触发整布局重采样

#### Scenario: 验证批六关全过

- **WHEN** 以每族 2 工况 × 2 变体运行验证批
- **THEN** 16 对全部通过六关校验；`verify_data.py` 反向闭环 rel L2 < 1e-3（参照阶段二实测 2.8e-8 量级）

### Requirement: 正式批量生成与反向闭环

SHALL 生成 4 族 × 50 工况 × 10 变体 = 2000 对至 `data/{family}/{family}_c{NNN}/`，每族落盘 `gen_report.json`；生成完成后：

1. `verify_data.py --data-root data` 对全部 2000 对独立反向校验，通过率 100%（rel L2 < 1e-3）
2. `dataset.py` 根目录递归加载回归：发现 2000 对 / 200 工况（4 族 × 50），单工况路径加载行为不变

#### Scenario: 生成完成且可信

- **WHEN** 四族生成与反向校验完成
- **THEN** 2000/2000 对通过六关 + 反向闭环；报告落盘可查

## REMOVED Requirements

### Requirement: 1314 对重真值化 + region 错标修复（阶段二遗留阶段三事项 #1/#2）

**Reason**：用户裁决（2026-10-01）彻底抛弃 retangle_data，不再使用；修复失去意义。

**Migration**：rect 族由新生成的 500 对（50 工况 × 10 变体，新标注）替代；`retangle_data/` 目录保留不删除但不再参与任何训练/评估；[[3. 数据量讨论与不规则区域拓展]] 第六节学习曲线命令的 `--data-path` 相应改为 `data/`。
