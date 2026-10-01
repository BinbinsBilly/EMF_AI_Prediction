# Tasks

> 2026-10-01 用户修订：① 抛弃 v1 基线复现（原 Task 6 / R5，已移入 spec REMOVED）；② 数据生成收敛为"脚本 + 一次性验证运行"（原 Task 7/8 合并），不产出完整数据集；③ 抛弃全量训练与学习曲线训练实验（推迟阶段三）。

- [x] Task 1: 环境重建与多工况递归数据加载（R1）
  - [x] 1.1 重装依赖：torch(CPU)、tensorboard、matplotlib（沙箱重置后 numpy/pandas/scipy 已装）
  - [x] 1.2 `dataset.py`：`_find_file_pairs` 支持递归扫描（data-path 为根目录时发现 `**/original_region_data_*.csv` 并配对），每对样本附 condition 元数据（父目录名）；返回结构兼容现有训练代码
  - [x] 1.3 回归验证：单工况目录路径行为不变；根目录模式实测样本数 ≈ 1314、工况数 ≈ 100；随机抽 3 对形状/掩码/EDT 正常（实测：1314 对/100 工况，跨模式 torch.equal 一致，train.py 冒烟通过）
- [x] Task 2: 训练/验证划分（R2）
  - [x] 2.1 实现划分函数：random 分层（按 condition 分层 80/20）与 group（工况整组 80/20）两模式 + seed
  - [x] 2.2 manifest 落盘（JSON：seed/mode/train/val 文件清单/condition 标注），train.py 新增 `--val-ratio --split-mode --split-seed --split-file`
  - [x] 2.3 验证（37/37 PASS）：同 seed 逐索引复现一致；group 模式零工况交集（并集=100）；random 每工况 val ≥1（1051/263，占比 20.02%）；每 epoch 验证循环已集成。附带修复：dataset.py 宽度对称填充（boundary21 的 50×50 样本跨工况 batch 必需，父代理已接受）
- [x] Task 3: 评估指标模块（R3）
  - [x] 3.1 新建 `eval_utils.py`：relative_l2、max_abs_error、bc_violation、laplace_residual、equipotential_residual（masked 口径与 train.py 十字腐蚀一致，erode_mask 独立实现待 Task 4 统一为单一来源）
  - [x] 3.2 新建 `eval.py`：加载权重对任意数据目录/划分出全套指标报告（CLI 冒烟通过，六指标 + JSON 报告）
  - [x] 3.3 toy 验证（27 项 PASS）：线性场 laplace_residual=1.2e-12、pred=target 时全部指标=0、常数场等位残差=0、全零 mask 无 NaN
- [x] Task 4: TensorBoard 训练可视化扩展（R4）
  - [x] 4.1 每 epoch 写验证指标标量：val/mse、val/rel_l2、val/max_err、val/bc_violation、val/laplace_residual、val/eq_residual（+val/loss，共 7 标量）
  - [x] 4.2 验证样本四联图（region/target/pred/error，--vis-samples K 个，`--vis-interval` 控制）+ 误差直方图（30 bins，add_histogram_raw）
  - [x] 4.3 `--run-name/--tag` + hparams（26 项超参含数据量/划分模式）记录
  - [x] 4.4 小规模冒烟（--limit-samples 8 --epochs 2，~5 分钟）：TB 实测含全部新标量（各 2 点）、四联图 2 样本×2 epoch、误差直方图、hparams 子 run；五路损失 + 16 RoPE 频率标量完好；erode_mask 已切换为 eval_utils 单一来源
- [x] Task 5: 现有数据真值场验证与裁决（R6）
  - [x] 5.1 新建 `verify_data.py`：独立 FD 求解器（5 点差分稀疏矩阵 + scipy sparse 直接/CG，Dirichlet 全封闭，相对残差 < 1e-10），对指定目录/全量对求解并与存储场对比
  - [x] 5.2 全量 1314 对交叉校验报告（每对：相对 L2、max 差、BC 精确、E=0、存储场自身 Laplace 残差）落盘 `verification_report.csv` + `verification_report.json` 摘要
  - [x] 5.3 依据 spec 阈值（相对 L2 < 1e-3）出具裁决：**FAIL**（rel L2 mean 0.128，1290/1314 超标；根因=存储场为未收敛 FD 迭代解）→ 新方案已回写 spec（gen_data.py 用 FD 精确解生成；现有数据重真值化推迟阶段三）
- [x] Task 6: 自动化数据生成脚本与一次性验证（R7 + R8，修订合并）
  - [x] 6.1 新建 `gen_data.py`：`--family rect|circle|polygon|blob`、`--conditions N`、`--variants M`、`--seed`、`--out /workspace/data` → 区域图（外框源极 + 内部 HV 电极 + 空气）→ FD 求解 → 导体写常数 → CSV 对（命名兼容 Task 1 递归加载）
  - [x] 6.2 落盘前自动校验：BC 精确、导体内 E=0、Laplace 残差 < tol（实测 max 0.015V）、二方法抽检（细网格 512 重解+池化降采样，实测 rel L2 ~5e-3 < 0.02）；失败拒写 + 非零退出
  - [x] 6.3 一次性验证运行：四族各 2 工况 × 2 变体 = 16 对全过校验（16s）；verify_data.py 反向校验闭环 PASS（rel L2 max 2.8e-8）；dataset 递归加载 16 对可读；blob 构造性不自交（星形域填充，扫描线抽检正常）；**未产出完整数据集**
- [x] Task 7: 数据量讨论文档（R9，修订：无训练实验）
  - [x] 7.1 proceedings 讨论文档（`proceedings/3. 数据量讨论与不规则区域拓展.md`，304 行/10 callouts/2 mermaid/3 公式）：每族 50–100 构型 × 5–15 变体建议、1314 对两维裁决（量近饱和/质需重真值化/族多样性空白）、不规则拓展分形课程、阶段三学习曲线实验设计（263/526/788/1051 子集 + 边际收益判据 + 可执行命令）
- [x] Task 8: 阶段二记录文档（R10，修订：无全量训练）
  - [x] 8.1 `proceedings/4. 阶段二记录.md` 完成（389 行/17 wikilinks/9 callouts/2 mermaid：验证裁决、管线与脚本说明（CLI 实测核对）、TensorBoard 使用、阶段三展望含 REMOVED 项迁移表）；附带修正 3 号文档中因并行执行过时的 --run-name warning（已改 [!tip] 更新说明）

# Task Dependencies

- Task 1 → Task 2/3/4（数据加载是地基）
- Task 5 仅依赖已装 scipy，可与 Task 1 并行；Task 6 依赖 Task 5 裁决
- Task 2、Task 3 依赖 Task 1，二者相互独立可并行
- Task 4 依赖 Task 2 + Task 3
- Task 7 依赖 Task 5（裁决结论）+ Task 6（生成能力佐证）
- Task 8 依赖全部
- 训练类工作仅剩 Task 4 的小规模冒烟；v2 全量训练、学习曲线实验、完整数据集生成均推迟阶段三
