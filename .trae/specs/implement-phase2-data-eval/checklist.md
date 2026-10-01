# Checklist

> 2026-10-01 修订：移除 R5（v1 基线）、R8 完整数据集产出、R9 学习曲线训练、R10 全量训练——见 spec.md REMOVED Requirements。

## R1 多工况递归数据加载
- [x] data-path 指向 `/workspace/retangle_data` 根目录时，实测发现 ~1314 对、覆盖 ~100 工况目录
- [x] 每对样本携带 condition 元数据（父目录名）
- [x] 单工况目录路径回归通过（旧默认路径行为不变）
- [x] 随机抽 3 对：形状/mask/EDT 距离/目标归一化与阶段一口径一致

## R2 训练/验证划分
- [x] random 模式：按 condition 分层，每工况 val ≥ 1
- [x] group 模式：train/val 工况零交集（并集 = 全部工况）
- [x] manifest JSON 落盘且含 seed/mode/文件清单/condition 标注
- [x] 同 seed 重跑划分结果完全一致；`--split-file` 可复用已有划分
- [x] 训练循环每 epoch 在 val 上评估并打印

## R3 评估指标模块
- [x] eval_utils.py 提供 relative_l2 / max_abs_error / bc_violation / laplace_residual / equipotential_residual
- [x] masked 口径与 train.py 一致（复用十字腐蚀 erode_mask，无量纲化 H/H² 同口径）
- [x] toy 验证：线性场 laplace_residual=0；pred=target 时全部指标=0；常数场等位残差=0
- [x] eval.py 可独立加载权重出报告（CLI 冒烟通过）

## R4 TensorBoard 训练可视化
- [x] 每 epoch 写入 val/mse、val/rel_l2、val/max_err、val/bc_violation、val/laplace_residual、val/eq_residual 标量
- [x] 验证样本四联图（region/target/pred/error）与误差直方图按 --vis-interval 出现
- [x] --run-name/--tag + hparams 记录全部超参与数据量、划分模式
- [x] 训练过程标量完整：五路损失 + 16 RoPE 频率（阶段一已有）不被破坏
- [x] 小规模冒烟后 TensorBoard 中新标量与图像全部可见

## R6 现有数据真值场验证与裁决
- [x] 独立 FD 求解器：全封闭 Dirichlet，稀疏求解相对残差 < 1e-10
- [x] 全量 1314 对交叉校验报告落盘（verification_report.csv + json），含每对相对 L2 / max 差 / BC / E=0 / 存储场 Laplace 残差
- [x] 裁决明确记录：PASS（相对 L2 < 1e-3 全对成立）或 FAIL（附新方案）
- [x] 抽查结果与 spec 勘察一致（BC 精确、E=0 精确、残差 max 1–7.5V 量级）

## R7+R8 自动化数据生成脚本与一次性验证（修订合并）
- [x] gen_data.py 支持 --family rect|circle|polygon|blob 与 --conditions/--variants/--seed/--out
- [x] 每对落盘前自动校验：BC 精确、导体内 E=0、Laplace 残差 < tol、二方法抽检（细网格降采样）一致
- [x] 校验失败样本不落盘且脚本非零退出
- [x] 输出布局 /workspace/data/<族>/<工况>/，文件命名兼容递归加载
- [x] 一次性验证运行：每族 2 工况 × 2 变体全过校验、递归加载计数正确、blob 不自交抽检通过
- [x] 未产出完整数据集（仅验证批量；完整生成留阶段三）

## R9 数据量讨论文档（修订：无训练实验）
- [x] proceedings 讨论文档完成：每族样本量建议、1314 对是否足够的分析结论、不规则拓展路线
- [x] 阶段三学习曲线实验设计写入文档（25/50/75/100% 子集方案 + 边际收益判据）

## R10 阶段二记录文档（修订：无全量训练）
- [x] proceedings/3. 阶段二记录.md 完成（Obsidian 格式：验证裁决、管线与脚本说明、TensorBoard 用法、阶段三展望）

## 已移除范围（2026-10-01 用户裁决，验证时跳过）
- v1 基线复现与零样本对比（原 R5）— 移除
- 完整新几何族数据集产出 — 推迟阶段三
- 学习曲线 4 子集训练实验 — 推迟阶段三
- v2 全量训练 — 推迟阶段三
