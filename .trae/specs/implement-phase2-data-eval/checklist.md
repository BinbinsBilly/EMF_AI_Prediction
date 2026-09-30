# Checklist

## R1 多工况递归数据加载
- [ ] data-path 指向 `/workspace/retangle_data` 根目录时，实测发现 ~1314 对、覆盖 ~100 工况目录
- [ ] 每对样本携带 condition 元数据（父目录名）
- [ ] 单工况目录路径回归通过（旧默认路径行为不变，旧冒烟命令可跑）
- [ ] 随机抽 3 对：形状/mask/EDT 距离/目标归一化与阶段一口径一致

## R2 训练/验证划分
- [ ] random 模式：按 condition 分层，每工况 val ≥ 1
- [ ] group 模式：train/val 工况零交集（并集 = 全部工况）
- [ ] manifest JSON 落盘且含 seed/mode/文件清单/condition 标注
- [ ] 同 seed 重跑划分结果完全一致；`--split-file` 可复用已有划分
- [ ] 训练循环每 epoch 在 val 上评估并打印

## R3 评估指标模块
- [ ] eval_utils.py 提供 relative_l2 / max_abs_error / bc_violation / laplace_residual / equipotential_residual
- [ ] masked 口径与 train.py 一致（复用十字腐蚀 erode_mask，无量纲化 H/H² 同口径）
- [ ] toy 验证：线性场 laplace_residual=0；pred=target 时全部指标=0；常数场等位残差=0
- [ ] eval.py 可独立加载权重出报告（CLI 冒烟通过）

## R4 TensorBoard 训练可视化
- [ ] 每 epoch 写入 val/mse、val/rel_l2、val/max_err、val/bc_violation、val/laplace_residual、val/eq_residual 标量
- [ ] 验证样本四联图（region/target/pred/error）与误差直方图按 --vis-interval 出现
- [ ] --run-name/--tag + hparams 记录全部超参与数据量、划分模式
- [ ] 训练过程标量完整：五路损失 + 16 RoPE 频率（阶段一已有）不被破坏
- [ ] 短训练冒烟后 TensorBoard 中新标量与图像全部可见

## R5 v1 基线复现
- [ ] model_v1.py 加载旧 dual_encoder_fno_model.pth `load_state_dict(strict=True)` 无 missing/unexpected keys
- [ ] v1 零样本评估产出全套 R3 指标（同 val split、同口径）
- [ ] v1 vs v2 对比结果已记录（TensorBoard 标量或 proceedings 表格）

## R6 现有数据真值场验证与裁决
- [ ] 独立 FD 求解器：全封闭 Dirichlet，稀疏求解相对残差 < 1e-10
- [ ] 全量 1314 对交叉校验报告落盘（verification_report），含每对相对 L2 / max 差 / BC / E=0 / 存储场 Laplace 残差
- [ ] 裁决明确记录：PASS（相对 L2 < 1e-3 全对成立）或 FAIL（附新方案）
- [ ] 抽查结果与 spec 勘察一致（BC 精确、E=0 精确、残差 max 1–7.5V 量级）

## R7 自动化数据生成脚本
- [ ] gen_data.py 支持 --family rect|circle|polygon|blob 与 --conditions/--variants/--seed/--out
- [ ] 每对落盘前自动校验：BC 精确、导体内 E=0、Laplace 残差 < tol、二方法抽检（细网格降采样）一致
- [ ] 校验失败样本不落盘且脚本非零退出
- [ ] 输出布局 /workspace/data/<族>/<工况>/，文件命名兼容 R1 递归加载
- [ ] 单族小批量冒烟通过（每族 2 工况 × 2 变体）

## R8 新几何族数据集
- [ ] rect/circle/polygon/blob 各族生成并通过全部自动校验
- [ ] 各族校验报告产出；递归加载计数正确
- [ ] blob 族几何不自交（生成参数限幅 + 抽检可视化确认）

## R9 学习曲线与讨论文档
- [ ] 25/50/75/100% 四个子集训练完成（同 val、同超参、run 命名可区分）
- [ ] val 相对 L2 vs N 曲线写入 TensorBoard
- [ ] proceedings 讨论文档完成：每族样本量建议、1314 对是否足够的实证结论、不规则拓展路线

## R10 全量训练与总结
- [ ] v2 全量训练曲线完整（损失 + 验证指标 + RoPE 频率）
- [ ] v1/v2 并排对比在 TensorBoard 可见
- [ ] proceedings/3. 阶段二记录.md 完成（Obsidian 格式）
