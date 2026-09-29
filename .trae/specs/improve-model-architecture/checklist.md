# Checklist

## 数据验证（Task 1）
- [ ] 实测记录：区域图唯一值集合 = {1,2,3}，值 3（边界）出现位置已明确
- [ ] 实测记录：电位全局 min/max（跨子目录抽样 ≥30 对），norm_factor 已按实测落定（原 20000 偏大已证实并修正）
- [ ] 实测记录：外边界环电位是否恒 0；源极连通域数量分布（loss_eq 设计依据）

## R1 可学习 RoPE
- [ ] RoPE 频率为每 block 每轴独立 nn.Parameter，log 参数化，几何级数初始化（覆盖 ~4px–256px 尺度）
- [ ] Q 用全分辨率坐标、K 用池化格心坐标（×downsample_factor 还原像素系）
- [ ] 频率值训练中写入 TensorBoard 可监控
- [ ] s2g 与 g2s 两 block 频率参数相互独立

## R2 轴向谱卷积
- [ ] AxialSpectralConv = y 轴 1D 谱卷积 → x 轴 1D 谱卷积，均带通道混合，AMP 下无 ComplexHalf 错误
- [ ] modes=32 时全模型参数 ≈ 0.7M（±10%），单谱块 ≈ 131k
- [ ] 1×1 skip + GELU 块结构保持

## R3/R4 损失
- [ ] loss_bc 仅在源极∪边界像素计算；loss_eq = mean(‖∇φ_pred‖²·ch0)，不依赖真值
- [ ] loss_p/loss_g 在物理单位下计算；真实 batch 上四项（p/g/bc/eq）对总损失贡献各 ≥1%
- [ ] erode_mask 实现并通过玩具张量单元验证（原实现伪梯度 > 0，新实现 ≈ 0）
- [ ] "为何不用解析/谱导数"已在 spec 记录（周期性假设被零填充违反）

## R5 归一化与稳定性
- [ ] 模型中 nn.BatchNorm2d 实例数为 0（扫描验证）
- [ ] CrossAttentionBlock 含 pre-norm GroupNorm(1)
- [ ] 梯度累积开关默认关闭，开启时 GradScaler 语义正确（N 次 scale/backward，1 次 step/update）

## R6 管线与回归
- [ ] train.py/visualize.py 全超参 CLI 化，无 Windows 绝对路径残留
- [ ] 五路损失分项写入 TensorBoard
- [ ] CPU 冒烟：--epochs 1 --limit-samples 4 跑通；~20 迭代损失下降、无 NaN
- [ ] visualize.py 对新模型 forward 出图不崩溃
- [ ] 旧 dual_encoder_fno_model.pth 未被修改或删除；新训练写 dual_encoder_fno_model_v2.pth
- [ ] mask 语义回归：填充区输出仍被损失排除；输入仍为 3 通道 one-hot
