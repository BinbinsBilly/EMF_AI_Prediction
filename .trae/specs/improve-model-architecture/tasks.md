# Tasks

- [ ] Task 1: 环境准备与数据验证（数据依赖参数落定）
  - [ ] 1.1 安装 CPU 依赖：torch(CPU wheel)、numpy、pandas、scipy、matplotlib、tensorboard
  - [ ] 1.2 编写临时脚本（放 /tmp）抽样 ≥30 对 CSV，实测：区域图唯一值集合与值 3 的空间位置；电位全局 min/max（跨子目录）；外边界环电位是否恒 0；源极连通域数量分布
  - [ ] 1.3 依据实测落定 spec 中"数据依赖的参数决策"表（norm_factor、各 λ 初值），回写 checklist 记录实测值
- [ ] Task 2: dataset.py 改造
  - [ ] 2.1 norm_factor 按 Task 1 实测值设定（构造参数默认值更新）
  - [ ] 2.2 验证：单样本 __getitem__ 形状仍为 (3,256,256)/(1,256,256)/(1,256,256)，目标值落入 [0,1]，填充区 one-hot 仍全零（回归确认）
- [ ] Task 3: model.py 注意力重写（R1 + R5 pre-norm）
  - [ ] 3.1 CrossAttentionBlock：num_heads=2，pre-norm GroupNorm(1)
  - [ ] 3.2 可学习 2D 轴向 RoPE：log-频率参数化、几何初始化（覆盖 ~4px–256px 尺度）、Q 全分辨率坐标 / K 池化格心坐标（×downsample_factor 还原像素系），频率写入日志接口
  - [ ] 3.3 验证：随机输入 forward 形状不变；RoPE 频率参数 requires_grad；两个 block 频率独立
- [ ] Task 4: model.py 谱卷积与归一化（R2 + R5）
  - [ ] 4.1 实现 AxialSpectralConv（y 后 x 两次 1D 谱卷积，各带通道混合，AMP 兼容沿用原 float32 策略）
  - [ ] 4.2 替换 4 个 SpectralConv2d，modes 默认 32；DualEncoderFNODecoder 签名保持 (modes, width)
  - [ ] 4.3 编码器 BatchNorm2d → GroupNorm(8)
  - [ ] 4.4 验证：全模型参数 ≈ 0.7M（±10%），nn.BatchNorm2d 实例数为 0，forward/backward 一次无 NaN
- [ ] Task 5: 损失与训练循环改造（R3 + R4 + R5 + R6）
  - [ ] 5.1 实现 erode_mask（min-pool 腐蚀），重写 calculate_physics_loss / calculate_gradient_loss 为"腐蚀内点 + 物理单位"版本；玩具张量单元验证伪梯度消除
  - [ ] 5.2 新增 loss_bc（源极∪边界 MSE）与 loss_eq（源极 ‖∇φ‖²）
  - [ ] 5.3 train.py：argparse（数据路径、五 λ、modes、batch、epochs、grad_accum、limit-samples），五路损失日志，梯度累积含 GradScaler 正确语义，新权重名 dual_encoder_fno_model_v2.pth
  - [ ] 5.4 验证：真实 batch 上五路损失均非零且 loss_p/g/bc/eq 对总损失贡献 ≥1%
- [ ] Task 6: visualize.py 适配与全链路冒烟（R6）
  - [ ] 6.1 适配新权重文件名（输入仍 3 通道，可视化逻辑不需改通道索引）
  - [ ] 6.2 CPU 冒烟：train.py --epochs 1 --limit-samples 4 跑通；~20 迭代损失下降无 NaN；visualize.py 对未训练新模型一次 forward 出图不崩溃
  - [ ] 6.3 确认旧 dual_encoder_fno_model.pth 未被改动

# Task Dependencies

- Task 1 → Task 2（norm_factor）、Task 5（λ 初值）依赖其实测结果
- Task 2、Task 3、Task 4 相互独立，可并行
- Task 5 依赖 Task 4（新模型）；损失函数本身不依赖 Task 2/3 可先行编写
- Task 6 依赖 Task 1–5 全部完成
