# Tasks

- [x] Task 1: 环境准备与数据验证（数据依赖参数落定）
  - [x] 1.1 安装 CPU 依赖：torch 2.14.0+cpu、numpy 2.5.3、pandas 3.0.6、scipy 1.18.1、matplotlib、tensorboard
  - [x] 1.2 编写临时脚本（/tmp/data_validation.py）抽样+全量实测：唯一值全集 {0,1,2,3}（0 罕见 2%）；电位 max=60000（全量）；边界(3)恒 60000V、源极(1)恒 0V；源极等位成立、连通域 1–2 个
  - [x] 1.3 已落定：norm_factor=60000、λ_phy=1、λ_grad=0.1、λ_bc=2、λ_eq=5（物理损失无量纲化 ×H/×H²），回写 spec 与 checklist
- [x] Task 2: dataset.py 改造
  - [x] 2.1 norm_factor 按 Task 1 实测值设定（默认 60000.0）
  - [x] 2.2 新增距离位置辅助输出：_compute_pos_dist（EDT 到源极/边界，像素单位）+ _pad_pos_dist（与主填充一致），__getitem__ 返回 4 元组
  - [x] 2.3 验证通过（46 项）：形状/目标∈[0,1]/填充区 one-hot 全零/距离通道源极=0、边界=0/无 NaN
- [x] Task 3: model.py 注意力重写（R1 4 轴 RoPE + pre-norm）
  - [x] 3.1 CrossAttentionBlock：num_heads=2、PixelLayerNorm pre-norm、forward(x, context, pos_dist=None)
  - [x] 3.2 4 轴 RoPE：rope_log_freqs (4,2) log 参数化、波长 224/28px 初始化、Q 全分辨率 / K 格心 (idx+0.5)×16、K 距离轴池化、get_rope_frequencies() 日志接口
  - [x] 3.3 验证通过：形状不变、频率参数梯度非零、两 block 参数独立、(4,2) 频率 ≈ 2π/{224,28}
- [x] Task 4: model.py 谱卷积与归一化（R2 + R5）
  - [x] 4.1 AxialSpectralConv：y 后 x 两次 1D 谱卷积、均带通道混合、AMP 兼容
  - [x] 4.2 fno1~4 替换完成，modes 默认 32，签名保持
  - [x] 4.3 BatchNorm2d → GroupNorm(8) 共 4 处
  - [x] 4.4 验证通过：总参数 689,105（0.689M，谱块合计 524,288）、BatchNorm 实例=0、forward/backward 无 NaN、CPU autocast 正常
- [x] Task 5: 损失与训练循环改造（R3 + R4 + R5 + R6）
  - [x] 5.1 erode_mask 5 点十字模板（torch.minimum(垂直,水平) 实现，规避可分离复合=盒式陷阱），物理/梯度损失腐蚀内点+无量纲化；玩具验证：旧伪梯度 14.125、新 0.0
  - [x] 5.2 loss_bc（源极∪边界 MSE）与 loss_eq（导体=源极∪边界内点，实测源极细条带内点=0 自然静默、边界内点 1350 生效）
  - [x] 5.3 argparse 全参数化（含 --limit-samples/--grad-accum）、五路损失+16 个 RoPE 频率写入 TensorBoard、AMP 条件化（CPU 关闭）、梯度累积含尾批 flush、权重 dual_encoder_fno_model_v2.pth
  - [x] 5.4 冒烟通过：2 epochs×4 样本，五路损失有限无 NaN、总损失 29.1→19.1 下降、loss_eq>0、loss_bc≈0.59、权重生成
- [x] Task 6: visualize.py 适配与全链路冒烟（R6）
  - [x] 6.1 argparse 化（data/model/modes/width/limit/output-dir）、Agg 后端、4 元组解包、model(inputs, pos_dist)、移除 Windows 路径；误差图口径与 train.py 对齐（复用 from train import erode_mask，import 无副作用已确认）
  - [x] 6.2 CPU 冒烟：train.py --epochs 1 --limit-samples 4 跑通（五路损失正常）；visualize.py 加载 v2 权重生成 2 张六联图（1800×1000 非空白）
  - [x] 6.3 旧 dual_encoder_fno_model.pth 未动（17465954B/mtime 一致）

# Task Dependencies

- Task 1 → Task 2（norm_factor）、Task 5（λ 初值）依赖其实测结果
- Task 2、Task 3、Task 4 相互独立，可并行
- Task 5 依赖 Task 2（距离位置辅助）、Task 3（新注意力）、Task 4（新模型）
- Task 6 依赖 Task 1–5 全部完成
