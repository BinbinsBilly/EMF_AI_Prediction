# Checklist

## 数据验证（Task 1）✅ 2026-09-29 实测
- [x] 实测记录：区域图唯一值全集 = {0,1,2,3}；值 0 罕见（~2% 样本），按 one-hot 全零处理；值 3（边界）为内部环形结构，非最外圈
- [x] 实测记录：电位全局 min=0 / **max=60000.0**（全量 1314 对扫描）；**norm_factor 落定 60000**（原 20000 偏小，目标曾越界至 3.0）
- [x] 实测记录：**边界(值3)电位恒 60000V、源极(值1)电位恒 0V**（30 样本并集均为单值）；源极严格等位（极差<1V 全成立）；源极连通域 1–2 个、像素 112–4876（loss_eq 依据充分）

## R1 可学习 4 轴 RoPE（融合距离）✅
- [x] RoPE 频率为每 block 每轴（共 4 轴）独立 nn.Parameter（rope_log_freqs (4,2)），log 参数化，几何级数初始化（波长 224/28px，覆盖 ~4px–256px 尺度）
- [x] 4 个位置轴 = y 像素坐标、x 像素坐标、到源极 EDT 距离、到边界 EDT 距离
- [x] 距离由 dataset 用 distance_transform_edt 在未填充区域图上计算，像素单位 zero-pad，作为位置辅助传入 attention（不进编码器通道）
- [x] Q 用全分辨率坐标、K 用池化格心坐标 (idx+0.5)×downsample_factor，K 距离轴同步池化
- [x] head_dim=16 拆 4 轴各 4 维（2 频率对/轴）
- [x] 频率值训练中写入 TensorBoard（16 标量：2 block × 4 轴 × 2 对，实测与初始化值吻合且随训练更新）
- [x] s2g 与 g2s 两 block、4 轴频率参数相互独立（is not + 梯度独立验证）
- [x] 距离轴承接"近源/近边界处场变化剧烈"的局部性先验（无 ALiBi、无因果掩码、无 K/V 屏蔽，代码确认不存在）

## R2 轴向谱卷积 ✅
- [x] AxialSpectralConv = y 轴 1D 谱卷积 → x 轴 1D 谱卷积，均带通道混合，AMP 下无 ComplexHalf 错误（CPU autocast 实测通过，沿用 float32 权重策略）
- [x] modes=32 时全模型参数 689,105（0.689M，±10% 内），单谱块 131,072
- [x] 1×1 skip + GELU 块结构保持

## R3/R4 损失 ✅
- [x] loss_bc 仅在源极∪边界像素计算（目标恒 0/1.0）；loss_eq = mean((∇φ·H)²·导体(源极∪边界)内点)，不依赖真值；实测源极细条带无内点、边界厚导体内点=1350，loss_eq 在边界内点生效（冒烟 >0）
- [x] erode_mask 为 5 点十字模板（torch.minimum(垂直,水平) 实现，加号形判别用例通过；规避"可分离复合=3×3 盒式"陷阱）
- [x] 全部物理损失无量纲化（×H 或 ×H²）；真实 batch：loss_g 加权贡献 ~1.5%、loss_bc ~2.5%（≥1%），loss_p=19 非零有限、loss_eq>0
- [x] erode_mask 玩具张量单元验证通过（旧逻辑伪梯度 14.125 > 0，新实现 0.0 < 1e-10）
- [x] "为何不用解析/谱导数"已在 spec 记录（周期性假设被零填充违反）

## R5 归一化与稳定性 ✅
- [x] 模型中 nn.BatchNorm2d 实例数为 0（扫描验证，GroupNorm(8) × 4 处）
- [x] CrossAttentionBlock 含 pre-norm 逐像素通道 LayerNorm（PixelLayerNorm 实现 spec 所述"GroupNorm(1) 即逐像素通道 LN"之意图）
- [x] 梯度累积开关默认 1（关闭），--grad-accum 3 实测含尾批 flush，AMP 分支 GradScaler 语义正确（N 次 scale/backward，1 次 step/update），CPU 分支 loss/N

## R6 管线与回归 ✅
- [x] train.py/visualize.py 全超参 CLI 化（含 --limit-samples/--limit），无 Windows 绝对路径残留
- [x] 五路损失分项 + 16 个 RoPE 频率写入 TensorBoard（实测 22 标量）
- [x] CPU 冒烟：--epochs 1/2 --limit-samples 4 跑通；总损失 29.1→19.1（2 epoch）下降、无 NaN
- [x] visualize.py 加载 v2 权重推理出图不崩溃（2 张六联图 1800×1000，非空白 44.5% 非白像素）
- [x] 旧 dual_encoder_fno_model.pth 未被修改（17465954B/mtime 一致）；新训练写 dual_encoder_fno_model_v2.pth（2,780,397B）
- [x] mask 语义回归：填充区输出仍被损失排除（masked mean 分母用 mask.sum）；输入仍为 3 通道 one-hot（距离为位置辅助，非输入通道）

## 最终统一验证（2026-09-29，12 项全 PASS）
BatchNorm=0 ✓ | 参数 689,105 ✓ | RoPE 独立 (4,2) ✓ | 频率初值 2π/{224,28} ✓ | forward 双路径形状 ✓ | 无 NaN ✓ | 十字腐蚀判别 ✓ | 五路损失有限 ✓ | loss_bc 量级 ✓ | loss_eq>0 ✓ | 旧权重未动 ✓ | 新权重存在 ✓
