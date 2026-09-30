"""eval_utils.py — 2D 静电场 PINN 统一评估指标模块（阶段二 Task 3）

口径与 train.py 完全一致（单一来源计划：train.py 后续将改为从本模块导入，
当前为避免 train.py 顶层 import 副作用 / 循环依赖，此处独立维护一份一致实现）：

  * 十字腐蚀 erode_mask：5 点十字（中心 + 上下左右全为 1 才存活），
    实现为垂直 (3,1) 与水平 (1,3) min-pool 的逐元素最小值，
    与 train.py 的 erode_mask 逐字一致。
  * 无量纲化：一阶差分乘 H、二阶差分乘 H²，H = pred.size(-2)
    （数据统一 pad 到 256，故 H=256），与 train.py 的
    calculate_physics_loss / calculate_eq_loss 同口径。
  * 掩码语义：逐样本在有效像素上计算指标，再对 batch 求平均；
    mask 全零（或指标区域为空）的样本计 0，不产生 NaN。
  * 差分：torch.gradient 两次（train.py 同款），内部中心差分、
    边界单侧差分；区域边界像素由十字腐蚀排除。

张量约定：pred/target/mask 统一 (B,1,H,W)，容忍 (B,H,W)（自动补通道维）；
region_input 为 one-hot (B,3,H,W)，通道顺序与 dataset.py 一致：
  ch0 = 源极 (region==1)，ch1 = 空气 (region==2)，ch2 = 边界/HV 电极 (region==3)
（padding 像素 one-hot 为 [0,0,0]，天然落在所有区域掩码之外）。
"""

import torch
import torch.nn.functional as F

_EPS = 1e-8


def _as_bchw(x, name):
    """单通道张量统一为 (B,1,H,W)：容忍 (B,H,W) 输入。"""
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if x.dim() != 4 or x.size(1) != 1:
        raise ValueError(
            f"{name} 期望 (B,1,H,W) 或 (B,H,W)，实际 shape={tuple(x.shape)}")
    return x


def _check_onehot(region_input):
    if region_input.dim() != 4 or region_input.size(1) != 3:
        raise ValueError(
            "region_input 期望 one-hot (B,3,H,W)（ch0=源极 ch1=空气 ch2=边界），"
            f"实际 shape={tuple(region_input.shape)}")
    return region_input


def erode_mask(mask, region_mask=None):
    """Interior-point erosion of a binary mask with the 5-point cross
    stencil: a pixel survives only if itself and its 4 cross neighbors
    (up/down/left/right) are all 1. This matches the central-difference
    stencil of torch.gradient exactly, so every finite-difference
    neighbor of a surviving pixel lies inside the mask and full-image
    differences never mix in padded/invalid pixels.

    Implemented as the elementwise minimum of a vertical and a
    horizontal 3-tap min-pool (-max_pool2d(-x, ...)). Note: composing
    the two pools instead (horizontal pool of the vertical pool) would
    yield the more conservative 3x3 box erosion, not the cross.

    If region_mask is given, neighbors must additionally lie inside
    region_mask (simplified as erode(mask * region_mask)).

    与 train.py 的 erode_mask 保持一致（逐字相同实现）；
    train.py 后续将改为从本模块导入单一来源。
    """
    m = mask * region_mask if region_mask is not None else mask
    vertical = -F.max_pool2d(-m, kernel_size=(3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-m, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def relative_l2(pred, target, mask):
    """‖pred−target‖₂ / ‖target‖₂，范数仅在 mask 有效像素上计算；
    逐样本计算后再对 batch 求平均。

    mask 全零（或目标范数为 0）的样本计 0，不产生 NaN。
    """
    pred = _as_bchw(pred, "pred")
    target = _as_bchw(target, "target")
    mask = _as_bchw(mask, "mask")

    diff = (pred - target) * mask
    num = diff.flatten(1).norm(dim=1)
    den = (target * mask).flatten(1).norm(dim=1)
    safe = torch.where(den > 0, den, torch.ones_like(den))
    per_sample = torch.where(den > 0, num / safe, torch.zeros_like(num))
    return per_sample.mean()


def max_abs_error(pred, target, mask):
    """逐样本 mask 有效像素内 max|pred−target|，再对 batch 求平均。

    mask 全零的样本计 0（|·|≥0 与 0 掩码相乘后 max=0），不产生 NaN。
    """
    pred = _as_bchw(pred, "pred")
    target = _as_bchw(target, "target")
    mask = _as_bchw(mask, "mask")

    err = (pred - target).abs() * mask
    per_sample = err.flatten(1).max(dim=1).values
    return per_sample.mean()


def bc_violation(pred, target, region_input, mask):
    """Dirichlet 边界违反量：源极(ch0) ∪ 边界(ch2) 像素上的
    max|pred−target|，逐样本计算后再对 batch 求平均。

    与 train.py calculate_bc_loss 的 bc_mask 同源（ch0+ch2，再夹到
    指示函数并与有效区 mask 相交；padding 像素 one-hot 全零，实数据
    上与 train.py 完全等价）。无边界像素 / mask 全零的样本计 0。
    """
    pred = _as_bchw(pred, "pred")
    target = _as_bchw(target, "target")
    mask = _as_bchw(mask, "mask")
    _check_onehot(region_input)

    bc_mask = (region_input[:, 0:1] + region_input[:, 2:3]).clamp(max=1.0) * mask
    err = (pred - target).abs() * bc_mask
    per_sample = err.flatten(1).max(dim=1).values
    return per_sample.mean()


def laplace_residual(pred, region_input, mask):
    """空气区 (ch1) 十字内点上的 |∇²φ·H²| 均值（逐样本均值再 batch 平均）。

    与 train.py calculate_physics_loss 同口径：torch.gradient 两次差分
    取 dxx+dyy，乘 H²（H=pred.size(-2)，数据 pad 到 256 即 256²=65536）
    无量纲化，内点掩码为空气区十字腐蚀（空气 one-hot 在 padding 区为 0，
    与有效区 mask 相交在实数据上等价于 train.py 的 erode_mask(air)）；
    区别仅在于这里返回均值绝对值 |·| 而非平方 |·|²。
    空气内点为空（如全导体样本 / mask 全零）时计 0。
    """
    pred = _as_bchw(pred, "pred")
    mask = _as_bchw(mask, "mask")
    _check_onehot(region_input)

    air = region_input[:, 1:2, :, :]
    interior = erode_mask(air, mask)
    H = pred.size(-2)

    dy, dx = torch.gradient(pred, dim=(-2, -1))
    dyy, _ = torch.gradient(dy, dim=(-2, -1))
    _, dxx = torch.gradient(dx, dim=(-2, -1))
    abs_residual = (dxx + dyy).abs() * (H ** 2) * interior

    num = abs_residual.flatten(1).sum(dim=1)
    den = interior.flatten(1).sum(dim=1)
    per_sample = num / den.clamp(min=_EPS)
    return per_sample.mean()


def equipotential_residual(pred, region_input):
    """导体（源极∪边界，ch0+ch2）十字内点上的 |∇φ|·H 均值
    （逐样本均值再 batch 平均）。

    与 train.py calculate_eq_loss 同口径：导体掩码 (ch0+ch2).clamp(max=1)
    十字腐蚀取内点，torch.gradient 一阶差分，乘 H（=256）无量纲化；
    区别仅在于这里返回 |∇φ|·H 的均值而非 (|∇φ|·H)² 的均值。
    导体内点为空（如全空气样本，或源极条带太薄被腐蚀殆尽）时计 0。
    """
    pred = _as_bchw(pred, "pred")
    _check_onehot(region_input)

    conductor = (region_input[:, 0:1, :, :] + region_input[:, 2:3, :, :]).clamp(max=1.0)
    interior = erode_mask(conductor)
    H = pred.size(-2)

    dy, dx = torch.gradient(pred, dim=(-2, -1))
    grad_mag = torch.sqrt(dy ** 2 + dx ** 2) * H

    num = (grad_mag * interior).flatten(1).sum(dim=1)
    den = interior.flatten(1).sum(dim=1)
    per_sample = num / den.clamp(min=_EPS)
    return per_sample.mean()


def _masked_mse(pred, target, mask):
    """逐样本 masked MSE（Σ((pred−target)²·mask)/Σmask，与 train.py
    calculate_masked_mse 同式的逐样本版），再 batch 平均；mask 全零计 0。"""
    pred = _as_bchw(pred, "pred")
    target = _as_bchw(target, "target")
    mask = _as_bchw(mask, "mask")

    squared = ((pred - target) * mask) ** 2
    num = squared.flatten(1).sum(dim=1)
    den = mask.flatten(1).sum(dim=1)
    per_sample = num / den.clamp(min=_EPS)
    return per_sample.mean()


def compute_all_metrics(pred, target, input, mask):
    """统一入口：一次前向的 batch 指标汇总。

    Args:
        pred:         (B,1,H,W) 模型预测（归一化电位，[0,1] 口径）
        target:       (B,1,H,W) 归一化电位真值
        input:        (B,3,H,W) 区域 one-hot（ch0=源极 ch1=空气 ch2=边界）
        mask:         (B,1,H,W) 有效区掩码

    Returns:
        dict[str, float]：mse / relative_l2 / max_abs_error / bc_violation /
        laplace_residual / equipotential_residual（全部为 Python float，
        batch 平均；空区域样本计 0，不产生 NaN）。
    """
    return {
        "mse": float(_masked_mse(pred, target, mask)),
        "relative_l2": float(relative_l2(pred, target, mask)),
        "max_abs_error": float(max_abs_error(pred, target, mask)),
        "bc_violation": float(bc_violation(pred, target, input, mask)),
        "laplace_residual": float(laplace_residual(pred, input, mask)),
        "equipotential_residual": float(equipotential_residual(pred, input)),
    }
