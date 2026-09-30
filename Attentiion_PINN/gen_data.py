#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_data.py — 2D 静电场 PINN 数据集自动生成（阶段二 Task 6，四几何族）

生成流程
========
1. 每个工况(condition)用独立 RNG 流（由 --seed 派生，完全可复现）在 256×W 基准
   画布上随机生成一族 HV 电极布局（参数空间见 GEOM_SPEC）：
     rect    : 1~10 个实心矩形，边长 16~110 px
     circle  : 1~6 个实心圆/椭圆，半轴 10~52 px
     polygon : 1~4 个直角多边形（L/T/U 模板，旋转 0/90/180/270°）
     blob    : 1~3 个极坐标傅里叶扰动 blob，r(θ)=r0(1+Σ_{k=2..5} a_k cos(kθ+φ_k))，
               |a_k|≤0.25 且 Σ|a_k|≤0.7，星形域逐像素精确填充（构造上不自交）
   布局约束：画布外框 1px 源极(0V)环=1；空气=2；HV 电极=3；电极与外框、电极两两
   之间留 ≥8 px 空气隙（bbox 各膨胀 4px 不重叠 ⇔ 像素间距 ≥8）。
2. 每个变体(variant)仅改变画布高度：基准布局中心裁剪到目标高度后重画 1px 源极环；
   裁剪后任一 HV 连通域 <20 px 或零电极 → 整个布局重采样（每工况最多 200 次）。
   高度：默认变体0=256，其余从 [246,236,...,166] 无放回抽取（--heights 可显式指定）。
   裁剪造成的"电极-源极环粘连"（0↔60000 界面跳变）为物理正常的不连续，
   与现有数据一致（Task 5 实测 391/1314 对存在）。
3. 真值：直接 import 复用 verify_data.solve_laplace_fd（Task 5 已验证的 5 点 FD
   精确解，spsolve 相对残差 ~1e-15，单一来源不复制实现）——
   空气=FD 解、源极=0V、HV=60000V；float32 存储。

落盘前五项校验（任一失败拒写/删除该对并记录，单对失败不影响其余，最终 exit 1）
======================================================================
① BC 精确      : 源极像素电位全 0、HV 全 60000（内存 float32 精确验证；
                  写盘后读回验证 region 逐值一致、BC 精确、空气值与意图 float32
                  值最大偏差 ≤ %.3f 舍入界 5.1e-4 V）
② 导体 E=0     : 源极∪HV 十字内点上、邻居与中心"同种导体"方向的差分精确为 0
                  （与 verify_data 语义一致，排除 src-HV 粘连界面的正常跳变）
③ 求解收敛      : solve_laplace_fd 相对残差 < --tol-solve（默认 1e-10）
④ 存储场 Laplace: 读回后的存储场在空气区十字内点 5 点离散 Laplace 残差 max < 0.05 V
⑤ 细网格抽检    : 每工况前 ceil(ratio×variants) 个变体（默认 25%，即变体0），
                  电极几何坐标×2（np.kron 2×2 上采样画 512 画布）重新求解后
                  2×2 均值池化降采样回原尺寸，与原解在空气区比较 rel L2 < 0.02

输出布局
========
  {out}/{family}/{family}_c{cond:03d}/original_region_data_{v}_{H}.csv
  {out}/{family}/{family}_c{cond:03d}/potential_distribution_{v}_{H}.csv
  {out}/{family}/gen_report.json        （每对校验结果 + 耗时 + 布局元数据）
CSV 格式与现有数据对齐：无表头、逗号分隔；region 整数、potential %.3f 定点。

用法
====
  python3 gen_data.py --family rect --conditions 2 --variants 2 --out /workspace/data
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_data import solve_laplace_fd  # noqa: E402  # 单一来源复用 Task5 求解器

# ---------------------------------------------------------------- 常量
SRC, AIR, HV = 1, 2, 3
V_SRC, V_HV = 0.0, 60000.0
BASE_H = 256                              # 布局基准画布高（宽度 = --size）
GAP = 8                                   # 电极-外框/电极间最小空气隙 (px)
MIN_ELECTRODE_PX = 20                     # 裁剪后 HV 连通域最小像素数
MAX_LAYOUT_TRIES = 200                    # 每工况布局重采样上限
PLACE_TRIES = 400                         # 单电极随机放置尝试上限
DEFAULT_HEIGHTS = list(range(256, 165, -10))   # [256,246,...,166]
FAM_ID = {"rect": 0, "circle": 1, "polygon": 2, "blob": 3}
LAP_TOL = 0.05                            # 校验④阈值 (V)
REFINE_TOL = 0.02                         # 校验⑤阈值 (rel L2)
ROUNDTRIP_TOL = 5.1e-4                    # %.3f 舍入上界 0.0005 + 余量 (V)
POT_FMT = "%.3f"                          # 与现有数据一致

GEOM_SPEC = {
    "rect": "1~10 个实心矩形电极，边长 16~110px 均匀随机，位置均匀随机；"
            "电极-外框及电极两两之间 ≥8px 空气隙",
    "circle": "1~6 个实心圆/椭圆电极，半轴 10~52px（行/列方向独立采样）；同上间隙约束",
    "polygon": "1~4 个直角多边形电极（L/T/U 模板：臂厚 10~22px、主长 36~84px、"
               "横臂 24~84px、U 开口 12~30px；整体旋转 0/90/180/270°）；同上间隙约束",
    "blob": "1~3 个极坐标傅里叶扰动 blob：r(θ)=r0(1+Σ_{k=2..5} a_k cos(kθ+φ_k))，"
            "r0∈[18,44]px、|a_k|≤0.25 且 Σ|a_k|≤0.7（星形域构造上不自交，"
            "逐像素 ρ≤r(θ) 精确填充）；同上间隙约束",
}


# ---------------------------------------------------------------- 几何基元
def _gap_ok(a, b, dilate=GAP // 2):
    """两个 bbox（r0,r1,c0,c1 闭区间）各膨胀 dilate 后不重叠 ⇔ 实际像素间距 ≥ GAP。"""
    r_ov = max(a[0] - dilate, b[0] - dilate) <= min(a[1] + dilate, b[1] + dilate)
    c_ov = max(a[2] - dilate, b[2] - dilate) <= min(a[3] + dilate, b[3] + dilate)
    return not (r_ov and c_ov)


def _try_place(rng, placed, h, w, H, W):
    """在内部区域（与外框环留 ≥GAP）随机放置 h×w bbox，与已放置电极保持 ≥GAP。"""
    r_lo, r_hi = 1 + GAP, H - 1 - GAP - h
    c_lo, c_hi = 1 + GAP, W - 1 - GAP - w
    if r_hi < r_lo or c_hi < c_lo:
        return None
    for _ in range(PLACE_TRIES):
        r0 = int(rng.integers(r_lo, r_hi + 1))
        c0 = int(rng.integers(c_lo, c_hi + 1))
        box = (r0, r0 + h - 1, c0, c0 + w - 1)
        if all(_gap_ok(box, b) for b in placed):
            return box
    return None


def _canvas(H, W):
    """全空气画布 + 外框 1px 源极环。"""
    region = np.full((H, W), AIR, dtype=np.uint8)
    region[0, :] = region[-1, :] = SRC
    region[:, 0] = region[:, -1] = SRC
    return region


# ---------------------------------------------------------------- 四族生成器
def gen_rect(rng, H, W):
    n = int(rng.integers(1, 11))
    placed, eles = [], []
    for _ in range(n):
        h = int(rng.integers(16, 111))
        w = int(rng.integers(16, 111))
        box = _try_place(rng, placed, h, w, H, W)
        if box is None:
            continue  # 放不下则少放（放不下≠失败）
        placed.append(box)
        eles.append({"kind": "rect", "h": h, "w": w,
                     "rows": [box[0], box[1]], "cols": [box[2], box[3]]})
    region = _canvas(H, W)
    for b in placed:
        region[b[0]:b[1] + 1, b[2]:b[3] + 1] = HV
    return region, eles


def gen_circle(rng, H, W):
    n = int(rng.integers(1, 7))
    placed, eles = [], []
    for _ in range(n):
        ax = int(rng.integers(10, 53))   # 半轴（列向）
        ay = int(rng.integers(10, 53))   # 半轴（行向）
        box = _try_place(rng, placed, 2 * ay + 1, 2 * ax + 1, H, W)
        if box is None:
            continue
        placed.append(box)
        eles.append({"kind": "ellipse", "cy": (box[0] + box[1]) // 2,
                     "cx": (box[2] + box[3]) // 2, "ay": ay, "ax": ax})
    region = _canvas(H, W)
    for e in eles:
        cy, cx, ay, ax = e["cy"], e["cx"], e["ay"], e["ax"]
        yy, xx = np.mgrid[cy - ay:cy + ay + 1, cx - ax:cx + ax + 1]
        m = ((yy - cy) / ay) ** 2 + ((xx - cx) / ax) ** 2 <= 1.0
        region[cy - ay:cy + ay + 1, cx - ax:cx + ax + 1][m] = HV
    return region, eles


def _make_poly(rng):
    """L/T/U 直角多边形模板（布尔 mask），整体旋转 0/90/180/270°。"""
    t = int(rng.integers(10, 23))        # 臂厚
    L1 = int(rng.integers(36, 85))       # 主长（总高）
    kind = int(rng.integers(0, 3))       # 0=L 1=T 2=U
    if kind == 2:
        g = int(rng.integers(12, 31))    # U 开口宽
        L2 = 2 * t + g
    else:
        L2 = int(rng.integers(24, 85))   # 横臂长
    m = np.zeros((L1, L2), dtype=bool)
    if kind == 0:                        # L：竖条 + 底横条
        m[:, :t] = True
        m[-t:, :] = True
        name = "L"
    elif kind == 1:                      # T：竖杆 + 顶横条
        c0 = (L2 - t) // 2
        m[:, c0:c0 + t] = True
        m[:t, :] = True
        name = "T"
    else:                                # U：双竖条 + 底横条
        m[:, :t] = True
        m[:, -t:] = True
        m[-t:, :] = True
        name = "U"
    k = int(rng.integers(0, 4))
    return name, k * 90, np.rot90(m, k)


def gen_polygon(rng, H, W):
    n = int(rng.integers(1, 5))
    placed, eles, masks = [], [], []
    for _ in range(n):
        name, rot, m = _make_poly(rng)
        h, w = m.shape
        box = _try_place(rng, placed, h, w, H, W)
        if box is None:
            continue
        placed.append(box)
        eles.append({"kind": name, "rot_deg": rot, "h": int(h), "w": int(w),
                     "rows": [box[0], box[1]], "cols": [box[2], box[3]]})
        masks.append((box, m))
    region = _canvas(H, W)
    for b, m in masks:
        region[b[0]:b[0] + m.shape[0], b[2]:b[2] + m.shape[1]][m] = HV
    return region, eles


def gen_blob(rng, H, W):
    n = int(rng.integers(1, 4))
    placed, eles = [], []
    for _ in range(n):
        r0 = float(rng.uniform(18.0, 44.0))
        a = rng.uniform(-0.25, 0.25, size=4)            # 谐波 k=2..5
        s = float(np.abs(a).sum())
        if s > 0.7:                                      # 限幅：保证 r(θ)≥0.3·r0>0
            a = a * (0.7 / s)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=4)
        rmax = r0 * (1.0 + float(np.abs(a).sum()))
        side = 2 * int(math.ceil(rmax)) + 1
        box = _try_place(rng, placed, side, side, H, W)
        if box is None:
            continue
        placed.append(box)
        eles.append({"kind": "blob", "r0": r0, "a": a.tolist(), "phi": phi.tolist(),
                     "center": [(box[0] + box[1]) // 2, (box[2] + box[3]) // 2],
                     "side": side})
    region = _canvas(H, W)
    for e in eles:
        cy, cx = e["center"]
        half = e["side"] // 2
        yy, xx = np.mgrid[cy - half:cy + half + 1, cx - half:cx + half + 1]
        dy = (yy - cy).astype(np.float64)
        dx = (xx - cx).astype(np.float64)
        rho = np.hypot(dx, dy)
        th = np.arctan2(dy, dx)
        r_th = e["r0"] * (1.0 + sum(e["a"][i] * np.cos((i + 2) * th + e["phi"][i])
                                    for i in range(4)))
        m = rho <= r_th
        region[cy - half:cy + half + 1, cx - half:cx + half + 1][m] = HV
    return region, eles


GENERATORS = {"rect": gen_rect, "circle": gen_circle,
              "polygon": gen_polygon, "blob": gen_blob}


# ---------------------------------------------------------------- 变体（高度裁剪）
def make_variant(region_base, H):
    """基准 256 布局中心裁剪到高度 H，外框重画 1px 源极环。"""
    top = (region_base.shape[0] - H) // 2
    v = region_base[top:top + H, :].copy()
    v[0, :] = v[-1, :] = SRC
    v[:, 0] = v[:, -1] = SRC
    return v


def variant_ok(v):
    """裁剪后：至少 1 个 HV 电极，且所有 HV 连通域 ≥ MIN_ELECTRODE_PX（防碎片）。"""
    lbl, n = ndimage.label(v == HV, structure=ndimage.generate_binary_structure(2, 1))
    if n == 0:
        return False, "no_electrode_after_crop"
    sizes = np.bincount(lbl.ravel())[1:]
    if int(sizes.min()) < MIN_ELECTRODE_PX:
        return False, f"electrode_fragment_lt{MIN_ELECTRODE_PX}px(min={int(sizes.min())})"
    return True, f"n_hv_components={n}"


def pick_heights(rng, variants, heights_list):
    if heights_list:
        return heights_list[:variants]
    if variants == 1:
        return [BASE_H]
    rest = [h for h in DEFAULT_HEIGHTS if h != BASE_H]
    return [BASE_H] + [int(x) for x in
                       rng.choice(np.array(rest), size=variants - 1, replace=False)]


# ---------------------------------------------------------------- 校验指标
def conductor_grad_max(pot, region):
    """② 导体 E=0：源极∪HV 十字内点上、邻居与中心同种导体方向的差分绝对值 max。
    （src-HV 裁剪粘连处的 0↔60000 界面跳变为物理正常不连续，不计——与
    verify_data 的 e0 语义完全一致。）"""
    H, W = region.shape
    cond = (region == SRC) | (region == HV)
    cp = np.pad(cond, 1)
    interior = cond & cp[:-2, 1:-1] & cp[2:, 1:-1] & cp[1:-1, :-2] & cp[1:-1, 2:]
    if not interior.any():
        return 0.0
    sp = np.pad(pot, 1)
    rp = np.pad(region, 1, constant_values=-1)
    gmax = 0.0
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nbv = sp[1 + di:H + 1 + di, 1 + dj:W + 1 + dj]
        nbr = rp[1 + di:H + 1 + di, 1 + dj:W + 1 + dj]
        g = np.abs(nbv - pot)[interior & (nbr == region)]
        if g.size:
            gmax = max(gmax, float(g.max()))
    return gmax


def air_laplace_max(pot, region):
    """④ 空气区十字内点（4 邻居均为空气）的 5 点离散 Laplace 残差 max。"""
    air = region == AIR
    H, W = region.shape
    ap = np.pad(air, 1)
    ai = air & ap[:-2, 1:-1] & ap[2:, 1:-1] & ap[1:-1, :-2] & ap[1:-1, 2:]
    if not ai.any():
        return 0.0
    s = pot.astype(np.float64)
    lap = 4.0 * s.copy()
    lap[:-1, :] -= s[1:, :]
    lap[1:, :] -= s[:-1, :]
    lap[:, :-1] -= s[:, 1:]
    lap[:, 1:] -= s[:, :-1]
    return float(np.abs(lap[ai]).max())


def refine_check(region, pot64):
    """⑤ 细网格抽检：几何坐标×2（np.kron 2×2）画 512 画布重解 → 2×2 均值池化
    降采样 → 与原解空气区 rel L2（离散误差收敛性二方法一致性）。"""
    big = np.kron(region, np.ones((2, 2), dtype=region.dtype))
    pot_big, info = solve_laplace_fd(big.astype(np.float64), V_src=V_SRC, V_hv=V_HV)
    for k in ("A", "b", "air_mask"):
        info.pop(k, None)
    H, W = region.shape
    pooled = pot_big.reshape(H, 2, W, 2).mean(axis=(1, 3))
    air = region == AIR
    d = (pot64 - pooled)[air]
    denom = float(np.linalg.norm(pot64[air]))
    rel = float(np.linalg.norm(d) / denom) if denom > 0 else float(np.linalg.norm(d))
    return rel, float(info["solve_residual"])


def src_hv_adjacent_px(region):
    """源极-HV 直接相邻像素数（几何统计，与 verify_data 同口径）。"""
    src_m = region == SRC
    hv_pad = np.pad(region == HV, 1)
    H, W = region.shape
    adj = 0
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        adj += int((src_m & hv_pad[1 + di:H + 1 + di, 1 + dj:W + 1 + dj]).sum())
    return adj


# ---------------------------------------------------------------- 单对处理
def process_pair(cond_dir, v_idx, height, region, do_refine, args):
    t0 = time.time()
    H, W = region.shape
    src_m = region == SRC
    hv_m = region == HV

    def finish(rec):
        rec["sec"] = round(time.time() - t0, 2)
        return rec

    rec = {
        "variant": v_idx, "height": height, "shape": f"{H}x{W}",
        "n_air": int((region == AIR).sum()),
        "n_hv_px": int(hv_m.sum()),
        "n_hv_components": int(
            ndimage.label(hv_m, structure=ndimage.generate_binary_structure(2, 1))[1]),
        "src_hv_adjacent_px": src_hv_adjacent_px(region),
        "refine_done": bool(do_refine),
        "ok": False, "fail_reason": "", "checks": {},
    }

    # ---- 真值求解（FD 精确解，单一来源复用 verify_data.solve_laplace_fd）
    pot, info = solve_laplace_fd(region.astype(np.float64), V_src=V_SRC, V_hv=V_HV)
    for k in ("A", "b", "air_mask"):
        info.pop(k, None)
    pot32 = pot.astype(np.float32)

    # ③ 求解收敛 + 几何异常防护（本生成器结构上不应出现 0 值/隔离空气）
    rec["checks"]["solve_residual"] = float(info["solve_residual"])
    if not info["solve_residual"] < args.tol_solve:
        rec["fail_reason"] = (f"solve_residual={info['solve_residual']:.3e} "
                              f">= {args.tol_solve:g}")
        return finish(rec)
    if info["n_invalid"] != 0 or info["isolated_air"] != 0:
        rec["fail_reason"] = (f"geometry_anomaly n_invalid={info['n_invalid']} "
                              f"isolated_air={info['isolated_air']}")
        return finish(rec)

    # ① BC 精确（内存 float32，浮点精确比较）
    if not (np.all(pot32[src_m] == np.float32(V_SRC))
            and np.all(pot32[hv_m] == np.float32(V_HV))):
        rec["fail_reason"] = "bc_not_exact_in_memory"
        return finish(rec)
    rec["checks"]["bc_exact_mem"] = True
    if not set(np.unique(region).tolist()) <= {SRC, AIR, HV}:
        rec["fail_reason"] = "region_value_out_of_range"
        return finish(rec)

    # ② 导体 E=0（同种导体方向，精确 0）
    e0_max = conductor_grad_max(pot32, region)
    rec["checks"]["e0_grad_max"] = e0_max
    if e0_max != 0.0:
        rec["fail_reason"] = f"conductor_E_not_zero(max={e0_max:g})"
        return finish(rec)

    # ④ 存储场（float32 内存场）空气区 Laplace 残差 —— 预检
    lap_f32 = air_laplace_max(pot32, region)
    rec["checks"]["stored_laplace_max_f32"] = lap_f32
    if lap_f32 >= LAP_TOL:
        rec["fail_reason"] = f"stored_laplace_f32={lap_f32:.4f}V >= {LAP_TOL}V"
        return finish(rec)

    # ⑤ 细网格抽检
    if do_refine:
        t_r = time.time()
        rel, resid_big = refine_check(region, pot)
        rec["checks"]["refine_rel_l2"] = rel
        rec["checks"]["refine_solve_residual"] = resid_big
        rec["checks"]["refine_sec"] = round(time.time() - t_r, 2)
        if not rel < REFINE_TOL:
            rec["fail_reason"] = f"refine_rel_l2={rel:.4f} >= {REFINE_TOL}"
            return finish(rec)

    # ---- 全部通过，落盘
    rpath = os.path.join(cond_dir, f"original_region_data_{v_idx}_{height}.csv")
    ppath = os.path.join(cond_dir, f"potential_distribution_{v_idx}_{height}.csv")
    np.savetxt(rpath, region, fmt="%d", delimiter=",")
    np.savetxt(ppath, pot32, fmt=POT_FMT, delimiter=",")

    # ①（落盘部分）读回往返一致 + ④ 存储场（读回后）Laplace 残差
    r_back = pd.read_csv(rpath, header=None).to_numpy()
    p_back = pd.read_csv(ppath, header=None).to_numpy()
    ok_shape = r_back.shape == region.shape and p_back.shape == region.shape
    ok_rt_region = bool(ok_shape and np.array_equal(r_back, region))
    ok_rt_bc = bool(np.all(p_back[src_m] == V_SRC) and np.all(p_back[hv_m] == V_HV))
    dmax = float(np.abs(p_back - pot32.astype(np.float64)).max()) if ok_shape else float("nan")
    lap_csv = air_laplace_max(p_back, region) if ok_shape else float("nan")
    rec["checks"].update({
        "region_roundtrip_exact": ok_rt_region,
        "bc_exact_roundtrip": ok_rt_bc,
        "roundtrip_max_abs_diff_v": dmax,
        "stored_laplace_max_csv": lap_csv,
    })
    if not (ok_rt_region and ok_rt_bc and dmax <= ROUNDTRIP_TOL and lap_csv < LAP_TOL):
        for p in (rpath, ppath):
            if os.path.exists(p):
                os.remove(p)
        rec["fail_reason"] = (f"roundtrip_check_failed(region_rt={ok_rt_region} "
                              f"bc_rt={ok_rt_bc} dmax={dmax:.2e} lap_csv={lap_csv:.4f})")
        return finish(rec)

    rec["ok"] = True
    rec["files"] = [os.path.basename(rpath), os.path.basename(ppath)]
    return finish(rec)


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(
        description="2D 静电场数据集自动生成（四几何族 + FD 精确解 + 五项校验）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--family", required=True, choices=sorted(FAM_ID),
                    help="几何族")
    ap.add_argument("--conditions", type=int, default=2, help="工况数")
    ap.add_argument("--variants", type=int, default=2, help="每工况变体数")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--out", default="/workspace/data", help="输出根目录")
    ap.add_argument("--size", type=int, default=256, help="画布宽（px），高度可变")
    ap.add_argument("--heights", default="",
                    help="逗号分隔高度列表（显式指定时按序取前 variants 个）；"
                         "默认变体0=256、其余从 [246,...,166] 无放回抽取")
    ap.add_argument("--tol-solve", type=float, default=1e-10,
                    help="校验③：FD 求解相对残差阈值")
    ap.add_argument("--refine-check-ratio", type=float, default=0.25,
                    help="校验⑤：每工况参与细网格抽检的变体比例")
    ap.add_argument("--no-refine", action="store_true", help="关闭细网格抽检（调试用）")
    args = ap.parse_args()

    if args.conditions < 1 or args.variants < 1:
        ap.error("--conditions/--variants 必须 ≥ 1")
    heights_list = []
    if args.heights.strip():
        try:
            heights_list = [int(x) for x in args.heights.split(",") if x.strip()]
        except ValueError:
            ap.error("--heights 必须是逗号分隔整数列表")
        if not heights_list or any(not (20 <= h <= BASE_H) for h in heights_list):
            ap.error(f"--heights 每项需满足 20 ≤ h ≤ {BASE_H}（基准画布高）")
        if len(heights_list) < args.variants:
            ap.error(f"--heights 仅 {len(heights_list)} 项 < --variants {args.variants}")
    elif args.variants > len(DEFAULT_HEIGHTS):
        ap.error(f"默认高度池仅 {len(DEFAULT_HEIGHTS)} 项，--variants 不得超过")

    n_refine = 0 if args.no_refine else int(
        math.ceil(args.refine_check_ratio * args.variants))
    n_refine = max(0, min(n_refine, args.variants))

    fam_out = os.path.join(args.out, args.family)
    os.makedirs(fam_out, exist_ok=True)
    t_start = time.time()

    report = {
        "meta": {
            "script": os.path.abspath(__file__),
            "family": args.family,
            "geom_spec": GEOM_SPEC[args.family],
            "seed": args.seed,
            "conditions": args.conditions,
            "variants_per_condition": args.variants,
            "size_w": args.size,
            "base_layout_canvas": f"{BASE_H}x{args.size}",
            "heights_explicit": heights_list or None,
            "heights_default_pool": None if heights_list else DEFAULT_HEIGHTS,
            "height_rule": ("explicit" if heights_list
                            else "variant0=256 + 无放回抽样"),
            "tol_solve": args.tol_solve,
            "laplace_tol_v": LAP_TOL,
            "roundtrip_tol_v": ROUNDTRIP_TOL,
            "refine_check_ratio": args.refine_check_ratio,
            "refine_tol_rel_l2": REFINE_TOL,
            "n_refine_variants_per_condition": n_refine,
            "gap_px": GAP,
            "min_electrode_px_after_crop": MIN_ELECTRODE_PX,
            "csv_format": {"region": "int,%d", "potential": f"float32,{POT_FMT}",
                           "delimiter": ",", "header": False},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "conditions": [],
    }

    planned = args.conditions * args.variants
    n_written = 0
    print(f"[{args.family}] 生成 {args.conditions} 工况 × {args.variants} 变体 "
          f"→ {fam_out}（seed={args.seed}, 每工况前 {n_refine} 个变体做细网格抽检）",
          flush=True)

    for cond in range(args.conditions):
        rng = np.random.default_rng([args.seed, FAM_ID[args.family], cond])
        heights = pick_heights(rng, args.variants, heights_list)
        cond_rec = {"condition": f"{args.family}_c{cond:03d}", "heights": heights,
                    "ok": True, "fail_reason": "", "layout_attempts": 0,
                    "electrodes": [], "variants": []}
        t_cond = time.time()

        # ---- 布局采样（裁剪碎片/零电极 → 整布局重采样）
        layout = None
        for attempt in range(1, MAX_LAYOUT_TRIES + 1):
            region_base, eles = GENERATORS[args.family](rng, BASE_H, args.size)
            vs = [make_variant(region_base, h) for h in heights]
            cond_rec["layout_attempts"] = attempt
            if all(variant_ok(v)[0] for v in vs):
                layout = (region_base, eles, vs)
                break
        if layout is None:
            cond_rec["ok"] = False
            cond_rec["fail_reason"] = (f"layout_resample_exhausted({MAX_LAYOUT_TRIES}): "
                                       "裁剪后始终出现 <20px 电极碎片或零电极")
            report["conditions"].append(cond_rec)
            print(f"[{args.family}] {cond_rec['condition']}: 布局重采样耗尽 → 工况失败",
                  flush=True)
            continue
        region_base, eles, vs = layout
        cond_rec["electrodes"] = eles
        cond_rec["n_electrodes_base"] = len(eles)

        cond_dir = os.path.join(fam_out, cond_rec["condition"])
        os.makedirs(cond_dir, exist_ok=True)
        print(f"[{args.family}] {cond_rec['condition']}: heights={heights}, "
              f"电极数(256布局)={len(eles)}, 布局尝试={cond_rec['layout_attempts']}",
              flush=True)

        for vi, (h, v) in enumerate(zip(heights, vs)):
            rec = process_pair(cond_dir, vi, h, v, vi < n_refine, args)
            cond_rec["variants"].append(rec)
            if rec["ok"]:
                n_written += 1
                extra = (f", refine_rel_l2={rec['checks']['refine_rel_l2']:.2e}"
                         if rec["refine_done"] else "")
                print(f"[{args.family}]   var{vi} H={h}: OK "
                      f"({rec['sec']:.1f}s, lap_max={rec['checks']['stored_laplace_max_csv']:.4f}V"
                      f"{extra})", flush=True)
            else:
                print(f"[{args.family}]   var{vi} H={h}: FAIL ({rec['fail_reason']})",
                      flush=True)
        cond_rec["sec"] = round(time.time() - t_cond, 1)
        report["conditions"].append(cond_rec)

    report["summary"] = {
        "pairs_planned": planned,
        "pairs_written": n_written,
        "pairs_failed": planned - n_written,
        "conditions_ok": sum(1 for c in report["conditions"] if c["ok"]),
        "checks": ["1_BC_exact(mem+roundtrip)", "2_conductor_E0(same-kind)",
                   "3_solve_residual<tol", "4_stored_laplace<0.05V",
                   "5_refine_rel_l2<0.02"],
        "duration_sec": round(time.time() - t_start, 1),
    }
    rep_path = os.path.join(fam_out, "gen_report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[{args.family}] 完成: {n_written}/{planned} 对通过全部校验并写入, "
          f"总耗时 {report['summary']['duration_sec']}s")
    print(f"[{args.family}] 报告: {rep_path}")
    sys.exit(0 if n_written == planned else 1)


if __name__ == "__main__":
    main()
