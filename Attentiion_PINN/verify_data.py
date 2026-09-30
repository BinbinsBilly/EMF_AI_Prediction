#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_data.py — 2D 静电场数据集真值交叉校验（阶段二 Task 5）

对数据根目录下每个工况目录中的 (region, potential) 数据对做独立交叉校验：

  1. 独立 5 点有限差分(FD)求解器在空气区(region==2)解 Laplace 方程：
       - region==1 源极        : Dirichlet, V_src = 0 V
       - region==3 HV 电极     : Dirichlet, V_hv = 60000 V
       - region==0 无效像素    : 【实证语义】按 0V 接地电极处理（与源极同电位）。
                                 实测全部 0 值像素上 stored 精确=0V，且按此语义
                                 boundary21/_50 系列 21 对 rel_l2~1e-6（即本网格
                                 FD 精确解）；若按绝缘处理 rel_l2 高达 1.4~5.7。
                                 绝缘语义作为对照保留（invalid_as='insulate'）。
       - 图像边界外的邻居      : 按绝缘处理（零通量近似）
     稀疏矩阵全部向量化组装（预计算空气像素索引数组 + pad 后的邻居值批量取值），
     scipy.sparse.linalg.spsolve 直接求解，求解后校验相对残差 ||Ax-b||/||b|| < 1e-10。

  2. 逐对指标：
       ① rel_l2_air   = ||fd - stored||_2 / ||stored||_2   （仅在 region==2 像素）
       ② max_abs_diff_air = max |fd - stored|               （仅在 region==2 像素）
       ③ BC 精确性    : stored 在源极是否恒 0 / HV 是否恒 60000 / 0值像素恒 0
                        （浮点精确比较）
       ④ 导体 E=0     : stored 在 (源极∪HV∪0值) 的十字内点上、邻居与中心为
                        同一种导体的方向，单侧差分最大值是否精确为 0
                        （src 与 HV 直接相邻处的 0↔60000 跳变是物理正常的界面
                        不连续，不计入；实测 432 对存在此类几何粘连）
       ⑤ stored 自身空气区 5 点离散 Laplace 残差 max（4 邻居均为空气的十字内点）
       ⑥ 额外诊断: stored 在 FD 方程组下的残差 max |A·stored_air - b|
                   （直接回答“stored 是否为本网格 5 点 FD 方程的解”）
       ⑦ 隐藏电极检测与修正几何对照: region 标空气但 stored 为导体值(0/60000)
                   的像素（region 图几何错标）；把这些像素按 stored 隐含几何
                   还原为电极后重解 rel_l2（分解"region 错标"与"stored 求解
                   精度"两大误差源）。

  3. 输出：
       verification_report.csv  逐对明细
       verification_report.json 汇总统计 + 最终裁决 PASS/FAIL

用法示例：
    python3 verify_data.py --limit 3                 # 冒烟
    python3 verify_data.py                           # 全量
    python3 verify_data.py --sample-per-dir 3        # 每目录抽 3 对（超时降级方案）
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

# ---------------------------------------------------------------- 常量
SRC, AIR, HV, INVALID = 1, 2, 3, 0
REGION_PREFIX = "original_region_data_"
POT_PREFIX = "potential_distribution_"
SOLVE_RESIDUAL_TOL = 1e-10


# ---------------------------------------------------------------- IO
def read_matrix(path):
    """读无表头 float CSV，统一 float64。"""
    return pd.read_csv(path, header=None, dtype=np.float64).to_numpy(dtype=np.float64)


def collect_pairs(data_root):
    """返回 [(condition_dir, suffix, region_path, pot_path, partner_exists), ...]

    递归扫描（与 dataset.py 的递归加载语义对齐）：任意深度目录下的
    original_region_data_*.csv 与同目录同后缀 potential_distribution_*.csv 配对，
    condition_dir 为相对 data_root 的目录路径。原先"仅一层 + results 前缀过滤"
    的行为是它在旧数据根目录（各 results* 目录平铺）上的特例。
    """
    pairs = []
    for dirpath, dirnames, filenames in os.walk(data_root):
        dirnames.sort()
        names = sorted(f for f in filenames
                       if f.startswith(REGION_PREFIX) and f.endswith(".csv"))
        if not names:
            continue
        rel = os.path.relpath(dirpath, data_root)
        for fname in names:
            suffix = fname[len(REGION_PREFIX):-len(".csv")]
            rpath = os.path.join(dirpath, fname)
            ppath = os.path.join(dirpath, POT_PREFIX + suffix + ".csv")
            pairs.append((rel, suffix, rpath, ppath, os.path.exists(ppath)))
    return pairs


# ---------------------------------------------------------------- FD 求解器
def solve_laplace_fd(region, V_src=0.0, V_hv=60000.0, invalid_as="ground"):
    """
    独立 5 点有限差分 Laplace 求解器。

    参数
    ----
    region : (H, W) 数组，值域 {0,1,2,3}
             1=源极(0V Dirichlet) 2=空气(未知数) 3=HV(60000V Dirichlet)
             0=无效像素，处理方式由 invalid_as 决定（见下）
    V_src, V_hv : 两种 Dirichlet 边界电位。
    invalid_as : 'ground' —— 0 值像素按 0V 接地 Dirichlet（等效源极）处理。
                      【实证语义】实测（boundary21/_50 系列）stored 在全部 0 值
                      像素上精确为 0V，且按此语义求解 rel_l2 ~1e-6，
                      证明数据生成器把 0 值当作接地电极、而非绝缘。
                 'insulate' —— 0 值像素按绝缘处理（零通量近似：跳过该邻居、
                      对角线权重相应减少）。作为对照语义保留。

    返回
    ----
    (potential, info) :
        potential : (H, W) 完整电位场（空气=FD 解，源极/0值电极=V_src=0，
                    HV=V_hv，被隔离空气填 0）
        info      : 诊断字典，含 n_air / n_invalid / isolated_air /
                    solve_residual，以及供残差复用的 'A' / 'b' / 'air_mask'
                    （调用方用完应 pop 掉，避免长期持有大对象）。

    处理说明
    --------
    * 出界邻居一律按绝缘（零通量）处理。
    * 防奇异：若某空气连通域（4-连通）完全不接触任何 Dirichlet 电极，
      其方程组纯 Neumann、解不唯一；这些空气像素从线性系统中剔除、
      电位填 0（物理上与任何导体完全隔离，电位不定），数量记入
      info['isolated_air']。实际数据外框为 1px 接地源极环，正常不触发。
    """
    region = np.asarray(region, dtype=np.float64)

    if invalid_as == "ground":
        # 0 值像素 → 0V 接地电极（与源极同电位）
        region_eff0 = region.copy()
        m0 = region_eff0 == INVALID
        region_eff0[m0] = SRC
    elif invalid_as == "insulate":
        region_eff0 = region
        m0 = np.zeros(region.shape, dtype=bool)
    else:
        raise ValueError(f"unknown invalid_as: {invalid_as}")

    H, W = region.shape
    info = {
        "n_air": 0,
        "n_invalid": int(m0.sum()),
        "isolated_air": 0,
        "solve_residual": 0.0,
    }

    pot = np.zeros((H, W), dtype=np.float64)
    pot[region_eff0 == SRC] = V_src
    pot[region_eff0 == HV] = V_hv

    air_all = region_eff0 == AIR
    info["n_air"] = int(air_all.sum())
    if not air_all.any():
        info["A"] = info["b"] = None
        info["air_mask"] = air_all
        return pot, info

    # ---- 防奇异：剔除不接触任何 Dirichlet 电极的空气连通域 ----
    conductive = (region_eff0 == SRC) | air_all | (region_eff0 == HV)
    lbl, n_lbl = ndimage.label(conductive,
                               structure=ndimage.generate_binary_structure(2, 1))
    dir_labels = np.unique(lbl[(region_eff0 == SRC) | (region_eff0 == HV)])
    dir_labels = dir_labels[dir_labels > 0]
    keep = np.zeros(n_lbl + 1, dtype=bool)
    keep[dir_labels] = True
    if n_lbl > 0 and not keep[1:].all():
        removed = air_all & ~keep[lbl]
        info["isolated_air"] = int(removed.sum())
        region_eff = region_eff0.copy()
        region_eff[removed] = INVALID  # 被隔离空气 → 绝缘（不参与方程）
    else:
        region_eff = region_eff0

    air = region_eff == AIR
    n_air = int(air.sum())
    if n_air == 0:
        info["A"] = info["b"] = None
        info["air_mask"] = air
        return pot, info  # 全部空气被隔离，保持 0

    # ---- 向量化组装 5 点 FD 稀疏矩阵 ----
    idx = np.full((H, W), -1, dtype=np.int64)
    idx[air] = np.arange(n_air)
    ai, aj = np.nonzero(air)  # 行主序，编号即 arange

    rp = np.full((H + 2, W + 2), -1.0, dtype=np.float64)  # 出界 = -1 → 绝缘
    rp[1:-1, 1:-1] = region_eff

    diag = np.zeros(n_air, dtype=np.float64)
    b = np.zeros(n_air, dtype=np.float64)
    off_r, off_c, off_v = [], [], []
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nb = rp[ai + di + 1, aj + dj + 1]          # 各空气像素在该方向的邻居值
        nb_air = nb == AIR
        nb_src = nb == SRC
        nb_hv = nb == HV
        conducting = nb_air | nb_src | nb_hv       # 导电邻居才贡献对角权重
        diag += conducting                          # 绝缘(0/出界)邻居：权重不加
        b += np.where(nb_src, V_src, 0.0) + np.where(nb_hv, V_hv, 0.0)
        if nb_air.any():
            rows = idx[ai[nb_air], aj[nb_air]]
            cols = idx[ai[nb_air] + di, aj[nb_air] + dj]
            off_r.append(rows)
            off_c.append(cols)
            off_v.append(np.full(rows.shape, -1.0))

    main_idx = np.arange(n_air)
    if off_r:
        rows_all = np.concatenate(off_r + [main_idx])
        cols_all = np.concatenate(off_c + [main_idx])
        vals_all = np.concatenate(off_v + [diag])
    else:
        rows_all, cols_all, vals_all = main_idx, main_idx, diag
    A = coo_matrix((vals_all, (rows_all, cols_all)),
                   shape=(n_air, n_air)).tocsc()

    x = spsolve(A, b)
    if not np.all(np.isfinite(x)):
        raise RuntimeError("FD solve returned non-finite values (singular system?)")

    resid = A.dot(x) - b
    bnorm = float(np.linalg.norm(b))
    rel = float(np.linalg.norm(resid) / bnorm) if bnorm > 0 else float(np.linalg.norm(resid))
    info["solve_residual"] = rel
    info["A"] = A
    info["b"] = b
    info["air_mask"] = air
    pot[air] = x
    return pot, info


# ---------------------------------------------------------------- 指标
def cross_validate_pair(region, stored, V_src=0.0, V_hv=60000.0):
    """对单对 (region, stored) 做全部交叉校验指标计算。返回指标字典。

    主语义：0 值像素按 0V 接地电极（实证生成器语义）。
    含 0 值像素的对额外给出绝缘语义对照指标（*_insul0）。
    """
    H, W = region.shape
    fd, info = solve_laplace_fd(region, V_src, V_hv, invalid_as="ground")
    A, b, air_kept = info.pop("A"), info.pop("b"), info.pop("air_mask")

    m = {
        "n_air": info["n_air"],
        "n_invalid": info["n_invalid"],
        "n_isolated_air": info["isolated_air"],
        "solve_residual": info["solve_residual"],
    }

    # ①② 空气区（原始 region==2 掩码，含被隔离空气）上的偏差
    air_all = region == AIR
    if air_all.any():
        d = (fd - stored)[air_all]
        s_norm = float(np.linalg.norm(stored[air_all]))
        m["rel_l2_air"] = (float(np.linalg.norm(d) / s_norm)
                           if s_norm > 0 else float(np.linalg.norm(d)))
        m["max_abs_diff_air"] = float(np.abs(d).max())
    else:
        m["rel_l2_air"] = 0.0
        m["max_abs_diff_air"] = 0.0

    # 隐藏电极检测：region 标空气但 stored 为导体值 → region 图几何错标
    m["hidden_ground_px"] = int(((region == AIR) & (stored == V_src)).sum())
    m["hidden_hv_px"] = int(((region == AIR) & (stored == V_hv)).sum())

    # ③ BC 精确性（浮点精确比较；若该类导体不存在则记 True 并在计数中体现）
    src_m = region == SRC
    hv_m = region == HV
    zero_m = region == INVALID  # 0 值像素（实证=0V 接地电极）
    m["has_src"] = bool(src_m.any())
    m["has_hv"] = bool(hv_m.any())
    m["bc_src_exact"] = bool((not src_m.any()) or np.all(stored[src_m] == V_src))
    m["bc_hv_exact"] = bool((not hv_m.any()) or np.all(stored[hv_m] == V_hv))
    m["bc_zero_exact"] = bool((not zero_m.any()) or np.all(stored[zero_m] == V_src))

    # 含 0 值像素时的绝缘语义对照（任务原始假设）
    if zero_m.any():
        fd_i, info_i = solve_laplace_fd(region, V_src, V_hv, invalid_as="insulate")
        di = (fd_i - stored)[air_all]
        si = float(np.linalg.norm(stored[air_all]))
        m["rel_l2_air_insul0"] = (float(np.linalg.norm(di) / si)
                                  if si > 0 else float(np.linalg.norm(di)))
        m["max_abs_diff_air_insul0"] = float(np.abs(di).max())
        m["solve_residual_insul0"] = info_i["solve_residual"]
    else:
        m["rel_l2_air_insul0"] = np.nan
        m["max_abs_diff_air_insul0"] = np.nan
        m["solve_residual_insul0"] = np.nan

    # ④ 导体 E=0：源极∪HV∪0值电极 的十字内点上四方向差分。
    #    语义修正：仅对"邻居与中心为同一种导体"的方向计差分 ——
    #    src 与 HV 直接相邻（几何粘连）处电位 0↔60000 跳变是物理正常的
    #    界面不连续，不属于导体内部 E≠0。实测 391 对存在此类粘连。
    cond = src_m | hv_m | zero_m
    # src-HV 直接相邻像素数（几何特征统计）
    adj = 0
    hv_pad = np.pad(hv_m, 1, mode="constant")
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        adj += int((src_m & hv_pad[1 + di:H + 1 + di, 1 + dj:W + 1 + dj]).sum())
    m["src_hv_adjacent_px"] = adj
    if cond.any():
        cp = np.zeros((H + 2, W + 2), dtype=bool)
        cp[1:-1, 1:-1] = cond
        interior = cond & cp[:-2, 1:-1] & cp[2:, 1:-1] & cp[1:-1, :-2] & cp[1:-1, 2:]
        if interior.any():
            gmax = 0.0
            sp = np.pad(stored, 1, mode="constant")
            rp = np.pad(region, 1, mode="constant", constant_values=-1)
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nbv = sp[1 + di:H + 1 + di, 1 + dj:W + 1 + dj]    # stored[i+di, j+dj]
                nbr = rp[1 + di:H + 1 + di, 1 + dj:W + 1 + dj]    # region[i+di, j+dj]
                same_kind = nbr == region                          # 同一种导体
                g = np.abs(nbv - stored)[interior & same_kind]
                if g.size:
                    gmax = max(gmax, float(g.max()))
            m["e0_grad_max"] = gmax
        else:
            m["e0_grad_max"] = 0.0
    else:
        m["e0_grad_max"] = 0.0
    m["e0_exact"] = bool(m["e0_grad_max"] == 0.0)

    # ⑤ stored 自身空气区 5 点离散 Laplace 残差（十字内点：4 邻居均为空气）
    ap = np.zeros((H + 2, W + 2), dtype=bool)
    ap[1:-1, 1:-1] = air_all
    air_interior = air_all & ap[:-2, 1:-1] & ap[2:, 1:-1] & ap[1:-1, :-2] & ap[1:-1, 2:]
    if air_interior.any():
        s = stored
        lap = 4.0 * s.copy()
        lap[:-1, :] -= s[1:, :]
        lap[1:, :] -= s[:-1, :]
        lap[:, :-1] -= s[:, 1:]
        lap[:, 1:] -= s[:, :-1]
        m["stored_laplace_max"] = float(np.abs(lap[air_interior]).max())
    else:
        m["stored_laplace_max"] = 0.0

    # ⑥ 额外诊断：stored 在本网格 FD 方程组下的残差（含 Dirichlet 邻居项）
    if A is not None and air_kept.any():
        r = A.dot(stored[air_kept]) - b
        m["stored_fd_residual_max"] = float(np.abs(r).max())
    else:
        m["stored_fd_residual_max"] = 0.0

    # ⑦ 修正几何对照：把"region 标空气但 stored 为导体值"的错标像素按 stored
    #    隐含几何还原为电极，再独立求解 —— 用于把 rel_l2 的两大误差源
    #    （region 图错标 vs stored 本身求解精度）分解开。
    if m["hidden_ground_px"] > 0 or m["hidden_hv_px"] > 0:
        region_fixed = region.copy()
        region_fixed[(region == AIR) & (stored == V_src)] = SRC
        region_fixed[(region == AIR) & (stored == V_hv)] = HV
        fd_f, info_f = solve_laplace_fd(region_fixed, V_src, V_hv,
                                        invalid_as="ground")
        air_f = region_fixed == AIR
        if air_f.any():
            df_ = (fd_f - stored)[air_f]
            sf_ = float(np.linalg.norm(stored[air_f]))
            m["rel_l2_air_fixedgeom"] = (float(np.linalg.norm(df_) / sf_)
                                         if sf_ > 0 else float(np.linalg.norm(df_)))
        else:
            m["rel_l2_air_fixedgeom"] = 0.0
        m["solve_residual_fixedgeom"] = info_f["solve_residual"]
    else:
        # 无错标，修正几何与原几何相同
        m["rel_l2_air_fixedgeom"] = m["rel_l2_air"]
        m["solve_residual_fixedgeom"] = m["solve_residual"]

    return m


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(
        description="2D 静电场数据集真值交叉校验（独立 FD 求解器）")
    ap.add_argument("--data-root", default="/workspace/retangle_data")
    ap.add_argument("--out-dir", default="/workspace/Attentiion_PINN")
    ap.add_argument("--limit", type=int, default=0,
                    help="只处理前 N 对（调试用），0=全量")
    ap.add_argument("--rel-l2-threshold", type=float, default=1e-3,
                    help="空气区相对 L2 PASS 阈值（默认 1e-3）")
    ap.add_argument("--sample-per-dir", type=int, default=0,
                    help="每工况目录抽样 N 对（0=全量；超时降级方案）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t_begin = time.time()

    pairs = collect_pairs(args.data_root)
    if not pairs:
        print(f"[ERROR] 在 {args.data_root} 下未找到任何数据对", file=sys.stderr)
        sys.exit(2)

    # 抽样模式：每目录取前 N 对
    sampling_mode = "full"
    if args.sample_per_dir > 0:
        by_dir = {}
        for p in pairs:
            by_dir.setdefault(p[0], []).append(p)
        pairs = []
        for d in sorted(by_dir):
            pairs.extend(by_dir[d][:args.sample_per_dir])
        sampling_mode = f"sampled_{args.sample_per_dir}_per_dir"

    total_all = len(collect_pairs(args.data_root))
    if args.limit > 0:
        pairs = pairs[:args.limit]
        sampling_mode = f"limit_{args.limit}"

    print(f"数据根目录: {args.data_root}")
    print(f"数据对总数: {total_all}，本次处理: {len(pairs)}（模式: {sampling_mode}）")
    print(f"相对 L2 阈值: {args.rel_l2_threshold:g}")
    sys.stdout.flush()

    records = []
    n_done = 0
    t0 = time.time()
    for k, (cond_dir, suffix, rpath, ppath, partner_ok) in enumerate(pairs):
        rec = {
            "condition": cond_dir,
            "suffix": suffix,
            "shape": "",
            "n_air": -1, "n_invalid": -1, "n_isolated_air": -1,
            "rel_l2_air": np.nan, "max_abs_diff_air": np.nan,
            "rel_l2_air_insul0": np.nan, "max_abs_diff_air_insul0": np.nan,
            "rel_l2_air_fixedgeom": np.nan, "solve_residual_fixedgeom": np.nan,
            "hidden_ground_px": -1, "hidden_hv_px": -1,
            "src_hv_adjacent_px": -1,
            "bc_src_exact": False, "bc_hv_exact": False, "bc_zero_exact": False,
            "e0_exact": False, "e0_grad_max": np.nan,
            "stored_laplace_max": np.nan,
            "stored_fd_residual_max": np.nan,
            "solve_residual": np.nan,
            "failed": True, "fail_reason": "",
        }
        try:
            if not partner_ok:
                raise RuntimeError("missing_partner_potential_csv")
            region = read_matrix(rpath)
            stored = read_matrix(ppath)
            if region.shape != stored.shape:
                raise RuntimeError(
                    f"shape_mismatch region{region.shape}_pot{stored.shape}")
            rec["shape"] = f"{region.shape[0]}x{region.shape[1]}"

            m = cross_validate_pair(region, stored)
            rec.update(m)

            if not np.isfinite(m["solve_residual"]) or \
                    m["solve_residual"] >= SOLVE_RESIDUAL_TOL:
                raise RuntimeError(
                    f"solve_residual_too_high({m['solve_residual']:.3e})")
            if not np.isfinite(m["rel_l2_air"]):
                raise RuntimeError("nonfinite_rel_l2")
            rec["failed"] = False
        except Exception as e:  # noqa: BLE001 —— 逐对隔离故障，不让单对中断全量
            rec["fail_reason"] = type(e).__name__ + ": " + str(e)[:300]
        records.append(rec)

        n_done += 1
        if n_done % 50 == 0 or n_done == len(pairs):
            el = time.time() - t0
            rate = el / n_done
            eta_min = rate * (len(pairs) - n_done) / 60.0
            print(f"[{n_done:5d}/{len(pairs)}] 已用 {el/60.0:6.1f} min，"
                  f"{rate:5.2f} s/对，预计剩余 {eta_min:5.1f} min", flush=True)

    duration = time.time() - t_begin
    df = pd.DataFrame.from_records(records)
    csv_path = os.path.join(args.out_dir, "verification_report.csv")
    df.to_csv(csv_path, index=False)

    # ---------------- 汇总 ----------------
    ok = df[~df["failed"]]
    n_total, n_failed = len(df), int(df["failed"].sum())

    def stats_of(series):
        v = pd.Series(series).dropna().to_numpy(dtype=np.float64)
        if v.size == 0:
            return {"min": None, "mean": None, "max": None,
                    "p50": None, "p90": None, "p99": None}
        return {
            "min": float(v.min()), "mean": float(v.mean()), "max": float(v.max()),
            "p50": float(np.percentile(v, 50)),
            "p90": float(np.percentile(v, 90)),
            "p99": float(np.percentile(v, 99)),
        }

    def stats(col):
        if len(ok) == 0:
            return {"min": None, "mean": None, "max": None,
                    "p50": None, "p90": None, "p99": None}
        return stats_of(ok[col])

    n_ok = len(ok)
    n_bc_src = int(ok["bc_src_exact"].sum()) if n_ok else 0
    n_bc_hv = int(ok["bc_hv_exact"].sum()) if n_ok else 0
    n_bc_zero = int(ok["bc_zero_exact"].sum()) if n_ok else 0
    n_e0 = int(ok["e0_exact"].sum()) if n_ok else 0
    n_rel_l2_pass = int((ok["rel_l2_air"] < args.rel_l2_threshold).sum()) if n_ok else 0
    n_isolated = int((df["n_isolated_air"].fillna(0) > 0).sum())
    n_with_invalid = int((df["n_invalid"].fillna(0) > 0).sum())
    n_hidden_ground = int((df["hidden_ground_px"].fillna(0) > 0).sum())
    n_hidden_hv = int((df["hidden_hv_px"].fillna(0) > 0).sum())
    tot_hidden_ground_px = int(df["hidden_ground_px"].fillna(0).sum())
    tot_hidden_hv_px = int(df["hidden_hv_px"].fillna(0).sum())
    n_src_hv_adj = int((df["src_hv_adjacent_px"].fillna(0) > 0).sum())
    tot_src_hv_adj_px = int(df["src_hv_adjacent_px"].fillna(0).sum())

    # 0 值对（21 个 boundary21/_50 系列）单独统计：实证 ground 语义 vs 绝缘语义
    inv_mask = df["n_invalid"].fillna(0) > 0
    zero_pairs_summary = None
    if inv_mask.any():
        zi = df[inv_mask & ~df["failed"]]
        zero_pairs_summary = {
            "n_pairs": int(inv_mask.sum()),
            "rel_l2_air_ground_semantics": stats_of(zi["rel_l2_air"]),
            "rel_l2_air_insulate_semantics": stats_of(zi["rel_l2_air_insul0"]),
            "max_abs_diff_air_insulate_semantics": stats_of(
                zi["max_abs_diff_air_insul0"]),
        }

    # 误差归因分解（基于修正几何对照 rel_l2_air_fixedgeom）
    n_rel_l2_fixed_pass = None
    attribution = None
    if n_ok:
        okf = ok["rel_l2_air_fixedgeom"]
        n_rel_l2_fixed_pass = int((okf < args.rel_l2_threshold).sum())
        n_fd_exact = int((ok["rel_l2_air"] < 1e-5).sum())
        n_mislabel_limited = int(((ok["rel_l2_air"] >= args.rel_l2_threshold) &
                                  (okf < args.rel_l2_threshold)).sum())
        n_accuracy_limited = int(((ok["rel_l2_air"] >= args.rel_l2_threshold) &
                                  (okf >= args.rel_l2_threshold)).sum())
        attribution = {
            "n_fd_exact_rel_l2_lt_1e-5": n_fd_exact,
            "n_pass_by_threshold": n_rel_l2_pass,
            "n_mislabel_limited_pass_after_geometry_fix": n_mislabel_limited,
            "n_accuracy_limited_still_fail_after_geometry_fix": n_accuracy_limited,
        }

    reasons = []
    if n_failed > 0:
        reasons.append(
            f"{n_failed}/{n_total} 对在校验流程中失败（求解/读取/配对异常）")
        for rsn, cnt in df[df["failed"]]["fail_reason"].str.split(":").str[0].value_counts().items():
            reasons.append(f"  - 失败类型 {rsn}: {cnt} 对")
    if n_ok > 0 and n_bc_src < n_ok:
        reasons.append(f"源极 BC 非精确: {n_ok - n_bc_src}/{n_ok} 对")
    if n_ok > 0 and n_bc_hv < n_ok:
        reasons.append(f"HV BC 非精确: {n_ok - n_bc_hv}/{n_ok} 对")
    if n_ok > 0 and n_bc_zero < n_ok:
        reasons.append(f"0值像素(接地) BC 非精确: {n_ok - n_bc_zero}/{n_ok} 对")
    if n_ok > 0 and n_e0 < n_ok:
        reasons.append(f"导体 E=0 非精确: {n_ok - n_e0}/{n_ok} 对")
    if n_ok > 0 and n_rel_l2_pass < n_ok:
        worst = ok.loc[ok["rel_l2_air"].idxmax()] if n_ok else None
        reasons.append(
            f"空气区相对 L2 >= {args.rel_l2_threshold:g}: "
            f"{n_ok - n_rel_l2_pass}/{n_ok} 对"
            + (f"（最差: {worst['condition']}/{worst['suffix']}, "
               f"rel_l2={worst['rel_l2_air']:.3e}）" if worst is not None else ""))

    verdict = "PASS" if not reasons else "FAIL"

    summary = {
        "meta": {
            "script": os.path.abspath(__file__),
            "data_root": os.path.abspath(args.data_root),
            "out_dir": os.path.abspath(args.out_dir),
            "rel_l2_threshold": args.rel_l2_threshold,
            "sampling_mode": sampling_mode,
            "pairs_total_in_dataset": total_all,
            "pairs_processed": n_total,
            "sampling_rate": (n_total / total_all) if total_all else None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": round(duration, 1),
        },
        "total_pairs": n_total,
        "n_success": n_ok,
        "n_failed": n_failed,
        "rel_l2_air": stats("rel_l2_air"),
        "max_abs_diff_air": stats("max_abs_diff_air"),
        "rel_l2_air_fixedgeom": stats("rel_l2_air_fixedgeom"),
        "error_attribution": attribution,
        "bc_src_exact_rate": (n_bc_src / n_ok) if n_ok else None,
        "bc_hv_exact_rate": (n_bc_hv / n_ok) if n_ok else None,
        "bc_zero_exact_rate": (n_bc_zero / n_ok) if n_ok else None,
        "e0_exact_rate": (n_e0 / n_ok) if n_ok else None,
        "rel_l2_pass_rate": (n_rel_l2_pass / n_ok) if n_ok else None,
        "stored_laplace_max": stats("stored_laplace_max"),
        "stored_fd_residual_max": stats("stored_fd_residual_max"),
        "solve_residual": stats("solve_residual"),
        "n_pairs_with_invalid_pixels": n_with_invalid,
        "n_pairs_with_isolated_air": n_isolated,
        "n_pairs_with_hidden_ground_mislabel": n_hidden_ground,
        "n_pairs_with_hidden_hv_mislabel": n_hidden_hv,
        "total_hidden_ground_px": tot_hidden_ground_px,
        "total_hidden_hv_px": tot_hidden_hv_px,
        "n_pairs_with_src_hv_adjacency": n_src_hv_adj,
        "total_src_hv_adjacent_px": tot_src_hv_adj_px,
        "zero_value_pairs_ground_vs_insulate": zero_pairs_summary,
        "verdict": verdict,
        "verdict_reasons": reasons,
        "notes": [
            "无效像素(值 0)的处理——实证修正：任务原假设为绝缘(零通量)，但实测表明数据生成器把 "
            "0 值像素当作 0V 接地电极(与源极同电位)：①全部 0 值像素上 stored 精确=0V；"
            "②按接地语义求解，boundary21/_50 系列 21 对 rel_l2 ~1e-6(stored 即本网格 5 点 FD 解)；"
            "③按绝缘语义求解，其中接触空气的对 rel_l2 高达 1.4~5.7。因此主指标采用接地语义，"
            "同时保留绝缘语义对照列(rel_l2_air_insul0)。",
            "0 值像素仅出现于 results_reduction_fixed_boundary21 目录的 21 个 *_50 后缀对(1.6%)。",
            "图像边界外的邻居按绝缘（零通量）处理。",
            "被完全隔离、不接触任何 Dirichlet 电极的空气连通域电位不定，统一填 0 并计入 n_isolated_air。",
            "CSV 以 float64 读入；rel_l2/max_diff 仅在 region==2 像素上计算。",
            "stored_fd_residual_max 为 stored 场代入本网格 5 点 FD 方程组(A·x=b)的最大残差，"
            "直接反映 stored 是否为本网格 FD 方程的解。",
            "hidden_ground_px/hidden_hv_px 检测 region 图标为空气(2)但 stored 为导体值(0/60000)的"
            "像素数——即 region 图与 stored 场生成几何不一致（隐藏电极错标）。",
            "E=0 检查语义：仅对'邻居与中心为同一种导体'的十字内点方向计差分。"
            "实测 391/1314 对几何中源极与 HV 电极直接相邻（src_hv_adjacent_px>0），"
            "接触处 0V↔60000V 跳变是物理正常的界面不连续，不属于导体内部 E≠0。",
        ],
    }
    json_path = os.path.join(args.out_dir, "verification_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---------------- 摘要打印 ----------------
    print("\n==================== 校验汇总 ====================")
    print(f"处理对数     : {n_total}（成功 {n_ok}，失败 {n_failed}）")
    if n_ok:
        r = summary["rel_l2_air"]
        d = summary["max_abs_diff_air"]
        print(f"空气区相对L2 : min={r['min']:.3e}  mean={r['mean']:.3e}  "
              f"p50={r['p50']:.3e}  p90={r['p90']:.3e}  max={r['max']:.3e}")
        print(f"空气区max|Δ| : min={d['min']:.4f}V  mean={d['mean']:.4f}V  "
              f"p50={d['p50']:.4f}V  p90={d['p90']:.4f}V  max={d['max']:.4f}V")
        print(f"BC 精确率    : 源极 {n_bc_src}/{n_ok}，HV {n_bc_hv}/{n_ok}，"
              f"0值(接地) {n_bc_zero}/{n_ok}")
        print(f"导体 E=0     : {n_e0}/{n_ok}")
        sl = summary["stored_laplace_max"]
        sf = summary["stored_fd_residual_max"]
        sr = summary["solve_residual"]
        print(f"stored 自身Laplace残差 max : min={sl['min']:.3e}  mean={sl['mean']:.3e}  max={sl['max']:.3e} V")
        print(f"stored 代入FD方程组残差 max: min={sf['min']:.3e}  mean={sf['mean']:.3e}  max={sf['max']:.3e} V")
        print(f"FD求解器相对残差           : max={sr['max']:.3e}（阈值 {SOLVE_RESIDUAL_TOL:g}）")
        print(f"含无效0值像素的对: {n_with_invalid}；含被隔离空气的对: {n_isolated}")
        print(f"region错标(隐藏接地电极): {n_hidden_ground} 对 / 共 {tot_hidden_ground_px} px；"
              f"隐藏HV电极: {n_hidden_hv} 对 / 共 {tot_hidden_hv_px} px")
        print(f"src-HV 直接相邻(几何粘连): {n_src_hv_adj} 对 / 共 {tot_src_hv_adj_px} px")
        if zero_pairs_summary:
            zg = zero_pairs_summary["rel_l2_air_ground_semantics"]
            zi2 = zero_pairs_summary["rel_l2_air_insulate_semantics"]
            print(f"0值对({zero_pairs_summary['n_pairs']}个) 接地语义 rel_l2: "
                  f"max={zg['max']:.3e} | 绝缘语义 rel_l2: min={zi2['min']:.3e} "
                  f"max={zi2['max']:.3e}")
        n_exact = int((ok["rel_l2_air"] < 1e-5).sum())
        print(f"其中 rel_l2 < 1e-5（本网格FD精确解级）: {n_exact}/{n_ok} 对")
        rf = summary["rel_l2_air_fixedgeom"]
        print(f"修正几何后 rel_l2: min={rf['min']:.3e}  mean={rf['mean']:.3e}  "
              f"p50={rf['p50']:.3e}  p90={rf['p90']:.3e}  max={rf['max']:.3e}")
        if attribution:
            print(f"误差归因: FD精确 {attribution['n_fd_exact_rel_l2_lt_1e-5']} 对 | "
                  f"原口径达标 {attribution['n_pass_by_threshold']} 对 | "
                  f"仅几何错标所致(修正后达标) {attribution['n_mislabel_limited_pass_after_geometry_fix']} 对 | "
                  f"修正后仍超标(求解精度所致) {attribution['n_accuracy_limited_still_fail_after_geometry_fix']} 对")
    print(f"\n>>> 裁决: {verdict}")
    for rsn in reasons:
        print(f"    {rsn}")
    print(f"\n明细报告: {csv_path}")
    print(f"汇总报告: {json_path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
