#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval.py — 2D 静电场 PINN 独立评估脚本（阶段二 Task 3）

加载训练权重，对指定数据集（目录或数据根目录，走 ElectrostaticDataset
递归加载）批量前向，用 eval_utils.compute_all_metrics 计算六项统一指标
（mse / relative_l2 / max_abs_error / bc_violation / laplace_residual /
equipotential_residual），打印表格并可选落盘 JSON 报告。

支持 --split-file 传入 train/val 划分 manifest（取 val 侧）：
  * manifest 约定为 JSON 且含 "train"/"val" 键（split_utils.py 并行任务
    尚未定稿，此处做宽松兼容）——val 条目可为字符串路径（region CSV
    路径，绝对/相对均可），也可为含路径值的 dict（取其全部字符串值）；
    先按规范化绝对路径匹配 dataset.file_pairs 的 region 路径，失败再按
    文件名匹配。
  * 无法解析（文件不存在 / 非 JSON / 无 val 键 / 零匹配）时报错退出，
    提示先运行 train.py 生成 manifest，或去掉 --split-file 全量评估。

用法示例：
    python3 eval.py --weights dual_encoder_fno_model_v2.pth \
        --data-path /workspace/retangle_data/results_reduction_fixed_boundary \
        --limit 8 --out report.json
"""

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import ElectrostaticDataset
from model import DualEncoderFNODecoder
from eval_utils import compute_all_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate DualEncoder-FNO electrostatic PINN (unified metrics)')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to the .pth state_dict to evaluate')
    parser.add_argument('--data-path', type=str,
                        default='/workspace/retangle_data/results_reduction_fixed_boundary',
                        help='Data directory (single condition) or data root '
                             '(recursive discovery, same as train.py)')
    parser.add_argument('--split-file', type=str, default=None,
                        help='Optional train/val split manifest (JSON with '
                             '"train"/"val" keys); evaluates the val subset')
    parser.add_argument('--limit', type=int, default=0,
                        help='Evaluate only the first N samples (0 = all)')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--model-type', type=str, default='v2', choices=['v2'],
                        help='Model architecture (v2 = DualEncoderFNODecoder; '
                             'more reserved for future variants)')
    parser.add_argument('--modes', type=int, default=32,
                        help='Spectral modes of DualEncoderFNODecoder (model.py)')
    parser.add_argument('--width', type=int, default=32,
                        help='Width of DualEncoderFNODecoder (model.py)')
    parser.add_argument('--no-edt', action='store_true',
                        help='Match no-EDT weights (RoPE y/x axes only)')
    parser.add_argument('--norm-factor', type=float, default=60000.0,
                        help='Potential normalization factor (dataset.py)')
    parser.add_argument('--out', type=str, default=None,
                        help='Optional path to write the JSON report')
    return parser.parse_args()


def fail(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(2)


def load_val_entries(split_file):
    """宽松解析 split manifest：JSON 且含 'val' 键。"""
    try:
        with open(split_file, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        fail(f"无法解析 --split-file '{split_file}'：{e}\n"
             f"        请先运行 train.py 生成 split manifest（split_utils.py），"
             f"或去掉 --split-file 进行全量评估。")
    if not isinstance(manifest, dict) or 'val' not in manifest:
        fail(f"--split-file '{split_file}' 中未找到 'val' 键（期望 JSON 键 "
             f"train/val）。\n        请先运行 train.py 生成 manifest，"
             f"或去掉 --split-file 进行全量评估。")
    val = manifest['val']
    if not isinstance(val, list):
        fail(f"--split-file 的 'val' 期望为列表，实际为 {type(val).__name__}。")
    return val


def match_val_indices(dataset, val_entries):
    """把 manifest val 条目匹配到 dataset.file_pairs 的 region 路径。

    条目可为字符串路径，或含路径值的 dict（收集其全部字符串值）。
    先按规范化绝对路径精确匹配，再按 basename 兜底；返回
    (sorted 去重索引, 未匹配条目列表)。
    """

    def norm(p):
        return os.path.normpath(os.path.abspath(p))

    by_path = {norm(r): i for i, (r, _) in enumerate(dataset.file_pairs)}
    by_name = {}
    for i, (r, _) in enumerate(dataset.file_pairs):
        by_name.setdefault(os.path.basename(r), []).append(i)

    candidates = []
    for e in val_entries:
        if isinstance(e, str):
            candidates.append(e)
        elif isinstance(e, dict):
            candidates.extend(v for v in e.values() if isinstance(v, str))

    matched, unmatched = [], []
    for c in candidates:
        i = by_path.get(norm(c))
        if i is None:
            hits = by_name.get(os.path.basename(c), [])
            i = hits[0] if len(hits) == 1 else None
        if i is None:
            unmatched.append(c)
        elif i not in matched:
            matched.append(i)
    return sorted(matched), unmatched


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.isfile(args.weights):
        fail(f"权重文件不存在：{args.weights}")
    if args.model_type != 'v2':
        fail(f"未知 --model-type：{args.model_type}（当前仅支持 v2，其余为预留）")

    dataset = ElectrostaticDataset(data_dir=args.data_path,
                                   norm_factor=args.norm_factor)
    if len(dataset) == 0:
        fail(f"在 {args.data_path} 下未找到任何 (region, potential) 数据对，"
             f"请检查 --data-path。")

    # ---- 样本选择：全量 / manifest val 子集 / limit 截断 ----
    indices = list(range(len(dataset)))
    split_desc = f"全量（{len(dataset)} 对）"
    if args.split_file:
        val_entries = load_val_entries(args.split_file)
        matched, unmatched = match_val_indices(dataset, val_entries)
        if not matched:
            fail(f"--split-file 的 val 条目（{len(val_entries)} 条）未能匹配到任何"
                 f"数据对（按 region 路径与文件名）。请检查 manifest 中的路径是否"
                 f"属于 --data-path，或先运行 train.py 生成 manifest。")
        indices = matched
        split_desc = (f"manifest val 子集（{len(val_entries)} 条 → 匹配 "
                      f"{len(matched)} 对"
                      + (f"，未匹配 {len(unmatched)} 条" if unmatched else "")
                      + "）")
    if args.limit and args.limit > 0:
        indices = indices[:args.limit]
        split_desc += f"，limit 截断 → {len(indices)} 对"
    if not indices:
        fail("选中的评估样本为空（--limit / --split-file 组合下无剩余样本）。")

    eval_set = Subset(dataset, indices) if len(indices) != len(dataset) else dataset
    loader = DataLoader(eval_set, batch_size=args.batch_size,
                        shuffle=False, num_workers=0)

    # ---- 模型与权重 ----
    model = DualEncoderFNODecoder(
        modes=args.modes, width=args.width, use_edt=not args.no_edt).to(device)
    state_dict = torch.load(args.weights, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except (RuntimeError, KeyError) as e:
        fail(f"权重加载失败（参数与 DualEncoderFNODecoder(modes={args.modes}, "
             f"width={args.width}) 不匹配）：{e}\n"
             f"        请用 --modes/--width 指定与训练时一致的构造参数。")
    model.eval()

    print(f"Using device: {device}")
    print(f"model-type : {args.model_type} (DualEncoderFNODecoder, "
          f"modes={args.modes}, width={args.width})")
    print(f"weights    : {os.path.abspath(args.weights)}")
    print(f"data-path  : {os.path.abspath(args.data_path)}")
    print(f"split-file : {args.split_file if args.split_file else '(none)'}")
    print(f"samples    : {split_desc}")
    print(f"batch-size : {args.batch_size}，共 {len(loader)} 个 batch")
    print(f"norm-factor: {args.norm_factor}")

    # ---- 评估循环：逐 batch 指标（已是逐样本平均）按样本数加权累计 ----
    sums = {}
    total = 0
    t0 = time.time()
    with torch.no_grad():
        for bi, (inputs, targets, masks, pos_dist) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            pos_dist = pos_dist.to(device)

            outputs = model(inputs, pos_dist)
            batch_metrics = compute_all_metrics(outputs, targets, inputs, masks)

            bsz = inputs.size(0)
            total += bsz
            for k, v in batch_metrics.items():
                sums[k] = sums.get(k, 0.0) + v * bsz
            if (bi + 1) % 25 == 0 or (bi + 1) == len(loader):
                print(f"  [{bi + 1}/{len(loader)}] 已评估 {total}/{len(indices)} 样本"
                      f"（{time.time() - t0:.1f}s）", flush=True)

    metrics = {k: v / total for k, v in sums.items()}

    # ---- 表格输出 ----
    print("\n" + "=" * 64)
    print(" 评估结果（batch 平均，归一化电位 [0,1] 口径）")
    print("=" * 64)
    print(f" {'metric':<24}{'value':>16}")
    print("-" * 64)
    for k, v in metrics.items():
        print(f" {k:<24}{v:>16.6e}")
    print("-" * 64)
    print(f" samples={total}  batches={len(loader)}  "
          f"elapsed={time.time() - t0:.1f}s")

    # ---- 可选 JSON 报告 ----
    if args.out:
        report = {
            "meta": {
                "weights": os.path.abspath(args.weights),
                "data_path": os.path.abspath(args.data_path),
                "split_file": os.path.abspath(args.split_file) if args.split_file else None,
                "split_desc": split_desc,
                "model_type": args.model_type,
                "modes": args.modes,
                "width": args.width,
                "norm_factor": args.norm_factor,
                "n_samples": total,
                "batch_size": args.batch_size,
                "limit": args.limit,
                "device": str(device),
                "elapsed_sec": round(time.time() - t0, 1),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "metrics": metrics,
        }
        out_dir = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nJSON 报告已写入: {os.path.abspath(args.out)}")


if __name__ == '__main__':
    main()
