#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pretrain_check.py — 训练前数据巡检可视化（无需模型权重）。

从数据根目录随机抽 N 个样本，每个画 2x3 六联图：
  region RGB(蓝=接地/灰=空气/红=HV/白=填充) | 目标电位 | EDT 到接地 | EDT 到 HV
  | region 唯一值文本 | 有效性掩码
用于正式训练前肉眼确认：标注 v2（内部接地=1、无 0 值）、归一化目标 ∈ [0,1]、
EDT 距离轴与掩码正确。

用法:
  python pretrain_check.py --data-path ../data --samples 4 --out ../runs/pretrain_check
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import ElectrostaticDataset  # noqa: E402

# 与 train.py log_val_visualization 同口径: 0=pad 白, 1=接地 蓝, 2=空气 灰, 3=HV 红
PALETTE = np.array([[255, 255, 255], [66, 133, 244], [158, 158, 158],
                    [234, 67, 53]], dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-path', default='../data')
    ap.add_argument('--samples', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='../runs/pretrain_check')
    args = ap.parse_args()

    ds = ElectrostaticDataset(data_dir=args.data_path)
    print(f'dataset: {len(ds)} pairs, {len(set(ds.conditions))} conditions')
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(ds), size=min(args.samples, len(ds)), replace=False)

    n_zero_region = 0
    for i, idx in enumerate(idxs):
        x, y, m, p = ds[int(idx)]
        x, y, m, p = x.numpy(), y.numpy(), m.numpy(), p.numpy()
        onehot = x                                    # (3,H,W)
        region_code = np.argmax(onehot, axis=0).astype(np.int8) + 1
        region_code[onehot.sum(axis=0) < 0.5] = 0     # 填充 → 0
        uniq = sorted(np.unique(region_code).tolist())
        if 0 in uniq and m[0].sum() == m[0].size:
            n_zero_region += 1                        # 无填充却含 0 → 异常
        cond = ds.get_condition(int(idx))

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes[0, 0].imshow(PALETTE[region_code], interpolation='nearest')
        axes[0, 0].set_title(f'Region uniq={uniq}\n(blue=gnd grey=air red=HV white=pad)')
        im = axes[0, 1].imshow(y[0], cmap='viridis', vmin=0, vmax=1)
        axes[0, 1].set_title(f'Target φ/60000\nrange=[{y[0][m[0]>0.5].min():.3f},'
                             f'{y[0][m[0]>0.5].max():.3f}]')
        fig.colorbar(im, ax=axes[0, 1], fraction=0.046)
        im = axes[0, 2].imshow(p[0], cmap='cividis')
        axes[0, 2].set_title('EDT to ground (px)')
        fig.colorbar(im, ax=axes[0, 2], fraction=0.046)
        im = axes[1, 0].imshow(p[1], cmap='cividis')
        axes[1, 0].set_title('EDT to HV (px)')
        fig.colorbar(im, ax=axes[1, 0], fraction=0.046)
        axes[1, 1].imshow(m[0], cmap='gray', interpolation='nearest')
        axes[1, 1].set_title(f'Valid mask (valid={int(m[0].sum())} px)')
        # 内部接地像素统计
        gnd_internal = int(((region_code == 1) & (m[0] > 0.5)).sum())
        axes[1, 2].text(0.05, 0.75, f'sample #{int(idx)}\ncondition: {cond}\n'
                        f'unique labels: {uniq}\n'
                        f'ground px (incl. frame): {gnd_internal}\n'
                        f'HV px: {int((region_code == 3).sum())}\n'
                        f'air px: {int((region_code == 2).sum())}\n'
                        f'pad px: {int((region_code == 0).sum())}',
                        transform=axes[1, 2].transAxes, fontsize=11,
                        verticalalignment='center', family='monospace')
        axes[1, 2].set_axis_off()
        for ax in axes.flat[:5]:
            ax.set_axis_off()
        fig.suptitle(f'Pretrain data check | {cond} | idx={int(idx)}')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = os.path.join(args.out, f'sample_{i}_idx{int(idx)}.png')
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f'saved {out} | uniq={uniq} | cond={cond}')

    print(f'\nnote: 有效区内出现 0 标签的样本数 = {n_zero_region}（应为 0）')
    print(f'output dir: {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()
