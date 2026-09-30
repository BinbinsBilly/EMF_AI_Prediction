"""Train/val split utilities for the electrostatic PINN dataset.

Provides deterministic, seed-controlled splits of an ElectrostaticDataset
into train/val index lists plus a JSON manifest on disk so a split can be
inspected and reproduced exactly across runs.

Split modes
-----------
random : stratified per condition. Each condition's sample indices are
    shuffled and ~val_ratio of them go to val. Every condition with >= 2
    samples keeps at least 1 train and 1 val sample; single-sample
    conditions go entirely to train. Per-condition val counts are then
    adjusted (largest-remainder) so the GLOBAL val ratio lands as close
    to val_ratio as the per-condition constraints allow.
group  : whole conditions are assigned to one side. Condition names are
    shuffled and greedily packed into the val group until its sample
    count reaches ~val_ratio * total; condition integrity always wins
    over ratio precision. Every sample of a condition stays on one side.
"""

import json
import math
import os
import random
from datetime import datetime


def make_split(dataset, val_ratio=0.2, mode='random', seed=42):
    """Split dataset indices into train/val lists.

    Args:
        dataset: ElectrostaticDataset (uses .conditions, 1:1 with indices).
        val_ratio: target fraction of samples in val.
        mode: 'random' (stratified per condition) or 'group'
            (whole-condition assignment).
        seed: RNG seed; identical (mode, seed, val_ratio, dataset) inputs
            always produce identical splits.

    Returns:
        (train_indices, val_indices): sorted int lists into `dataset`.
    """
    by_cond = {}
    for idx in range(len(dataset)):
        by_cond.setdefault(dataset.get_condition(idx), []).append(idx)

    rng = random.Random(seed)
    if mode == 'random':
        train_indices, val_indices = _split_random(by_cond, val_ratio, rng)
    elif mode == 'group':
        train_indices, val_indices = _split_group(by_cond, val_ratio, rng)
    else:
        raise ValueError(f"Unknown split mode: {mode!r} (expected 'random' or 'group')")

    return sorted(train_indices), sorted(val_indices)


def _split_random(by_cond, val_ratio, rng):
    """Stratified split: shuffle within each condition, take ~val_ratio
    to val under per-condition constraints (val>=1 & train>=1 for
    multi-sample conditions; singles to train)."""
    sizes = {cond: len(idxs) for cond, idxs in by_cond.items()}
    total = sum(sizes.values())

    alloc = {}
    for cond, n in sizes.items():
        if n < 2:
            alloc[cond] = 0  # single-sample condition: all train
        else:
            alloc[cond] = max(1, min(n - 1, int(math.floor(n * val_ratio))))

    # Largest-remainder adjustment so the global val count hits
    # round(val_ratio * total) as closely as the constraints allow.
    target = int(round(val_ratio * total))
    current = sum(alloc.values())
    if current < target:
        # Grow conditions that are most under-entitled first
        order = sorted(
            (c for c in alloc if sizes[c] >= 2 and alloc[c] < sizes[c] - 1),
            key=lambda c: (-(sizes[c] * val_ratio - alloc[c]), c))
        for cond in order:
            if current >= target:
                break
            alloc[cond] += 1
            current += 1
    elif current > target:
        # Shrink conditions that are most over-entitled first
        order = sorted(
            (c for c in alloc if alloc[c] > 1),
            key=lambda c: (alloc[c] - sizes[c] * val_ratio, c))
        for cond in order:
            if current <= target:
                break
            alloc[cond] -= 1
            current -= 1

    train_indices, val_indices = [], []
    for cond in sorted(by_cond):  # sorted -> deterministic RNG consumption
        idxs = list(by_cond[cond])
        rng.shuffle(idxs)
        n_val = alloc[cond]
        val_indices.extend(idxs[:n_val])
        train_indices.extend(idxs[n_val:])
    return train_indices, val_indices


def _split_group(by_cond, val_ratio, rng):
    """Whole-condition split: shuffle condition names, greedily pack
    conditions into val until ~val_ratio of samples are covered."""
    conditions = sorted(by_cond)
    if len(conditions) < 2:
        raise ValueError(
            "group split requires >= 2 conditions, got "
            f"{len(conditions)} ({conditions or 'none'})")

    total = sum(len(by_cond[c]) for c in conditions)
    target = int(round(val_ratio * total))

    order = list(conditions)
    rng.shuffle(order)

    val_conds, train_conds = [], []
    val_samples = 0
    for cond in order:
        n = len(by_cond[cond])
        if val_samples + n <= target:
            val_conds.append(cond)
            val_samples += n
        else:
            train_conds.append(cond)

    # Guarantee a non-empty val group even if every condition is larger
    # than the target (move the smallest train condition over), and a
    # non-empty train group in the degenerate all-val case.
    if not val_conds:
        smallest = min(train_conds, key=lambda c: (len(by_cond[c]), c))
        val_conds.append(smallest)
        train_conds.remove(smallest)
    if not train_conds:
        largest = max(val_conds, key=lambda c: (len(by_cond[c]), c))
        train_conds.append(largest)
        val_conds.remove(largest)

    train_indices = [i for c in train_conds for i in by_cond[c]]
    val_indices = [i for c in val_conds for i in by_cond[c]]
    return train_indices, val_indices


def save_split_manifest(path, dataset, train_indices, val_indices,
                        mode, seed, val_ratio):
    """Write the split to `path` as JSON.

    The manifest stores the FULL logical split (region file absolute path
    + condition per entry); runtime-only knobs such as limit_samples are
    not baked in, so the same manifest can reproduce the split at any
    truncation level.

    Returns the manifest dict that was written."""
    def entry(idx):
        region_path, _ = dataset.file_pairs[idx]
        return {
            'region': os.path.abspath(region_path),
            'condition': dataset.get_condition(idx),
        }

    manifest = {
        'seed': seed,
        'mode': mode,
        'val_ratio': val_ratio,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'total_samples': len(dataset),
        'num_train': len(train_indices),
        'num_val': len(val_indices),
        'train': [entry(i) for i in train_indices],
        'val': [entry(i) for i in val_indices],
    }
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def load_split_manifest(path):
    """Load a manifest written by save_split_manifest.

    Returns:
        (train_entries, val_entries): lists of (region_path, condition)
        tuples, in manifest order."""
    with open(path) as f:
        manifest = json.load(f)

    def entries(key):
        return [(e['region'], e['condition']) for e in manifest.get(key, [])]

    return entries('train'), entries('val')


def indices_from_manifest(dataset, manifest):
    """Map manifest entries back to dataset indices.

    Args:
        dataset: ElectrostaticDataset whose file_pairs are searched for
            each entry's region path (exact match on the normalized
            absolute path).
        manifest: the (train_entries, val_entries) tuple returned by
            load_split_manifest, or a full manifest dict with 'train' /
            'val' entry lists.

    Returns:
        (train_indices, val_indices): int lists into `dataset`.

    Raises:
        ValueError: if any region path in the manifest is not present in
            dataset.file_pairs."""
    if isinstance(manifest, dict):
        train_entries = [(e['region'], e['condition']) for e in manifest.get('train', [])]
        val_entries = [(e['region'], e['condition']) for e in manifest.get('val', [])]
    else:
        train_entries, val_entries = manifest

    path_to_idx = {}
    for idx, (region_path, _) in enumerate(dataset.file_pairs):
        path_to_idx[os.path.normpath(os.path.abspath(region_path))] = idx

    def to_indices(entries, split_name):
        indices, missing = [], []
        for region_path, _condition in entries:
            key = os.path.normpath(os.path.abspath(region_path))
            if key in path_to_idx:
                indices.append(path_to_idx[key])
            else:
                missing.append(region_path)
        if missing:
            raise ValueError(
                f"{len(missing)} region file(s) from the {split_name} split "
                f"not found in dataset (data_dir mismatch?), e.g. {missing[:3]}")
        return indices

    return to_indices(train_entries, 'train'), to_indices(val_entries, 'val')
