import argparse
import contextlib
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import ElectrostaticDataset
from eval_utils import compute_all_metrics, erode_mask  # noqa: F401 (erode_mask re-exported for visualize.py)
from model import DualEncoderFNODecoder
from split_utils import (indices_from_manifest, load_split_manifest,
                         make_split, save_split_manifest)


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    return device


def parse_args():
    parser = argparse.ArgumentParser(description='Train DualEncoder-FNO electrostatic PINN')
    parser.add_argument('--data-path', type=str,
                        default='/workspace/retangle_data/results_reduction_fixed_boundary',
                        help='Directory containing input/target CSV pairs')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--modes', type=int, default=32)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--lambda-phy', type=float, default=1.0)
    parser.add_argument('--lambda-grad', type=float, default=0.1)
    parser.add_argument('--lambda-bc', type=float, default=2.0)
    parser.add_argument('--lambda-eq', type=float, default=5.0)
    parser.add_argument('--grad-accum', type=int, default=1,
                        help='Accumulate gradients over N batches before each optimizer step')
    parser.add_argument('--limit-samples', type=int, default=0,
                        help='Use only the first N training samples (0 = all); '
                             'does not affect the val split')
    parser.add_argument('--save-dir', type=str,
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help='Directory to save model weights')
    parser.add_argument('--log-dir', type=str, default='runs/DualEncoder_FNO_v2',
                        help='TensorBoard log directory')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Fraction of samples placed in the validation split')
    parser.add_argument('--split-mode', type=str, default='random',
                        choices=['random', 'group'],
                        help='random: stratified split per condition; '
                             'group: whole conditions assigned to one side')
    parser.add_argument('--split-seed', type=int, default=42,
                        help='Seed controlling the train/val split RNG')
    parser.add_argument('--split-file', type=str, default='',
                        help='Path to an existing split manifest to load '
                             '(takes priority over computing a fresh split)')
    parser.add_argument('--split-out', type=str, default='',
                        help='Where to write the split manifest '
                             '(default: <log-dir>/split_manifest.json)')
    parser.add_argument('--vis-interval', type=int, default=1,
                        help='Write val sample figures + error histogram '
                             'every N epochs')
    parser.add_argument('--vis-samples', type=int, default=2,
                        help='Number of validation samples (first K) shown '
                             'in the per-interval quad figures and pooled '
                             'error histogram')
    parser.add_argument('--run-name', type=str, default='',
                        help='TensorBoard run directory name under --log-dir '
                             '(default: auto v2_<timestamp>)')
    parser.add_argument('--tag', type=str, default='',
                        help='Optional tag appended to the run name '
                             '(<run-name>_<tag>)')
    return parser.parse_args()


def build_dataloaders(args):
    """Build train/val DataLoaders over one shared ElectrostaticDataset.

    The split (train/val index lists) is loaded from --split-file when
    given, otherwise freshly computed from --split-mode/--split-seed/
    --val-ratio; a freshly computed split is always persisted to
    --split-out (default <log-dir>/split_manifest.json). --limit-samples
    truncates the TRAIN indices only (smoke mode): the val side and the
    saved manifest always keep the full logical split.
    """
    dataset = ElectrostaticDataset(data_dir=args.data_path)
    if len(dataset) == 0:
        print("No data found! Please check the data path.")
        return None

    if args.split_file:
        train_entries, val_entries = load_split_manifest(args.split_file)
        train_indices, val_indices = indices_from_manifest(
            dataset, (train_entries, val_entries))
        print(f"Loaded split manifest from {args.split_file}: "
              f"{len(train_indices)} train / {len(val_indices)} val samples.")
    else:
        train_indices, val_indices = make_split(
            dataset, val_ratio=args.val_ratio, mode=args.split_mode,
            seed=args.split_seed)
        split_out = args.split_out or os.path.join(args.log_dir, 'split_manifest.json')
        save_split_manifest(split_out, dataset, train_indices, val_indices,
                            mode=args.split_mode, seed=args.split_seed,
                            val_ratio=args.val_ratio)
        print(f"Split (mode={args.split_mode}, seed={args.split_seed}, "
              f"val_ratio={args.val_ratio}): {len(train_indices)} train / "
              f"{len(val_indices)} val samples. Manifest saved to "
              f"{os.path.abspath(split_out)}")

    train_pool = list(train_indices)
    if args.limit_samples and args.limit_samples > 0:
        train_pool = train_pool[:args.limit_samples]
        print(f"Limiting train subset to first {len(train_pool)} samples (smoke mode).")

    train_loader = DataLoader(Subset(dataset, train_pool),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_indices),
                            batch_size=args.batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, len(dataset)


def calculate_masked_mse(outputs, targets, masks):
    masked_outputs = outputs * masks
    masked_targets = targets * masks
    squared_error = (masked_outputs - masked_targets) ** 2
    loss = squared_error.sum() / (masks.sum() + 1e-8)
    return loss


def calculate_physics_loss(prediction, inputs):
    """Laplace residual on the interior of the air region (inputs ch1).

    Finite differences are taken on the full prediction field; the eroded
    interior mask of the air region guarantees every stencil neighbor is an
    air pixel, so no boundary/padding jumps leak into the residual.
    Dimensionless with the fixed reference scale H (grid height in pixels).
    """
    air = inputs[:, 1:2, :, :]
    interior = erode_mask(air)
    H = prediction.size(-2)

    dy, dx = torch.gradient(prediction, dim=(-2, -1))
    dyy, _ = torch.gradient(dy, dim=(-2, -1))
    _, dxx = torch.gradient(dx, dim=(-2, -1))
    laplacian = dxx + dyy

    squared_residual = (laplacian * H ** 2) ** 2 * interior
    loss = squared_residual.sum() / (interior.sum() + 1e-8)
    return loss


def calculate_gradient_loss(outputs, targets, masks):
    """First-order gradient matching on the eroded interior of the valid region.

    Differences are taken on the full tensors first, then weighted by the
    eroded interior mask: the erosion ensures every stencil neighbor is a
    valid pixel, so the zero-padded region never produces spurious
    gradients at the valid-region boundary.
    """
    interior = erode_mask(masks)
    H = targets.size(-2)

    dy_pred, dx_pred = torch.gradient(outputs, dim=(-2, -1))
    dy_target, dx_target = torch.gradient(targets, dim=(-2, -1))

    loss_y = ((dy_pred - dy_target) ** 2 * interior).sum()
    loss_x = ((dx_pred - dx_target) ** 2 * interior).sum()
    loss = (H ** 2) * (loss_y + loss_x) / (interior.sum() + 1e-8)
    return loss


def calculate_bc_loss(outputs, targets, inputs):
    """Dirichlet soft constraint on source (ch0) union boundary (ch2)."""
    bc_mask = inputs[:, 0:1, :, :] + inputs[:, 2:3, :, :]
    squared_error = (outputs - targets) ** 2 * bc_mask
    loss = squared_error.sum() / (bc_mask.sum() + 1e-8)
    return loss


def calculate_eq_loss(outputs, inputs):
    """Equipotential (E = 0) constraint on the interior of the conductor,
    i.e. the union of source (ch0) and boundary (ch2) electrodes.

    E = 0 holds inside any conductor in electrostatic equilibrium. In the
    real data the source electrodes are thin strips (<= 2px) whose eroded
    interior is empty, so they are naturally silent here; the boundary is
    a thick conductor (~1350 interior pixels per sample), so its interior
    carries the constraint.
    """
    conductor = (inputs[:, 0:1, :, :] + inputs[:, 2:3, :, :]).clamp(max=1.0)
    interior = erode_mask(conductor)
    H = outputs.size(-2)

    dy, dx = torch.gradient(outputs, dim=(-2, -1))
    squared_field = (dy ** 2 + dx ** 2) * interior
    loss = (H ** 2) * squared_field.sum() / (interior.sum() + 1e-8)
    return loss


def train_one_epoch(model, loader, optimizer, device, scaler, amp_enabled,
                    lambda_phy, lambda_grad, lambda_bc, lambda_eq, grad_accum):
    model.train()
    running = {'total': 0.0, 'mse': 0.0, 'phy': 0.0, 'grad': 0.0, 'bc': 0.0, 'eq': 0.0}
    num_batches = len(loader)
    amp_ctx = torch.amp.autocast('cuda') if amp_enabled else contextlib.nullcontext()

    optimizer.zero_grad()
    for batch_idx, (inputs, targets, masks, pos_dist) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        masks = masks.to(device)
        pos_dist = pos_dist.to(device)

        with amp_ctx:
            outputs = model(inputs, pos_dist)

            loss_d = calculate_masked_mse(outputs, targets, masks)
            loss_p = calculate_physics_loss(outputs, inputs)
            loss_g = calculate_gradient_loss(outputs, targets, masks)
            loss_bc = calculate_bc_loss(outputs, targets, inputs)
            loss_eq = calculate_eq_loss(outputs, inputs)

            loss = (loss_d
                    + lambda_phy * loss_p
                    + lambda_grad * loss_g
                    + lambda_bc * loss_bc
                    + lambda_eq * loss_eq)

        if amp_enabled:
            # N scaled backward passes, then a single scaler.step + update
            scaler.scale(loss).backward()
        else:
            # Divide by N so the accumulated gradient keeps the same scale
            (loss / grad_accum).backward()

        if (batch_idx + 1) % grad_accum == 0:
            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        running['total'] += loss.item()
        running['mse'] += loss_d.item()
        running['phy'] += loss_p.item()
        running['grad'] += loss_g.item()
        running['bc'] += loss_bc.item()
        running['eq'] += loss_eq.item()

    # Flush leftover accumulated gradients when num_batches % grad_accum != 0
    if num_batches % grad_accum != 0:
        if amp_enabled:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    return {key: value / num_batches for key, value in running.items()}


def validate(model, loader, device, lambda_phy, lambda_grad, lambda_bc,
             lambda_eq, writer=None, epoch=None):
    """Validation pass: the same five-term weighted total loss used for
    training (forward only, no optimizer step) plus the full
    eval_utils metric set, both averaged over batches.

    When writer/epoch are given, writes the val/* scalars
    (loss/mse/rel_l2/max_err/bc_violation/laplace_residual/eq_residual)
    once per epoch. Returns the batch-averaged dict for the main loop
    ('total' and 'mse' keys are kept for backward compatibility).
    """
    model.eval()
    running = {'total': 0.0, 'mse': 0.0, 'rel_l2': 0.0, 'max_err': 0.0,
               'bc_violation': 0.0, 'laplace_residual': 0.0,
               'eq_residual': 0.0}
    num_batches = len(loader)

    with torch.no_grad():
        for inputs, targets, masks, pos_dist in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            pos_dist = pos_dist.to(device)

            outputs = model(inputs, pos_dist)

            loss_d = calculate_masked_mse(outputs, targets, masks)
            loss_p = calculate_physics_loss(outputs, inputs)
            loss_g = calculate_gradient_loss(outputs, targets, masks)
            loss_bc = calculate_bc_loss(outputs, targets, inputs)
            loss_eq = calculate_eq_loss(outputs, inputs)

            loss = (loss_d
                    + lambda_phy * loss_p
                    + lambda_grad * loss_g
                    + lambda_bc * loss_bc
                    + lambda_eq * loss_eq)

            metrics = compute_all_metrics(outputs, targets, inputs, masks)

            running['total'] += loss.item()
            running['mse'] += metrics['mse']
            running['rel_l2'] += metrics['relative_l2']
            running['max_err'] += metrics['max_abs_error']
            running['bc_violation'] += metrics['bc_violation']
            running['laplace_residual'] += metrics['laplace_residual']
            running['eq_residual'] += metrics['equipotential_residual']

    model.train()
    avg = {key: value / num_batches for key, value in running.items()}

    if writer is not None and epoch is not None:
        writer.add_scalar('val/loss', avg['total'], epoch)
        writer.add_scalar('val/mse', avg['mse'], epoch)
        writer.add_scalar('val/rel_l2', avg['rel_l2'], epoch)
        writer.add_scalar('val/max_err', avg['max_err'], epoch)
        writer.add_scalar('val/bc_violation', avg['bc_violation'], epoch)
        writer.add_scalar('val/laplace_residual', avg['laplace_residual'], epoch)
        writer.add_scalar('val/eq_residual', avg['eq_residual'], epoch)

    return avg


def log_rope_frequencies(writer, epoch, model):
    """Log the learnable RoPE frequencies of both cross-attention blocks."""
    blocks = {'s2g': model.cross_attn_s2g, 'g2s': model.cross_attn_g2s}
    for name, block in blocks.items():
        freqs = block.get_rope_frequencies().detach().cpu()  # (4 axes, pairs)
        for axis in range(freqs.shape[0]):
            for pair in range(freqs.shape[1]):
                writer.add_scalar(f'RoPE_Freq/{name}_axis{axis}_pair{pair}',
                                  freqs[axis, pair].item(), epoch)


def log_visualization(writer, epoch, model, loader, device):
    """Write Target/Prediction images of the first sample, padding masked out."""
    inputs, targets, masks, pos_dist = next(iter(loader))
    inputs = inputs.to(device)
    pos_dist = pos_dist.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(inputs, pos_dist)
    model.train()

    vis_mask = masks[0].cpu()  # (1, H, W)
    vis_target = targets[0].cpu() * vis_mask
    vis_output = outputs[0].detach().cpu() * vis_mask

    writer.add_image('Visual/Target_Potential', vis_target, epoch)
    writer.add_image('Visual/Predicted_Potential', vis_output, epoch)


# RGB palette for the region map: 0=padding (white), 1=source (blue),
# 2=air (grey), 3=HV electrode (red)
_REGION_PALETTE = np.array([
    [1.00, 1.00, 1.00],
    [0.10, 0.35, 0.70],
    [0.60, 0.60, 0.60],
    [0.85, 0.15, 0.15],
], dtype=np.float32)


def log_val_visualization(writer, epoch, model, val_loader, device, vis_samples):
    """Per-interval validation visualization: for the first K val samples,
    a 1x4 figure (region map / target potential / predicted potential /
    |error|), plus one pooled |pred-target| error histogram over the K
    samples' masked pixels.

    The val loader wraps a Subset of the shared dataset, so condition
    names are resolved through Subset.indices (subset position ->
    original dataset index -> dataset.conditions).
    """
    val_subset = val_loader.dataset  # Subset(dataset, val_indices)
    base_dataset = val_subset.dataset
    k = min(vis_samples, len(val_subset))
    if k <= 0:
        return

    batch = [val_subset[i] for i in range(k)]
    inputs = torch.stack([b[0] for b in batch]).to(device)
    targets = torch.stack([b[1] for b in batch])
    masks = torch.stack([b[2] for b in batch])
    pos_dist = torch.stack([b[3] for b in batch]).to(device)
    conditions = [base_dataset.get_condition(val_subset.indices[i])
                  for i in range(k)]

    model.eval()
    with torch.no_grad():
        outputs = model(inputs, pos_dist)
    model.train()

    onehot_np = inputs.detach().cpu().numpy()   # (K,3,H,W)
    target_np = targets.numpy()                 # (K,1,H,W)
    mask_np = masks.numpy()                     # (K,1,H,W)
    pred_np = outputs.detach().cpu().numpy()    # (K,1,H,W)

    error_chunks = []
    for i in range(k):
        onehot = onehot_np[i]
        tgt = target_np[i, 0] * mask_np[i, 0]
        pred = pred_np[i, 0] * mask_np[i, 0]
        err = np.abs(pred_np[i, 0] - target_np[i, 0]) * mask_np[i, 0]
        error_chunks.append(err[mask_np[i, 0] > 0.5])

        # Region code: argmax over one-hot channels +1 -> 1=src/2=air/3=HV;
        # padding pixels have an all-zero one-hot and are forced to 0.
        region_code = np.argmax(onehot, axis=0).astype(np.int8) + 1
        region_code[onehot.sum(axis=0) < 0.5] = 0
        region_rgb = _REGION_PALETTE[region_code]

        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
        axes[0].imshow(region_rgb, interpolation='nearest')
        axes[0].set_title('Region (blue=src, grey=air, red=HV, white=pad)')
        axes[1].imshow(tgt, cmap='viridis', vmin=0.0, vmax=1.0)
        axes[1].set_title('Target potential')
        axes[2].imshow(pred, cmap='viridis', vmin=0.0, vmax=1.0)
        axes[2].set_title('Predicted potential')
        err_vmax = max(float(err.max()), 1e-8)
        axes[3].imshow(err, cmap='magma', vmin=0.0, vmax=err_vmax)
        axes[3].set_title('|Error| (masked)')
        for ax in axes:
            ax.set_axis_off()
        fig.suptitle(f'Epoch {epoch + 1} | {conditions[i]}', fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        writer.add_figure(f'val/sample_{i}', fig, epoch)
        plt.close(fig)

    # Pooled |pred-target| histogram over the K samples' masked pixels
    values = np.concatenate([c for c in error_chunks if c.size])
    if values.size == 0:
        return
    vmax = float(values.max())
    if vmax <= 0:
        vmax = 1e-8
    counts, limits = np.histogram(values, bins=30, range=(0.0, vmax))
    writer.add_histogram_raw(
        'val/error_hist', min=float(values.min()), max=float(values.max()),
        num=int(values.size), sum=float(values.sum()),
        sum_squares=float((values ** 2).sum()),
        bucket_limits=limits[1:].tolist(), bucket_counts=counts.tolist(),
        global_step=epoch)


def save_model(model, save_dir, filename="dual_encoder_fno_model_v2.pth"):
    save_path = os.path.join(save_dir, filename)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def train():
    args = parse_args()
    device = get_device()

    loaders = build_dataloaders(args)
    if loaders is None:
        return
    train_loader, val_loader, num_samples = loaders

    print(f"Found {num_samples} samples.")

    model = DualEncoderFNODecoder(modes=args.modes, width=args.width).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # AMP only on CUDA; plain forward/backward on CPU
    amp_enabled = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda') if amp_enabled else None

    # Run directory: <log-dir>/<run-name>[_<tag>], auto-named when empty
    run_name = args.run_name or f'v2_{time.strftime("%Y%m%d_%H%M%S")}'
    if args.tag:
        run_name = f'{run_name}_{args.tag}'
    run_dir = os.path.join(args.log_dir, run_name)
    writer = SummaryWriter(log_dir=run_dir)

    # Hparams: all JSON-serializable argparse args + split sizes. Note
    # add_hparams writes its own sub-run directory (known behavior); the
    # placeholder metric keeps metric_dict non-empty.
    hparam_dict = {k: v for k, v in vars(args).items()
                   if isinstance(v, (bool, int, float, str))}
    hparam_dict['num_train_samples'] = len(train_loader.dataset)
    hparam_dict['num_val_samples'] = len(val_loader.dataset)
    writer.add_hparams(hparam_dict, {'hparam/placeholder': 0})

    print("Starting training with Dual Encoder FNO Model...")
    print(f"AMP enabled: {amp_enabled}, grad accumulation: {args.grad_accum}")
    print(f"TensorBoard run directory: {os.path.abspath(run_dir)}")

    for epoch in range(args.epochs):
        avg = train_one_epoch(model, train_loader, optimizer, device, scaler, amp_enabled,
                              args.lambda_phy, args.lambda_grad, args.lambda_bc, args.lambda_eq,
                              args.grad_accum)
        scheduler.step(avg['total'])
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg['total']:.6f}, MSE: {avg['mse']:.6f}, "
              f"Phy: {avg['phy']:.6f}, Grad: {avg['grad']:.6f}, BC: {avg['bc']:.6f}, "
              f"Eq: {avg['eq']:.6f}, LR: {current_lr:.2e}")

        writer.add_scalar('Loss/Total', avg['total'], epoch)
        writer.add_scalar('Loss/MSE', avg['mse'], epoch)
        writer.add_scalar('Loss/Physics', avg['phy'], epoch)
        writer.add_scalar('Loss/Gradient', avg['grad'], epoch)
        writer.add_scalar('Loss/BC', avg['bc'], epoch)
        writer.add_scalar('Loss/Equipotential', avg['eq'], epoch)

        if len(val_loader) > 0:
            val_avg = validate(model, val_loader, device, args.lambda_phy,
                               args.lambda_grad, args.lambda_bc, args.lambda_eq,
                               writer=writer, epoch=epoch)
            print(f"[Epoch {epoch + 1}] val_loss={val_avg['total']:.6f}, "
                  f"val_mse={val_avg['mse']:.6f}, "
                  f"val_rel_l2={val_avg['rel_l2']:.6f}, "
                  f"val_max_err={val_avg['max_err']:.6f}, "
                  f"val_bc_violation={val_avg['bc_violation']:.6f}, "
                  f"val_laplace_residual={val_avg['laplace_residual']:.6f}, "
                  f"val_eq_residual={val_avg['eq_residual']:.6f}")

        log_rope_frequencies(writer, epoch, model)

        if len(val_loader) > 0 and (epoch + 1) % args.vis_interval == 0:
            log_val_visualization(writer, epoch, model, val_loader, device,
                                  args.vis_samples)

        if (epoch + 1) % 10 == 0:
            log_visualization(writer, epoch, model, train_loader, device)

    writer.close()
    save_model(model, save_dir=args.save_dir)


if __name__ == '__main__':
    train()
