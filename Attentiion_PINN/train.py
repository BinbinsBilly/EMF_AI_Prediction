import argparse
import contextlib
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import ElectrostaticDataset
from model import DualEncoderFNODecoder


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
                        help='Use only the first N sample pairs (0 = all)')
    parser.add_argument('--save-dir', type=str,
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help='Directory to save model weights')
    parser.add_argument('--log-dir', type=str, default='runs/DualEncoder_FNO_v2',
                        help='TensorBoard log directory')
    return parser.parse_args()


def get_dataloader(data_path, batch_size, limit_samples=0):
    dataset = ElectrostaticDataset(data_dir=data_path)
    if len(dataset) == 0:
        print("No data found! Please check the data path.")
        return None, 0

    num_samples = len(dataset)
    if limit_samples and limit_samples > 0:
        num_samples = min(limit_samples, num_samples)
        dataset = Subset(dataset, list(range(num_samples)))
        print(f"Limiting dataset to first {num_samples} samples (smoke mode).")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader, num_samples


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
    """
    m = mask * region_mask if region_mask is not None else mask
    vertical = -F.max_pool2d(-m, kernel_size=(3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-m, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


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


def save_model(model, save_dir, filename="dual_encoder_fno_model_v2.pth"):
    save_path = os.path.join(save_dir, filename)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def train():
    args = parse_args()
    device = get_device()

    train_loader, num_samples = get_dataloader(args.data_path, args.batch_size, args.limit_samples)
    if train_loader is None:
        return

    print(f"Found {num_samples} samples.")

    model = DualEncoderFNODecoder(modes=args.modes, width=args.width).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # AMP only on CUDA; plain forward/backward on CPU
    amp_enabled = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda') if amp_enabled else None

    writer = SummaryWriter(log_dir=args.log_dir)

    print("Starting training with Dual Encoder FNO Model...")
    print(f"AMP enabled: {amp_enabled}, grad accumulation: {args.grad_accum}")

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

        log_rope_frequencies(writer, epoch, model)

        if (epoch + 1) % 10 == 0:
            log_visualization(writer, epoch, model, train_loader, device)

    writer.close()
    save_model(model, save_dir=args.save_dir)


if __name__ == '__main__':
    train()
