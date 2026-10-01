import argparse
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')  # headless environment: no display available
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import DualEncoderFNODecoder
from dataset import ElectrostaticDataset
# train.py's argparse only executes inside train() under the __main__ guard,
# so importing it has no side effects. Reuse erode_mask so that the error
# maps below stay aligned with the training loss conventions.
from train import erode_mask


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize DualEncoder-FNO electrostatic PINN predictions')
    parser.add_argument('--data-path', type=str,
                        default='/workspace/retangle_data/results_reduction_fixed_boundary',
                        help='Directory containing input/target CSV pairs')
    parser.add_argument('--model-path', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'dual_encoder_fno_model_v2.pth'),
                        help='Path to trained model weights')
    parser.add_argument('--modes', type=int, default=32,
                        help='Spectral modes of the FNO decoder (must match checkpoint)')
    parser.add_argument('--width', type=int, default=32,
                        help='Feature width of the model (must match checkpoint)')
    parser.add_argument('--limit', type=int, default=20,
                        help='Number of samples to visualize')
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'visualizations_output'),
                        help='Directory to save the visualization PNGs')
    return parser.parse_args()


def calculate_mse_error_map(outputs, targets, masks):
    # Same per-pixel form as train.calculate_masked_mse (raw valid mask)
    masked_outputs = outputs * masks
    masked_targets = targets * masks
    squared_error = (masked_outputs - masked_targets) ** 2
    return squared_error


def calculate_physics_error_map(prediction, inputs):
    # Aligned with train.calculate_physics_loss: Laplace residual from
    # full-field second differences, weighted by the eroded interior of
    # the air region (inputs ch1), dimensionless with reference scale H.
    air = inputs[:, 1:2, :, :]
    interior = erode_mask(air)
    H = prediction.size(-2)

    dy, dx = torch.gradient(prediction, dim=(-2, -1))
    dyy, _ = torch.gradient(dy, dim=(-2, -1))
    _, dxx = torch.gradient(dx, dim=(-2, -1))
    laplacian = dxx + dyy

    return (laplacian * H ** 2) ** 2 * interior


def calculate_gradient_error_map(outputs, targets, masks):
    # Aligned with train.calculate_gradient_loss: full-field first
    # differences, weighted by the eroded interior of the valid mask,
    # dimensionless with reference scale H.
    interior = erode_mask(masks)
    H = targets.size(-2)

    dy_pred, dx_pred = torch.gradient(outputs, dim=(-2, -1))
    dy_target, dx_target = torch.gradient(targets, dim=(-2, -1))

    return (H ** 2) * (((dy_pred - dy_target) ** 2) + ((dx_pred - dx_target) ** 2)) * interior


def get_robust_vmax(data, p=99):
    if np.all(np.isnan(data)): return 1.0
    return np.nanpercentile(data, p)


def visualize():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to {output_dir}")

    dataset = ElectrostaticDataset(data_dir=args.data_path)
    if len(dataset) == 0:
        print("Dataset is empty.")
        return

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    model = DualEncoderFNODecoder(modes=args.modes, width=args.width).to(device)

    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model loaded from {args.model_path}")
    else:
        print(f"Model not found at {args.model_path}")
        return

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.limit: break

            inputs, targets, masks, pos_dist = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            pos_dist = pos_dist.to(device)

            outputs = model(inputs, pos_dist)

            mse_map = calculate_mse_error_map(outputs, targets, masks)
            grad_map = calculate_gradient_error_map(outputs, targets, masks)
            phy_map = calculate_physics_error_map(outputs, inputs)

            input_onehot = inputs[0].cpu()
            vis_input = (input_onehot[0] * 1 + input_onehot[1] * 2 + input_onehot[2] * 3).numpy()

            vis_target = targets[0, 0].cpu().numpy()
            vis_output = outputs[0, 0].cpu().numpy()
            vis_mask = masks[0, 0].cpu().numpy()

            vis_output_masked = vis_output.copy()
            vis_output_masked[vis_mask == 0] = np.nan

            vis_target_masked = vis_target.copy()
            vis_target_masked[vis_mask == 0] = np.nan

            vis_mse = mse_map[0, 0].cpu().numpy()
            vis_grad = grad_map[0, 0].cpu().numpy()
            vis_phy = phy_map[0, 0].cpu().numpy()

            vis_mse[vis_mask == 0] = np.nan
            vis_phy[vis_mask == 0] = np.nan
            vis_grad[vis_mask == 0] = np.nan

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))

            im0 = axes[0, 0].imshow(vis_input, cmap='tab10', interpolation='nearest')
            axes[0, 0].set_title('Input Mask')
            plt.colorbar(im0, ax=axes[0, 0])

            global_min = min(np.nanmin(vis_target_masked), np.nanmin(vis_output_masked))
            global_max = max(np.nanmax(vis_target_masked), np.nanmax(vis_output_masked))

            im1 = axes[0, 1].imshow(vis_target_masked, cmap='jet', vmin=global_min, vmax=global_max)
            axes[0, 1].set_title(f'Ground Truth\nRange: [{np.nanmin(vis_target_masked):.2f}, {np.nanmax(vis_target_masked):.2f}]')
            plt.colorbar(im1, ax=axes[0, 1])

            im2 = axes[0, 2].imshow(vis_output_masked, cmap='jet', vmin=global_min, vmax=global_max)
            axes[0, 2].set_title(f'Prediction (Masked)\nRange: [{np.nanmin(vis_output_masked):.2f}, {np.nanmax(vis_output_masked):.2f}]')
            plt.colorbar(im2, ax=axes[0, 2])

            vmax_mse = get_robust_vmax(vis_mse)
            im3 = axes[1, 0].imshow(vis_mse, cmap='inferno', vmin=0, vmax=vmax_mse)
            axes[1, 0].set_title(f'MSE Error (Squared)\n99% Max: {vmax_mse:.2e}')
            plt.colorbar(im3, ax=axes[1, 0])

            vmax_phy = get_robust_vmax(vis_phy)
            im4 = axes[1, 1].imshow(vis_phy, cmap='inferno', vmin=0, vmax=vmax_phy)
            axes[1, 1].set_title(f'Physics Error\n99% Max: {vmax_phy:.2e}')
            plt.colorbar(im4, ax=axes[1, 1])

            vmax_grad = get_robust_vmax(vis_grad)
            im5 = axes[1, 2].imshow(vis_grad, cmap='inferno', vmin=0, vmax=vmax_grad)
            axes[1, 2].set_title(f'Gradient Error\n99% Max: {vmax_grad:.2e}')
            plt.colorbar(im5, ax=axes[1, 2])

            plt.suptitle(f'Sample {i}', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'result_{i}.png'))
            plt.close()
            print(f"Saved result_{i}.png")

if __name__ == "__main__":
    visualize()
