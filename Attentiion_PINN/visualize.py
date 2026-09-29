import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import DualEncoderFNODecoder
from dataset import ElectrostaticDataset

def calculate_mse_error_map(outputs, targets, masks):
    masked_outputs = outputs * masks
    masked_targets = targets * masks
    squared_error = (masked_outputs - masked_targets) ** 2
    return squared_error

def calculate_gradient_error_map(outputs, targets, masks):
    masked_outputs = outputs * masks
    masked_targets = targets * masks

    dy_pred, dx_pred = torch.gradient(masked_outputs, dim=(-2, -1))
    dy_target, dx_target = torch.gradient(masked_targets, dim=(-2, -1))
    
    masked_dy_pred = dy_pred * masks
    masked_dx_pred = dx_pred * masks
    masked_dy_target = dy_target * masks
    masked_dx_target = dx_target * masks
    
    error_y = (masked_dy_pred - masked_dy_target) ** 2
    error_x = (masked_dx_pred - masked_dx_target) ** 2
    
    return error_x + error_y

def calculate_physics_error_map(prediction, mask):
    dy, dx = torch.gradient(prediction, dim=(-2, -1))
    dyy, _ = torch.gradient(dy, dim=(-2, -1))
    _, dxx = torch.gradient(dx, dim=(-2, -1))
    laplacian = dxx + dyy
    
    region_2_mask = mask[:, 1:2, :, :] 
    masked_laplacian = laplacian * region_2_mask
    return masked_laplacian ** 2

def visualize():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "dual_encoder_fno_model.pth")
    data_path = r"C:\Users\15107\Desktop\德显大创2D电磁场预测\retangle_data\results_reduction_fixed_boundary"
    output_dir = os.path.join(current_dir, "visualizations_output")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to {output_dir}")
    
    dataset = ElectrostaticDataset(data_dir=data_path)
    if len(dataset) == 0:
        print("Dataset is empty.")
        return
        
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    modes = 16
    width = 32
    model = DualEncoderFNODecoder(modes=modes, width=width).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    else:
        print(f"Model not found at {model_path}")
        return

    model.eval()
    
    with torch.no_grad():
        for i, (inputs, targets, masks) in enumerate(loader):
            if i >= 20: break 
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            
            outputs = model(inputs)
            
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
            
            def get_robust_vmax(data, p=99):
                if np.all(np.isnan(data)): return 1.0
                return np.nanpercentile(data, p)

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
