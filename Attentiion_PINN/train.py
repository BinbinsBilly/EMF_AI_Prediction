import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import sys
from torch.utils.tensorboard import SummaryWriter

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import ElectrostaticDataset
from model import DualEncoderFNODecoder

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    return device

def get_dataloader(data_path, batch_size):
    dataset = ElectrostaticDataset(data_dir=data_path)
    if len(dataset) == 0:
        print("No data found! Please check the data path.")
        return None, 0
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader, len(dataset)

def calculate_masked_mse(outputs, targets, masks):
    masked_outputs = outputs * masks
    masked_targets = targets * masks
    squared_error = (masked_outputs - masked_targets) ** 2
    loss = squared_error.sum() / (masks.sum() + 1e-8)
    return loss

def calculate_gradient_loss(outputs, targets, masks):
    masked_outputs = outputs * masks
    masked_targets = targets * masks

    dy_pred, dx_pred = torch.gradient(masked_outputs, dim=(-2, -1))
    dy_target, dx_target = torch.gradient(masked_targets, dim=(-2, -1))
    
    masked_dy_pred = dy_pred * masks
    masked_dx_pred = dx_pred * masks
    masked_dy_target = dy_target * masks
    masked_dx_target = dx_target * masks
    
    loss_y = F.mse_loss(masked_dy_pred, masked_dy_target, reduction='sum')
    loss_x = F.mse_loss(masked_dx_pred, masked_dx_target, reduction='sum')
    
    return (loss_x + loss_y) / (masks.sum() + 1e-8)

def calculate_physics_loss(prediction, mask):
    # mask is one-hot: (B, 3, H, W)
    # Channel 1: Air
    dy, dx = torch.gradient(prediction, dim=(-2, -1))
    dyy, _ = torch.gradient(dy, dim=(-2, -1))
    _, dxx = torch.gradient(dx, dim=(-2, -1))
    laplacian = dxx + dyy
    
    region_2_mask = mask[:, 1:2, :, :] 
    masked_laplacian = laplacian * region_2_mask
    
    num_valid_pixels = torch.sum(region_2_mask) + 1e-8
    loss = torch.sum(masked_laplacian ** 2) / num_valid_pixels
    return loss

def train_one_epoch(model, loader, optimizer, device, scaler, lambda_phy=0.5, lambda_grad=0.1):
    model.train()
    epoch_loss = 0.0
    epoch_loss_d = 0.0
    epoch_loss_p = 0.0
    epoch_loss_g = 0.0
    
    for batch_idx, (inputs, targets, masks) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Updated API for autocast
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            
            loss_d = calculate_masked_mse(outputs, targets, masks)
            loss_p = calculate_physics_loss(outputs, inputs)
            loss_g = calculate_gradient_loss(outputs, targets, masks)
            
            loss = loss_d + lambda_phy * loss_p + lambda_grad * loss_g
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        epoch_loss_d += loss_d.item()
        epoch_loss_p += loss_p.item()
        epoch_loss_g += loss_g.item()
        
    num_batches = len(loader)
    return epoch_loss / num_batches, epoch_loss_d / num_batches, epoch_loss_p / num_batches, epoch_loss_g / num_batches

def log_to_tensorboard(writer, epoch, avg_loss, avg_loss_d, avg_loss_p, avg_loss_g, model, loader, device):
    writer.add_scalar('Loss/Total', avg_loss, epoch)
    writer.add_scalar('Loss/MSE_Loss', avg_loss_d, epoch)
    writer.add_scalar('Loss/Physics_Loss', avg_loss_p, epoch)
    writer.add_scalar('Loss/Gradient_Loss', avg_loss_g, epoch)
    
    if (epoch + 1) % 10 == 0:
        inputs, targets, masks = next(iter(loader))
        inputs = inputs.to(device)
        targets = targets.to(device)
        masks = masks.to(device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
        model.train()
        
        # Visualization
        input_onehot = inputs[0].cpu()
        vis_input = (input_onehot[0] * 1 + input_onehot[1] * 2 + input_onehot[2] * 3).unsqueeze(0) / 3.0
        
        vis_target = targets[0].cpu()
        vis_output = outputs[0].cpu()
        
        # Mask output for visualization
        vis_mask = masks[0].cpu()
        vis_output[vis_mask == 0] = 0 # Or some background value
        
        writer.add_image('Visual/Input_Mask', vis_input, epoch)
        writer.add_image('Visual/Target_Potential', vis_target, epoch)
        writer.add_image('Visual/Predicted_Potential', vis_output, epoch)

def save_model(model, save_dir, filename="model.pth"):
    save_path = os.path.join(save_dir, filename)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def train():
    device = get_device()

    # Hyperparameters
    learning_rate = 1e-3
    batch_size = 2 # Reduced for 4GB GPU
    epochs = 50
    lambda_phy = 0.5
    lambda_grad = 0.1
    
    modes = 16 # Reduced modes for memory if needed, or keep 32
    width = 32

    data_path = r"C:\Users\15107\Desktop\德显大创2D电磁场预测\retangle_data\results_reduction_fixed_boundary"
    
    train_loader, num_samples = get_dataloader(data_path, batch_size)
    if train_loader is None:
        return
    
    print(f"Found {num_samples} samples.")

    model = DualEncoderFNODecoder(modes=modes, width=width).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # verbose parameter is deprecated
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    # Updated API for GradScaler
    scaler = torch.amp.GradScaler('cuda')

    writer = SummaryWriter(log_dir='runs/DualEncoder_FNO_experiment')

    print("Starting training with Dual Encoder FNO Model...")
    
    for epoch in range(epochs):
        avg_loss, avg_loss_d, avg_loss_p, avg_loss_g = train_one_epoch(model, train_loader, optimizer, device, scaler, lambda_phy, lambda_grad)
        scheduler.step(avg_loss)
        
        # Manually print learning rate if needed
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, MSE: {avg_loss_d:.6f}, Phy: {avg_loss_p:.6f}, Grad: {avg_loss_g:.6f}, LR: {current_lr:.2e}")
        
        log_to_tensorboard(writer, epoch, avg_loss, avg_loss_d, avg_loss_p, avg_loss_g, model, train_loader, device)

    writer.close()
    save_model(model, save_dir="Attentiion_PINN", filename="dual_encoder_fno_model.pth")

if __name__ == '__main__':
    train()
