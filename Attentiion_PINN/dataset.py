import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import distance_transform_edt

class ElectrostaticDataset(Dataset):
    def __init__(self, data_dir, target_height=256, norm_factor=60000.0):
        """
        Args:
            data_dir (str): Path to the directory containing the CSV files.
            target_height (int): The target height to pad the data to.
            norm_factor (float): Factor to normalize the target potential.
                               Default 60000.0 (measured global max potential).
        """
        self.data_dir = data_dir
        self.target_height = target_height
        self.norm_factor = norm_factor
        self.file_pairs = self._find_file_pairs(data_dir)

    def _find_file_pairs(self, data_dir):
        """Finds and pairs input and target CSV files."""
        file_pairs = []
        # Find all original_region_data_*.csv files
        search_pattern = os.path.join(data_dir, "original_region_data_*.csv")
        input_files = glob.glob(search_pattern)

        for input_path in input_files:
            # Construct the corresponding potential_distribution filename
            filename = os.path.basename(input_path)
            suffix = filename.replace("original_region_data_", "")
            target_filename = "potential_distribution_" + suffix
            target_path = os.path.join(data_dir, target_filename)

            if os.path.exists(target_path):
                file_pairs.append((input_path, target_path))
            else:
                print(f"Warning: Target file not found for {input_path}")
        return file_pairs

    def __len__(self):
        return len(self.file_pairs)

    def _load_data(self, input_path, target_path):
        """Reads CSV files and returns numpy arrays."""
        # Read CSVs using pandas (assuming no header)
        input_df = pd.read_csv(input_path, header=None, dtype=np.float32)
        target_df = pd.read_csv(target_path, header=None, dtype=np.float32)
        return input_df.values, target_df.values

    def _normalize_target(self, target_np):
        """Normalizes the target potential distribution."""
        return target_np / self.norm_factor

    def _pad_data(self, input_np, target_np):
        """
        Pads data to target_height and returns a validity mask.
        Returns:
            padded_input, padded_target, mask
        """
        h, w = input_np.shape
        # Create a mask indicating valid regions (1 for valid, 0 for padded)
        mask_np = np.ones_like(input_np, dtype=np.float32)
        
        if h < self.target_height:
            pad_total = self.target_height - h
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            
            pad_width = ((pad_top, pad_bottom), (0, 0))
            
            # Pad with zeros. ((top, bottom), (left, right))
            input_np = np.pad(input_np, pad_width, mode='constant', constant_values=0)
            target_np = np.pad(target_np, pad_width, mode='constant', constant_values=0)
            mask_np = np.pad(mask_np, pad_width, mode='constant', constant_values=0)
            
        return input_np, target_np, mask_np

    def _compute_pos_dist(self, input_np):
        """
        Computes distance-based positional auxiliary maps on the UNPADDED
        original region map (values: 1=source, 2=air, 3=boundary).
        Units are pixels (aligned with the RoPE pixel coordinate axes,
        no extra normalization/scaling).
        Returns:
            pos_dist: (2, H, W) float32
                Channel 0: Euclidean distance to the nearest source pixel.
                Channel 1: Euclidean distance to the nearest boundary pixel.
        """
        dist_to_source = distance_transform_edt(input_np != 1).astype(np.float32)
        dist_to_boundary = distance_transform_edt(input_np != 3).astype(np.float32)
        return np.stack([dist_to_source, dist_to_boundary], axis=0)

    def _pad_pos_dist(self, pos_dist_np):
        """
        Pads the (2, H, W) pos_dist to target_height along H with the same
        symmetric top/bottom zero-padding scheme as _pad_data.
        Padded regions in the distance maps are 0.
        """
        _, h, _ = pos_dist_np.shape
        if h < self.target_height:
            pad_total = self.target_height - h
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            pad_width = ((0, 0), (pad_top, pad_bottom), (0, 0))
            pos_dist_np = np.pad(pos_dist_np, pad_width, mode='constant', constant_values=0)
        return pos_dist_np

    def _to_tensors(self, input_np, target_np, mask_np):
        """Converts numpy arrays to PyTorch tensors with channel dimension."""
        # input_np: (H, W) with values 0, 1, 2, 3
        # One-Hot Encoding for Input: (3, H, W)
        # Channel 0: Source (Val 1)
        # Channel 1: Air (Val 2)
        # Channel 2: Boundary (Val 3)
        # Padding (Val 0) will be [0, 0, 0]
        
        h, w = input_np.shape
        input_onehot = np.zeros((3, h, w), dtype=np.float32)
        
        input_onehot[0, input_np == 1] = 1.0
        input_onehot[1, input_np == 2] = 1.0
        input_onehot[2, input_np == 3] = 1.0
        
        input_tensor = torch.from_numpy(input_onehot) # (3, H, W)
        target_tensor = torch.from_numpy(target_np).unsqueeze(0) # (1, H, W)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0) # (1, H, W)
        
        return input_tensor, target_tensor, mask_tensor

    def __getitem__(self, idx):
        input_path, target_path = self.file_pairs[idx]

        # 1. Load Data
        input_np, target_np = self._load_data(input_path, target_path)

        # 2. Normalize
        target_np = self._normalize_target(target_np)

        # 3. Compute distance positional maps on the UNPADDED region map
        pos_dist_np = self._compute_pos_dist(input_np)

        # 4. Pad (same symmetric zero-padding scheme; distance maps are 0 in padded regions)
        input_np, target_np, mask_np = self._pad_data(input_np, target_np)
        pos_dist_np = self._pad_pos_dist(pos_dist_np)

        # 5. Convert to Tensor
        input_tensor, target_tensor, mask_tensor = self._to_tensors(input_np, target_np, mask_np)
        pos_dist_tensor = torch.from_numpy(np.ascontiguousarray(pos_dist_np)) # (2, H, W)

        return input_tensor, target_tensor, mask_tensor, pos_dist_tensor
