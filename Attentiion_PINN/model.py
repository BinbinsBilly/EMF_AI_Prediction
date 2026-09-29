import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        # Use float32 storage for weights to be compatible with AMP GradScaler
        # Shape: (..., 2) representing real and imag parts
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype=torch.float32))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype=torch.float32))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        # Disable AMP for FFT/Complex operations to avoid ComplexHalf errors
        # Updated API for autocast
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            batchsize = x.shape[0]
            x_ft = torch.fft.rfft2(x)
            
            # Convert stored float weights to complex on the fly
            w1 = torch.view_as_complex(self.weights1)
            w2 = torch.view_as_complex(self.weights2)

            out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
            out_ft[:, :, :self.modes1, :self.modes2] = \
                self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], w1)
            out_ft[:, :, -self.modes1:, :self.modes2] = \
                self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], w2)

            x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
            return x

class LightweightUNetEncoder(nn.Module):
    """
    Lightweight U-Net Encoder for Source Channel.
    Outputs feature map of same resolution as input (or downsampled if needed).
    Here we keep it same resolution to match FNO requirements easily, or we can downsample.
    Let's output features at 1/4 resolution to save compute, then FNO can work on that.
    But FNO usually works better on global features.
    Let's keep it simple: A few convs with residual connections, preserving resolution.
    """
    def __init__(self, in_channels=1, base_channels=16):
        super(LightweightUNetEncoder, self).__init__()
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU()
        )
        
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.GELU()
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.GELU()
        )
        
        # Upsample back to original resolution to fuse with geometry
        self.up1 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        
        self.out_conv = nn.Conv2d(base_channels, base_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        x_up = self.up1(x3)
        x_up = x_up + x2 # Skip connection
        x_up = self.up2(x_up)
        x_up = x_up + x1 # Skip connection
        
        return self.out_conv(x_up)

class GeometryEmbedding(nn.Module):
    """
    Embedding for Air and Boundary channels.
    Input: (B, 2, H, W)
    """
    def __init__(self, in_channels=2, out_channels=16):
        super(GeometryEmbedding, self).__init__()
        # 1x1 Conv acts as per-pixel embedding
        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.GELU()
        )
        # Spatial mixing
        self.spatial_mix = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.spatial_mix(x)
        return x

class CrossAttentionBlock(nn.Module):
    """
    Cross Attention between Source Features and Geometry Features.
    Includes downsampling for Key/Value to save memory (Linear Complexity-ish).
    """
    def __init__(self, dim, num_heads=4, downsample_factor=8):
        super(CrossAttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.downsample_factor = downsample_factor

        self.q_proj = nn.Conv2d(dim, dim, 1)
        self.k_proj = nn.Conv2d(dim, dim, 1)
        self.v_proj = nn.Conv2d(dim, dim, 1)
        self.out_proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x, context):
        """
        x: Query (B, C, H, W)
        context: Key/Value (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Downsample context (Key/Value) to reduce memory complexity from O((HW)^2) to O(HW * (HW/k^2))
        if self.downsample_factor > 1:
            # Use adaptive pooling to ensure fixed small size or just stride
            # Adaptive pooling is safer for varying input sizes
            h_k, w_k = H // self.downsample_factor, W // self.downsample_factor
            context_pooled = F.adaptive_avg_pool2d(context, (h_k, w_k))
        else:
            context_pooled = context
            h_k, w_k = H, W

        # Q: (B, heads, head_dim, H*W)
        q = self.q_proj(x).view(B, self.num_heads, self.head_dim, H*W)
        
        # K, V: (B, heads, head_dim, h_k*w_k)
        k = self.k_proj(context_pooled).view(B, self.num_heads, self.head_dim, h_k*w_k)
        v = self.v_proj(context_pooled).view(B, self.num_heads, self.head_dim, h_k*w_k)

        # Attention Map: (B, heads, H*W, h_k*w_k)
        # For 256x256 input and factor 8 (32x32 context):
        # 65536 * 1024 elements per head ~ 67M elements (268MB float32)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        # Output: (B, heads, head_dim, H*W)
        # v: (..., D, S), attn.T: (..., S, N) -> (..., D, N)
        out = (v @ attn.transpose(-2, -1)).reshape(B, C, H, W)
        
        return self.out_proj(out)

class DualEncoderFNODecoder(nn.Module):
    def __init__(self, modes=32, width=32):
        super(DualEncoderFNODecoder, self).__init__()
        
        self.modes = modes
        self.width = width
        
        # Encoders
        self.source_encoder = LightweightUNetEncoder(in_channels=1, base_channels=width)
        self.geo_encoder = GeometryEmbedding(in_channels=2, out_channels=width)
        
        # Cross Attention
        # Source attending to Geometry
        # Downsample factor 16 (16x16 context) reduces memory usage significantly for 4GB GPU
        # Attention Map: (B, heads, 65536, 256) -> ~16M elements * 4 bytes = 64MB per batch
        self.cross_attn_s2g = CrossAttentionBlock(width, downsample_factor=16)
        # Geometry attending to Source
        self.cross_attn_g2s = CrossAttentionBlock(width, downsample_factor=16)
        
        # Fusion
        self.fusion_conv = nn.Conv2d(width * 2, width, 1)
        
        # FNO Decoder
        self.fno1 = SpectralConv2d(width, width, modes, modes)
        self.fno2 = SpectralConv2d(width, width, modes, modes)
        self.fno3 = SpectralConv2d(width, width, modes, modes)
        self.fno4 = SpectralConv2d(width, width, modes, modes)
        
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        self.w4 = nn.Conv2d(width, width, 1)
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(width, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        # x: (B, 3, H, W)
        # Split channels
        source = x[:, 0:1, :, :] # (B, 1, H, W)
        geo = x[:, 1:3, :, :]    # (B, 2, H, W)
        
        # Encode
        feat_s = self.source_encoder(source) # (B, width, H, W)
        feat_g = self.geo_encoder(geo)       # (B, width, H, W)
        
        # Cross Attention Interaction
        # We downsample for attention to save memory if needed, but here we keep full res for precision
        # If OOM, consider downsampling before attention
        
        # Source queries Geometry
        attn_s = self.cross_attn_s2g(feat_s, feat_g)
        # Geometry queries Source
        attn_g = self.cross_attn_g2s(feat_g, feat_s)
        
        # Fuse: Original features + Attended features
        feat_s_fused = feat_s + attn_s
        feat_g_fused = feat_g + attn_g
        
        # Concatenate
        fused = torch.cat([feat_s_fused, feat_g_fused], dim=1)
        x = self.fusion_conv(fused)
        
        # FNO Decoder
        x1 = self.fno1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.fno2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.fno3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.fno4(x)
        x2 = self.w4(x)
        x = x1 + x2
        
        # Output
        out = self.out_conv(x)
        return out
