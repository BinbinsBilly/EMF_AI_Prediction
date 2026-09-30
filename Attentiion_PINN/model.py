import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class AxialSpectralConv(nn.Module):
    """
    Axial spectral convolution: two sequential 1D spectral convolutions,
    first along the y-axis (dim=-2), then along the x-axis (dim=-1).
    Each stage performs full channel mixing (stage 1: in->out, stage 2: out->out),
    so the composite operator is not a rank-1 separable kernel.
    """
    def __init__(self, in_channels, out_channels, modes):
        super(AxialSpectralConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # float32 storage (real/imag in trailing dim) for AMP GradScaler compatibility
        self.weights_y = nn.Parameter(
            (1 / (in_channels * out_channels))
            * torch.rand(in_channels, out_channels, modes, 2, dtype=torch.float32)
        )
        self.weights_x = nn.Parameter(
            (1 / (out_channels * out_channels))
            * torch.rand(out_channels, out_channels, modes, 2, dtype=torch.float32)
        )

    def forward(self, x):
        # Disable AMP for FFT/Complex operations to avoid ComplexHalf errors
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            B, _, H, W = x.shape
            m = self.modes

            # --- Stage 1: spectral conv along y (dim=-2) ---
            w_y = torch.view_as_complex(self.weights_y)            # (C_in, C_out, m)
            x_ft = torch.fft.rfft(x, dim=-2)                       # (B, C_in, H//2+1, W)
            out_ft = torch.zeros(B, self.out_channels, H // 2 + 1, W,
                                 dtype=x_ft.dtype, device=x.device)
            out_ft[:, :, :m, :] = torch.einsum("bifw,iof->bofw",
                                               x_ft[:, :, :m, :], w_y)
            x = torch.fft.irfft(out_ft, n=H, dim=-2)               # (B, C_out, H, W)

            # --- Stage 2: spectral conv along x (dim=-1) ---
            w_x = torch.view_as_complex(self.weights_x)            # (C_out, C_out, m)
            x_ft = torch.fft.rfft(x, dim=-1)                       # (B, C_out, H, W//2+1)
            out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                                 dtype=x_ft.dtype, device=x.device)
            out_ft[:, :, :, :m] = torch.einsum("bihf,iof->bohf",
                                               x_ft[:, :, :, :m], w_x)
            x = torch.fft.irfft(out_ft, n=W, dim=-1)               # (B, C_out, H, W)

            return x

class LightweightUNetEncoder(nn.Module):
    """
    Lightweight U-Net Encoder for Source Channel.
    Outputs feature map of same resolution as input.
    """
    def __init__(self, in_channels=1, base_channels=16):
        super(LightweightUNetEncoder, self).__init__()
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU()
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels*2),
            nn.GELU()
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels*4),
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
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.spatial_mix(x)
        return x

class PixelLayerNorm(nn.Module):
    """
    LayerNorm over the channel dimension for (B, C, H, W) feature maps,
    applied independently per pixel.
    """
    def __init__(self):
        super(PixelLayerNorm, self).__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)                # (B, H, W, C)
        x = F.layer_norm(x, (C,))
        return x.permute(0, 3, 1, 2)             # (B, C, H, W)

class CrossAttentionBlock(nn.Module):
    """
    Cross Attention between two feature maps with:
      - pre-norm (per-pixel channel LayerNorm) before the Q / K / V projections
      - 4-axis RoPE with learnable frequencies, applied to Q and K:
          axis 0: row coordinate y
          axis 1: column coordinate x
          axis 2: distance to source electrode
          axis 3: distance to boundary
      - K/V spatial downsampling via adaptive average pooling
    head_dim is split into 4 axis segments; within each segment adjacent dims
    are paired and rotated (requires head_dim % 8 == 0 so that each axis holds
    head_dim//4 dims = head_dim//8 frequency pairs).
    """
    def __init__(self, dim, num_heads=2, downsample_factor=16):
        super(CrossAttentionBlock, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim % 8 == 0, \
            "head_dim must be divisible by 8 (4 axes x paired rotations)"
        self.scale = self.head_dim ** -0.5
        self.downsample_factor = downsample_factor

        # Pre-norms (no learnable parameters)
        self.norm_q = PixelLayerNorm()
        self.norm_kv = PixelLayerNorm()

        # 1x1 conv projections
        self.q_proj = nn.Conv2d(dim, dim, 1)
        self.k_proj = nn.Conv2d(dim, dim, 1)
        self.v_proj = nn.Conv2d(dim, dim, 1)
        self.out_proj = nn.Conv2d(dim, dim, 1)

        # Learnable RoPE frequencies, log-parameterized: theta = exp(param)
        # Shape: (4 axes, head_dim//8 pairs per axis)
        pairs = self.head_dim // 8
        # Geometric wavelength series (224, 28, 3.5, ... px) covering
        # ~4px to ~256px spatial scales; theta_init = 2*pi / wavelength
        wavelengths = 224.0 / (8.0 ** torch.arange(pairs, dtype=torch.float32))
        theta_init = 2.0 * math.pi / wavelengths
        self.rope_log_freqs = nn.Parameter(
            torch.log(theta_init).unsqueeze(0).repeat(4, 1)
        )

    def get_rope_frequencies(self):
        """Return RoPE frequencies theta = exp(rope_log_freqs), shape (4, pairs)."""
        return torch.exp(self.rope_log_freqs)

    def _build_positions(self, B, H, W, pos_dist, device, offset, stride):
        """
        Build position tensor (B, H*W, 4) for the 4 RoPE axes.
        pos_dist: (B, 2, H, W) or None; ch0 = dist to source, ch1 = dist to boundary.
        offset/stride map grid indices to pixel coordinates:
          Q (full res): offset=0.0, stride=1.0   -> raw indices
          K (pooled):   offset=0.5, stride=downsample_factor -> pooled cell centers
        """
        ys = (torch.arange(H, device=device, dtype=torch.float32) + offset) * stride
        xs = (torch.arange(W, device=device, dtype=torch.float32) + offset) * stride
        y_grid = ys.view(1, H, 1).expand(B, H, W)
        x_grid = xs.view(1, 1, W).expand(B, H, W)
        if pos_dist is not None:
            d_src = pos_dist[:, 0].float()
            d_bnd = pos_dist[:, 1].float()
        else:
            d_src = torch.zeros(B, H, W, device=device, dtype=torch.float32)
            d_bnd = torch.zeros(B, H, W, device=device, dtype=torch.float32)
        pos = torch.stack([y_grid, x_grid, d_src, d_bnd], dim=-1)  # (B, H, W, 4)
        return pos.reshape(B, H * W, 4)

    def _apply_rope(self, t, pos):
        """
        Rotate adjacent dim pairs of t with per-axis learnable frequencies.
        t:   (B, heads, N, head_dim)
        pos: (B, N, 4)
        For a pair (d0, d1), position pos and frequency theta:
            d0' = d0*cos(pos*theta) - d1*sin(pos*theta)
            d1' = d0*sin(pos*theta) + d1*cos(pos*theta)
        """
        B, hd, N, D = t.shape
        seg = D // 4          # dims per axis
        pairs = seg // 2      # frequency pairs per axis
        t = t.reshape(B, hd, N, 4, pairs, 2)
        d0 = t[..., 0]        # (B, hd, N, 4, pairs)
        d1 = t[..., 1]

        freqs = torch.exp(self.rope_log_freqs)          # (4, pairs)
        angles = pos.unsqueeze(-1) * freqs              # (B, N, 4, pairs)
        cos = torch.cos(angles).unsqueeze(1)            # (B, 1, N, 4, pairs)
        sin = torch.sin(angles).unsqueeze(1)

        d0_new = d0 * cos - d1 * sin
        d1_new = d0 * sin + d1 * cos
        out = torch.stack([d0_new, d1_new], dim=-1)     # (B, hd, N, 4, pairs, 2)
        return out.reshape(B, hd, N, D)

    def forward(self, x, context, pos_dist=None):
        """
        x:        query features       (B, C, H, W)
        context:  key/value features   (B, C, H, W)
        pos_dist: (B, 2, H, W) or None; ch0 = dist to source, ch1 = dist to boundary
        """
        B, C, H, W = x.shape
        device = x.device

        # Downsample context (Key/Value) to reduce memory complexity
        if self.downsample_factor > 1:
            h_k, w_k = H // self.downsample_factor, W // self.downsample_factor
            context_pooled = F.adaptive_avg_pool2d(context, (h_k, w_k))
        else:
            context_pooled = context
            h_k, w_k = H, W

        # Projections with pre-norm; layout (B, heads, N, head_dim)
        q = self.q_proj(self.norm_q(x)) \
                .view(B, self.num_heads, self.head_dim, H * W).transpose(-2, -1)
        k = self.k_proj(self.norm_kv(context_pooled)) \
                .view(B, self.num_heads, self.head_dim, h_k * w_k).transpose(-2, -1)
        v = self.v_proj(self.norm_kv(context_pooled)) \
                .view(B, self.num_heads, self.head_dim, h_k * w_k).transpose(-2, -1)

        # Positions: Q at full resolution, K at pooled cell centers
        q_pos = self._build_positions(B, H, W, pos_dist, device,
                                      offset=0.0, stride=1.0)
        k_dist = None
        if pos_dist is not None:
            k_dist = F.adaptive_avg_pool2d(pos_dist, (h_k, w_k))
        k_pos = self._build_positions(B, h_k, w_k, k_dist, device,
                                      offset=0.5, stride=float(self.downsample_factor))

        # RoPE on Q and K (relative positions show up in the dot product)
        q = self._apply_rope(q, q_pos)
        k = self._apply_rope(k, k_pos)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale    # (B, heads, N, S)
        attn = attn.softmax(dim=-1)
        out = attn @ v                                    # (B, heads, N, head_dim)
        out = out.transpose(-2, -1).reshape(B, C, H, W)

        return self.out_proj(out)

class DualEncoderFNODecoder(nn.Module):
    def __init__(self, modes=32, width=32):
        super(DualEncoderFNODecoder, self).__init__()

        self.modes = modes
        self.width = width

        # Encoders
        self.source_encoder = LightweightUNetEncoder(in_channels=1, base_channels=width)
        self.geo_encoder = GeometryEmbedding(in_channels=2, out_channels=width)

        # Cross Attention (num_heads=2 -> head_dim=16 for width=32)
        # Source attending to Geometry; K/V downsample factor 16 keeps
        # the attention map small for 4GB GPUs.
        self.cross_attn_s2g = CrossAttentionBlock(width, num_heads=2, downsample_factor=16)
        # Geometry attending to Source
        self.cross_attn_g2s = CrossAttentionBlock(width, num_heads=2, downsample_factor=16)

        # Fusion
        self.fusion_conv = nn.Conv2d(width * 2, width, 1)

        # FNO Decoder (axial spectral convolutions)
        self.fno1 = AxialSpectralConv(width, width, modes)
        self.fno2 = AxialSpectralConv(width, width, modes)
        self.fno3 = AxialSpectralConv(width, width, modes)
        self.fno4 = AxialSpectralConv(width, width, modes)

        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        self.w4 = nn.Conv2d(width, width, 1)

        self.out_conv = nn.Sequential(
            nn.Conv2d(width, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x, pos_dist=None):
        # x: (B, 3, H, W) one-hot; pos_dist: (B, 2, H, W) or None
        # (ch0 = dist to source, ch1 = dist to boundary)
        # Split channels
        source = x[:, 0:1, :, :] # (B, 1, H, W)
        geo = x[:, 1:3, :, :]    # (B, 2, H, W)

        # Encode
        feat_s = self.source_encoder(source) # (B, width, H, W)
        feat_g = self.geo_encoder(geo)       # (B, width, H, W)

        # Cross Attention Interaction
        # Source queries Geometry
        attn_s = self.cross_attn_s2g(feat_s, feat_g, pos_dist)
        # Geometry queries Source
        attn_g = self.cross_attn_g2s(feat_g, feat_s, pos_dist)

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
