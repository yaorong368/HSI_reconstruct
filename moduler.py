import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# import trimesh
# import pyrender
from scipy.spatial.transform import Rotation as R
from scipy.signal import convolve2d



# # === Spectral Encoder using 1x1 Conv ===
class SpectralEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU()
        )

    def forward(self, x):
        return self.encoder(x)

# === Residual Block ===
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, channels),   # <-- was BatchNorm2d
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, channels)    # <-- was BatchNorm2d
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))
    

# === Attention (Channel SE Block) ===
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale
    

# === Spatial Decoder with Upsample + Conv ===
class SpatialDecoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.block1 = ResidualBlock(in_channels // 2)

        self.conv2 = nn.Conv2d(in_channels // 2, in_channels * 3 // 4, kernel_size=3, padding=1)
        self.block2 = ResidualBlock(in_channels * 3 // 4)

        self.out_conv = nn.Conv2d(in_channels * 3 // 4, num_classes, kernel_size=1)

    def forward(self, x, target_size=None):
        # --- always upscale to 3N dynamically ---
        n = x.shape[-1]  # input spatial size
        x = F.interpolate(x, scale_factor=3, mode="bilinear", align_corners=False)  # [B, 64, 3N, 3N]

        x = F.gelu(self.conv1(x))
        x = self.block1(x)

        x = F.gelu(self.conv2(x))
        x = self.block2(x)

        x = self.out_conv(x)  # [B, num_classes, 3N, 3N]
        return x


# === Full Model ===
class SpectralSpatialHSIModel(nn.Module):
    def __init__(self, in_channels=30, num_classes=17):
        super().__init__()
        # Step 1: spectral encoder produces feature maps
        self.spectral = SpectralEncoder(in_channels, hidden_channels=128)

        # Step 2: attention
        self.attention = SEBlock(128)

        # Step 3: spatial residual processing
        self.spatial = nn.Sequential(
            ResidualBlock(128),
            # nn.Conv2d(128, 64, kernel_size=3, padding=1),
            ResidualBlock(128)
        )

        # Step 4: decoder upsamples to 3× and predicts classes
        self.decoder1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.decoder2 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
        
        # #---for zoom below
        # # Step 4: decoder upsamples to 3× and predicts classes
        # self.updecoder = SpatialDecoder(128, num_classes)

    def forward(self, x):
        x = self.spectral(x)     # [B, 64, N, N]
        x = self.attention(x)    
        x = self.spatial(x)      
        # x = self.updecoder(x)
        x = self.decoder1(x)      # [B, num_classes, 3N, 3N]
        x = self.decoder2(x) 
        return x
    

#-------------------
#-------------------
#-------------------
#-------------------spectral prediction 

    

# #----channel-transformer reconstruction
# #----channel-transformer reconstruction
# #----channel-transformer reconstruction
# #----channel-transformer reconstruction
# #----channel-transformer reconstruction
# #----channel-transformer reconstruction
# #----channel-transformer reconstruction
# #----channel-transformer reconstruction
# #----channel-transformer reconstruction
# #----channel-transformer reconstruction

class PerChannelTransformer(nn.Module):
    """
    Processes each channel independently:
    [B, C, H, W] -> [B, C, L, d_model], L = H*W
    """
    def __init__(self, in_channels, d_model, nhead, num_layers):
        super().__init__()
        self.in_channels = in_channels
        self.proj = nn.ModuleList([nn.Linear(1, d_model) for _ in range(in_channels)])
        self.encoders = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.0, batch_first=True),
                num_layers=num_layers
            ) for _ in range(in_channels)
        ])

    def forward(self, x):  # x: [B, C, H, W]
        B, C, H, W = x.shape
        L = H * W
        outs = []
        for c in range(C):
            # [B, H, W] -> [B, L, 1]
            xc = x[:, c, :, :].reshape(B, L, 1)
            # project -> [B, L, d_model]
            xc = self.proj[c](xc)
            # batch_first=True, so no need to permute
            xc = self.encoders[c](xc)  # [B, L, d_model]
            outs.append(xc)
        # [B, C, L, d_model]
        return torch.stack(outs, dim=1)


class ResBlock1D(nn.Module):
    def __init__(self, channels, k=3, gn_groups=8):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=k, padding=p)
        self.gn1   = nn.GroupNorm(min(gn_groups, channels), channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=k, padding=p)
        self.gn2   = nn.GroupNorm(min(gn_groups, channels), channels)
    def forward(self, x):  # [N, Dfeat, Cin]
        r = x
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return F.relu(x + r)

class PixelSpectrumClassifierHeadConv(nn.Module):
    """
    per_channel_out: [B, Cin, L, d_model]
    returns logits:  [B, L, Cclasses]
    """
    def __init__(self, Cin=18, d_model=64, num_classes=71, depth=3, k=3, dropout=0.0):
        super().__init__()
        self.blocks = nn.Sequential(*[ResBlock1D(d_model, k=k, dropout=dropout) for _ in range(depth)])
        # conv over band axis, then GAP to collapse band length
        self.proj = nn.Conv1d(d_model, num_classes, kernel_size=1)   # [N, num_classes, Cin]
        self.pool = nn.AdaptiveAvgPool1d(1)                          # -> [N, num_classes, 1]

    def forward(self, per_channel_out):
        B, Cin, L, D = per_channel_out.shape
        z = per_channel_out.permute(0, 2, 3, 1).reshape(B*L, D, Cin)  # [B*L, D, Cin]
        z = self.blocks(z)                                           # [B*L, D, Cin]
        z = self.proj(z)                                             # [B*L, K, Cin]
        z = self.pool(z).squeeze(-1)                                 # [B*L, K]
        logits = z.view(B, L, -1)                                    # [B, L, K]
        return logits

# --- modified reconstructor -> classifier ---
class HSIReconstructor(nn.Module):
    """
    Input : x  [B, Cin, H, W]
    Output: y  [B, K(num_classes), H, W]  (logits; apply softmax in loss if needed)
    """
    def __init__(self, in_channels=18, d_model=64, nhead=4, num_layers=2,
                 num_classes=71, depth=3, k=3, dropout=0.0):
        super().__init__()
        self.per_channel = PerChannelTransformer(in_channels, d_model, nhead, num_layers)
        self.head = PixelSpectrumClassifierHeadConv(Cin=in_channels, d_model=d_model,
                                                    num_classes=num_classes,
                                                    depth=depth, k=k, dropout=dropout)

    def forward(self, x):
        B, Cin, H, W = x.shape
        per_channel_out = self.per_channel(x)            # [B, Cin, L, d_model]
        logits_flat = self.head(per_channel_out)         # [B, L, K]
        y = logits_flat.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # [B, K, H, W]
        return y

#----------------------------------------
# ===================== Patch Embedding =====================
class SequentialPatchEmbed(nn.Module):
    def __init__(self, C, d_model=256, p=4):
        super().__init__()
        self.p = p
        self.proj = nn.Linear(C * p * p, d_model)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.p

        # 改用张量方式计算 pad（兼容 trace）
        H_t = torch.tensor(H, device=x.device)
        W_t = torch.tensor(W, device=x.device)
        pad_h = (p - (H_t % p)) % p
        pad_w = (p - (W_t % p)) % p

        pad_h_i = int(pad_h.item())
        pad_w_i = int(pad_w.item())
        x = F.pad(x, (0, pad_w_i, 0, pad_h_i))

        H_pad, W_pad = x.shape[2], x.shape[3]
        Hp, Wp = H_pad // p, W_pad // p

        patches = F.unfold(x, kernel_size=p, stride=p).transpose(1, 2)
        tokens = self.proj(patches)

        # 不再存 tensor→int 的 meta，仅存 Tensor 自身
        meta = {
            "H": H_t, "W": W_t,
            "H_pad": torch.tensor(H_pad), "W_pad": torch.tensor(W_pad),
            "Hp": torch.tensor(Hp), "Wp": torch.tensor(Wp),
            "p": torch.tensor(p)
        }
        return tokens, meta

# ===================== Transformer Encoder =====================
class ViTEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, depth=6, dim_ff=1024, dropout=0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, tokens):
        return self.encoder(tokens)  # [B, L, d_model]

# ===================== Decoder =====================
class SegDecoder(nn.Module):
    def __init__(self, d_model, num_class, p):
        super().__init__()
        self.p = int(p)
        self.up = nn.ConvTranspose2d(d_model, d_model, kernel_size=self.p, stride=self.p)
        self.head = nn.Conv2d(d_model, num_class, kernel_size=1)

    def forward(self, tokens, meta):
        """
        tokens: [B, L, d_model]
        meta: dict from embed
        """
        B, L, D = tokens.shape
        Hp, Wp = meta["Hp"], meta["Wp"]
        H, W = meta["H"], meta["W"]
        H_pad, W_pad = meta["H_pad"], meta["W_pad"]

        # [B,L,D]→[B,D,Hp,Wp]
        feat = tokens.transpose(1, 2).contiguous().view(B, D, Hp, Wp)
        feat = self.up(feat)             # [B, D, H_pad, W_pad]
        logits = self.head(feat)         # [B, num_class, H_pad, W_pad]
        logits = logits[:, :, :H, :W]    # 无条件裁剪（no-op 若相等）
        return logits

# ===================== Full Model =====================
class SequentialPatchViTForSegmentation(nn.Module):
    def __init__(self, C, num_class, H, W,
                 d_model=256, p=4, nhead=8, depth=6,
                 dim_ff=1024, dropout=0.0):
        super().__init__()
        self.embed = SequentialPatchEmbed(C, d_model, p)
        self.encoder = ViTEncoder(d_model, nhead, depth, dim_ff, dropout)
        self.decoder = SegDecoder(d_model, num_class, p)

    def forward(self, x):
        tokens, meta = self.embed(x)
        tokens = self.encoder(tokens)
        out = self.decoder(tokens, meta)
        return out  # [B, num_class, H, W]

