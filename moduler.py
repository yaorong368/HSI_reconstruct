import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# import trimesh
# import pyrender
from scipy.spatial.transform import Rotation as R
from scipy.signal import convolve2d





class CubeModel(nn.Module):
    def __init__(self, input_channels, output_dim, input_size=16):  # default now 16
        super(CubeModel, self).__init__()
        self.output_dim = output_dim

        # Conv Block 1
        self.conv1_1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> 8x8

        # Conv Block 2
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> 4x4

        # Conv Block 3
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> 2x2

        # Conv Block 4 (no pooling here)
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4_2 = nn.BatchNorm2d(256)

        # Conv Block 5
        self.conv5_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5_1 = nn.BatchNorm2d(256)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5_2 = nn.BatchNorm2d(256)

        self.dropout = nn.Dropout(p=0.1)

        # Flatten size
        flatten_size = self._get_flatten_size(input_channels, input_size)

        self.fc1 = nn.Linear(flatten_size, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, output_dim)

    def _get_flatten_size(self, input_channels, input_size):
        with torch.no_grad():
            x = torch.zeros(1, input_channels, input_size, input_size)
            was_training = self.training
            self.eval()  # Avoid batch norm crash

            x = F.relu(self.bn1_1(self.conv1_1(x)))
            x = F.relu(self.bn1_2(self.conv1_2(x)))
            x = self.pool1(x)

            x = F.relu(self.bn2_1(self.conv2_1(x)))
            x = F.relu(self.bn2_2(self.conv2_2(x)))
            x = self.pool2(x)

            x = F.relu(self.bn3_1(self.conv3_1(x)))
            x = F.relu(self.bn3_2(self.conv3_2(x)))
            x = self.pool3(x)

            x = F.relu(self.bn4_1(self.conv4_1(x)))
            x = F.relu(self.bn4_2(self.conv4_2(x)))

            x = F.relu(self.bn5_1(self.conv5_1(x)))
            x = F.relu(self.bn5_2(self.conv5_2(x)))

            flatten_size = x.view(1, -1).size(1)
            if was_training:
                self.train()
        print(f"[INFO] Flatten size calculated: {flatten_size}")
        return flatten_size

    def forward(self, x):
        batch_size = x.size(0)

        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = self.dropout(x)

        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = self.dropout(x)

        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        x = self.dropout(x)

        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn5_1(self.conv5_1(x)))
        x = F.relu(self.bn5_2(self.conv5_2(x)))
        x = self.dropout(x)

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)

        x = self.fc3(x)
        return x





class SpectralAttention(nn.Module):
    def __init__(self, channels):
        super(SpectralAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W]
        w = x.mean(dim=(2, 3))              # Global avg pool → [B, C]
        w = self.fc(w).unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        return x * w                        # Channel-wise attention

class SpectralCubeNetV2(nn.Module):
    def __init__(self, in_channels=30, num_classes=71):
        super(SpectralCubeNetV2, self).__init__()

        # 1. Spectral-Spatial feature extractor using 3D convolutions
        self.conv3d_block = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(32, 64, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )

        # 2. Collapse spectral dimension
        self.pool = nn.AdaptiveAvgPool3d((1, 8, 8))  # Output shape: [B, 128, 1, 8, 8]

        # 3. Flatten and apply spectral attention
        self.attention = SpectralAttention(128)

        # 4. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)  # For BCEWithLogitsLoss
        )

    def forward(self, x):
        # Input shape: [B, C, H, W]
        B, C, H, W = x.shape
        x = x.unsqueeze(1)  # → [B, 1, C, H, W]
        
        x = self.conv3d_block(x)  # → [B, 128, C', H', W']
        x = self.pool(x)          # → [B, 128, 1, 8, 8]
        x = x.squeeze(2)          # → [B, 128, 8, 8]

        x = self.attention(x)     # Apply spectral attention

        x = x.flatten(start_dim=1)  # → [B, 128*8*8]
        x = self.classifier(x)      # → [B, num_classes]
        return x


#---------------
# # === Denoising Frontend ===
# class DenoisingHead(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.head = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.head(x)


# # === Updated SpectralTransformer ===

# class SpatialRefiner(nn.Module):
#     def __init__(self, in_channels, kernel_size=5):
#         super().__init__()
#         self.refine = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=in_channels),  # depthwise smoothing
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.refine(x)

# class SpectralTransformer(nn.Module):
#     def __init__(self, in_channels=20, embed_dim=64, num_heads=4, num_layers=4):
#         super().__init__()
#         self.linear_proj = nn.Linear(in_channels, embed_dim)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim,
#             nhead=num_heads,
#             dim_feedforward=embed_dim * 4,  # stronger internal representation
#             batch_first=True,
#             norm_first=False
#         )
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = x.permute(0, 2, 3, 1).reshape(B * H * W, C)  # [BHW, C]
#         x = self.linear_proj(x).unsqueeze(1)            # [BHW, 1, D]
#         x = self.encoder(x.repeat(1, C, 1))              # [BHW, C, D], C-length sequence for each pixel
#         x = x.mean(dim=1).reshape(B, H, W, -1).permute(0, 3, 1, 2)  # [B, D, H, W]
#         return x

# # === Residual Block ===
# class ResidualBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(channels)
#         )
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         return self.relu(x + self.block(x))

# # === Full Model ===
# class HSIReconstructor(nn.Module):
#     def __init__(self, in_channels=30, num_classes=1):
#         super().__init__()
#         self.denoiser = DenoisingHead(in_channels)

#         self.initial = nn.Sequential(
#             SpectralTransformer(in_channels=in_channels, embed_dim=128),
#             nn.BatchNorm2d(128),
#             nn.ReLU()
#         )

#         self.encoder = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             ResidualBlock(128),
#             ResidualBlock(128)
#         )

#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             ResidualBlock(128)
#         )

#         self.refiner = SpatialRefiner(128)

#         # === Upsample to 100×100 ===
#         self.final = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 25x25 → 50x50
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 50x50 → 100x100
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(32, num_classes, kernel_size=1)                         # → [B, 1, 100, 100]
#         )

#     def forward(self, x):
#         if x.ndim == 3:
#             x = x.unsqueeze(0)  # [C, H, W] → [1, C, H, W]

#         x = self.denoiser(x)       # [1, 30, 25, 25]
#         x = self.initial(x)        # [1, 128, 25, 25]
#         x = self.encoder(x)        # [1, 128, 25, 25]
#         x = self.bottleneck(x)     # [1, 128, 25, 25]
#         x = self.refiner(x)        # [1, 128, 25, 25]
#         x = self.final(x)          # [1, 1, 100, 100]
#         return x.squeeze(0)        # → [1, 100, 100]

###------no transformer


# === Spectral Encoder using 1x1 Conv ===
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
        # self.block = nn.Sequential(
        #     nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(channels),
        #     nn.GELU(),
        #     nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(channels)
        # )
        
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
        self.decode = nn.Sequential(
            # Upsample 50 → 100
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # [B, 64, 50, 50] → [B, 64, 100, 100]
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1), # 64 → 32
            nn.GELU(),
            ResidualBlock(in_channels // 2),                                   # → [B, 32, 100, 100]

            # Optional: increase channel to 48 or 64 if needed
            nn.Conv2d(in_channels // 2, in_channels * 3 // 4, kernel_size=3, padding=1),  # 32 → 48
            nn.GELU(),
            ResidualBlock(in_channels * 3 // 4),                                          # → [B, 48, 100, 100]

            # Final classifier
            nn.Conv2d(in_channels * 3 // 4, num_classes, kernel_size=1)  # [B, 48, 100, 100] → [B, num_classes, 100, 100]
        )

    def forward(self, x):
        return self.decode(x)

# === Full Model ===
class SpectralSpatialHSIModel(nn.Module):
    def __init__(self, in_channels=30, num_classes=1):
        super().__init__()
        self.spectral = SpectralEncoder(in_channels, hidden_channels=num_classes)
        self.attention = SEBlock(num_classes)
        self.spatial = nn.Sequential(
            ResidualBlock(num_classes),
            ResidualBlock(num_classes)
        )
        # self.decoder = SpatialDecoder(64, num_classes)

    def forward(self, x):
     
        x = self.spectral(x)         # [B, 64, 25, 25]
        x = self.attention(x)        # SE Attention
        x = self.spatial(x)          # residual processing
        # x = self.decoder(x)          # [B, 1, 100, 100]
        return x

