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




class DeepLSTMTemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=71, num_layers=2, dropout=0.5):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # 1D conv over the temporal (sequence) axis
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.final_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [batch, d, l] -> [batch, l, d]
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)  # [batch, l, hidden_dim]

        # Temporal encoding: [batch, hidden_dim, l]
        encoded = self.encoder(lstm_out.transpose(1, 2))  # → [batch, hidden_dim, l]
        pooled = torch.mean(encoded, dim=2)  # → [batch, hidden_dim]

        logits = self.fc(self.final_dropout(pooled))
        return logits  # Use with BCEWithLogitsLoss