from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as nnf


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = nnf.gelu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = nnf.gelu(x + residual)
        return x


class TransitResNet1D(nn.Module):
    def __init__(self, base_channels: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv1d(1, c, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(c),
            nn.GELU(),
        )
        self.layer1 = nn.Sequential(
            ResidualBlock1D(c, c, stride=1, dropout=dropout),
            ResidualBlock1D(c, c, stride=1, dropout=dropout),
        )
        self.layer2 = nn.Sequential(
            ResidualBlock1D(c, c * 2, stride=2, dropout=dropout),
            ResidualBlock1D(c * 2, c * 2, stride=1, dropout=dropout),
        )
        self.layer3 = nn.Sequential(
            ResidualBlock1D(c * 2, c * 4, stride=2, dropout=dropout),
            ResidualBlock1D(c * 4, c * 4, stride=1, dropout=dropout),
        )
        self.layer4 = nn.Sequential(
            ResidualBlock1D(c * 4, c * 4, stride=2, dropout=dropout),
            ResidualBlock1D(c * 4, c * 4, stride=1, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 4, c * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.head(x)
