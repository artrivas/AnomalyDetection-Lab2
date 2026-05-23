"""Discriminative segmentation subnetwork for DRAEM."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.reconstructive_subnetwork import ConvBlock, DownBlock


class UpBlock(nn.Module):
    """U-Net upsampling block with skip concatenation."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DiscriminativeSubNetwork(nn.Module):
    """U-Net-like segmentation network that returns raw anomaly logits."""

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 1,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.encoder_1 = ConvBlock(in_channels, base_channels)
        self.encoder_2 = DownBlock(base_channels, base_channels * 2)
        self.encoder_3 = DownBlock(base_channels * 2, base_channels * 4)
        self.encoder_4 = DownBlock(base_channels * 4, base_channels * 8)
        self.bottleneck = DownBlock(base_channels * 8, base_channels * 8)

        self.decoder_4 = UpBlock(base_channels * 8, base_channels * 8, base_channels * 8)
        self.decoder_3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.decoder_2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.decoder_1 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.output_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_1 = self.encoder_1(x)
        skip_2 = self.encoder_2(skip_1)
        skip_3 = self.encoder_3(skip_2)
        skip_4 = self.encoder_4(skip_3)
        x = self.bottleneck(skip_4)

        x = self.decoder_4(x, skip_4)
        x = self.decoder_3(x, skip_3)
        x = self.decoder_2(x, skip_2)
        x = self.decoder_1(x, skip_1)
        return self.output_conv(x)
