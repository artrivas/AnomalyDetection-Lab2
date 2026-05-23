"""Reconstructive subnetwork for DRAEM."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """Two convolution layers with batch normalization and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Downsampling block used by the DRAEM encoders."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """Upsampling block for the reconstructive decoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return self.conv(x)


class ReconstructiveSubNetwork(nn.Module):
    """Encoder-decoder image reconstruction network.

    The network follows the DRAEM reference structure: a convolutional encoder
    compresses the anomalous input and a decoder reconstructs normal appearance.
    The final sigmoid constrains the reconstruction to image values in [0, 1].
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.encoder_1 = ConvBlock(in_channels, base_channels)
        self.encoder_2 = DownBlock(base_channels, base_channels * 2)
        self.encoder_3 = DownBlock(base_channels * 2, base_channels * 4)
        self.encoder_4 = DownBlock(base_channels * 4, base_channels * 8)
        self.bottleneck = DownBlock(base_channels * 8, base_channels * 8)

        self.decoder_4 = DecoderBlock(base_channels * 8, base_channels * 8)
        self.decoder_3 = DecoderBlock(base_channels * 8, base_channels * 4)
        self.decoder_2 = DecoderBlock(base_channels * 4, base_channels * 2)
        self.decoder_1 = DecoderBlock(base_channels * 2, base_channels)
        self.output_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_1(x)
        x = self.encoder_2(x)
        x = self.encoder_3(x)
        x = self.encoder_4(x)
        x = self.bottleneck(x)

        x = self.decoder_4(x)
        x = self.decoder_3(x)
        x = self.decoder_2(x)
        x = self.decoder_1(x)
        return torch.sigmoid(self.output_conv(x))
