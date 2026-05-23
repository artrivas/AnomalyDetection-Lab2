"""Dice loss for anomaly segmentation."""

from __future__ import annotations

import torch
from torch import nn


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation logits."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape != target.shape:
            raise ValueError(
                f"DiceLoss expects matching shapes, got {logits.shape} and {target.shape}."
            )

        probabilities = torch.sigmoid(logits)
        probabilities = probabilities.flatten(start_dim=1)
        target = target.float().flatten(start_dim=1)

        intersection = (probabilities * target).sum(dim=1)
        denominator = probabilities.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        return 1.0 - dice.mean()
