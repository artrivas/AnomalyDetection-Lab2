"""Focal and combined DRAEM losses."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.losses.dice_loss import DiceLoss
from src.losses.ssim_loss import ReconstructionLoss


class FocalLoss(nn.Module):
    """Binary focal loss over raw logits."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Invalid reduction '{reduction}'.")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        probability = torch.sigmoid(logits)
        p_t = probability * target + (1.0 - probability) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        focal_loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class SegmentationLoss(nn.Module):
    """BCEWithLogits + Dice loss with optional focal term."""

    def __init__(
        self,
        use_focal: bool = False,
        focal_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.use_focal = use_focal
        self.focal_weight = focal_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(
        self,
        logits: torch.Tensor,
        synthetic_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bce_loss = self.bce_loss(logits, synthetic_mask.float())
        dice_loss = self.dice_loss(logits, synthetic_mask)
        focal_loss = self.focal_loss(logits, synthetic_mask)

        total = bce_loss + dice_loss
        if self.use_focal:
            total = total + self.focal_weight * focal_loss

        return {
            "segmentation_loss": total,
            "bce_loss": bce_loss,
            "dice_loss": dice_loss,
            "focal_loss": focal_loss,
        }


class DRAEMLoss(nn.Module):
    """Combined DRAEM reconstruction and segmentation objective."""

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        segmentation_weight: float = 1.0,
        ssim_weight: float = 0.0,
        use_focal: bool = False,
        focal_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.segmentation_weight = segmentation_weight
        self.reconstruction_loss = ReconstructionLoss(ssim_weight=ssim_weight)
        self.segmentation_loss = SegmentationLoss(
            use_focal=use_focal,
            focal_weight=focal_weight,
        )

    def forward(
        self,
        reconstruction: torch.Tensor,
        clean_image: torch.Tensor,
        logits: torch.Tensor,
        synthetic_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        reconstruction_terms = self.reconstruction_loss(reconstruction, clean_image)
        segmentation_terms = self.segmentation_loss(logits, synthetic_mask)

        total_loss = (
            self.reconstruction_weight * reconstruction_terms["reconstruction_loss"]
            + self.segmentation_weight * segmentation_terms["segmentation_loss"]
        )

        return {
            "total_loss": total_loss,
            **reconstruction_terms,
            **segmentation_terms,
        }
