"""Reconstruction losses for DRAEM."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss implemented with a uniform local window."""

    def __init__(
        self,
        window_size: int = 11,
        c1: float = 0.01**2,
        c2: float = 0.03**2,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.c1 = c1
        self.c2 = c2
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"SSIMLoss expects matching shapes, got {prediction.shape} and "
                f"{target.shape}."
            )

        channels = prediction.shape[1]
        kernel = _uniform_kernel(
            channels,
            self.window_size,
            device=prediction.device,
            dtype=prediction.dtype,
        )
        padding = self.window_size // 2

        mu_prediction = F.conv2d(prediction, kernel, padding=padding, groups=channels)
        mu_target = F.conv2d(target, kernel, padding=padding, groups=channels)
        mu_prediction_sq = mu_prediction.pow(2)
        mu_target_sq = mu_target.pow(2)
        mu_product = mu_prediction * mu_target

        sigma_prediction_sq = (
            F.conv2d(prediction * prediction, kernel, padding=padding, groups=channels)
            - mu_prediction_sq
        )
        sigma_target_sq = (
            F.conv2d(target * target, kernel, padding=padding, groups=channels)
            - mu_target_sq
        )
        sigma_product = (
            F.conv2d(prediction * target, kernel, padding=padding, groups=channels)
            - mu_product
        )

        numerator = (2 * mu_product + self.c1) * (2 * sigma_product + self.c2)
        denominator = (
            (mu_prediction_sq + mu_target_sq + self.c1)
            * (sigma_prediction_sq + sigma_target_sq + self.c2)
            + self.eps
        )
        ssim = numerator / denominator
        return torch.clamp((1.0 - ssim) / 2.0, min=0.0, max=1.0).mean()


class ReconstructionLoss(nn.Module):
    """Weighted MSE + SSIM reconstruction loss."""

    def __init__(self, mse_weight: float = 1.0, ssim_weight: float = 0.0) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.ssim_loss = SSIMLoss()

    def forward(
        self,
        reconstruction: torch.Tensor,
        clean_image: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mse_loss = F.mse_loss(reconstruction, clean_image)
        ssim_loss = self.ssim_loss(reconstruction, clean_image)
        total = self.mse_weight * mse_loss + self.ssim_weight * ssim_loss
        return {
            "reconstruction_loss": total,
            "mse_loss": mse_loss,
            "ssim_loss": ssim_loss,
        }


def _uniform_kernel(
    channels: int,
    window_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    kernel = torch.ones(
        (channels, 1, window_size, window_size),
        device=device,
        dtype=dtype,
    )
    return kernel / float(window_size * window_size)
