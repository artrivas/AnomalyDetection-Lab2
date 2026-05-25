"""Reusable anomaly-map smoothing and binary-mask post-processing."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def smooth_numpy_map(anomaly_map: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return anomaly_map.astype(np.float32)
    tensor = torch.from_numpy(anomaly_map).view(1, 1, *anomaly_map.shape).float()
    return gaussian_smooth(tensor, sigma)[0, 0].numpy().astype(np.float32)


def gaussian_smooth(anomaly_map: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return anomaly_map

    radius = max(1, int(3 * sigma))
    coords = torch.arange(
        -radius,
        radius + 1,
        device=anomaly_map.device,
        dtype=anomaly_map.dtype,
    )
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_x = kernel_1d.view(1, 1, 1, -1)
    kernel_y = kernel_1d.view(1, 1, -1, 1)

    smoothed = F.pad(anomaly_map, (radius, radius, 0, 0), mode="reflect")
    smoothed = F.conv2d(smoothed, kernel_x)
    smoothed = F.pad(smoothed, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(smoothed, kernel_y)


def postprocess_mask(
    mask: np.ndarray,
    min_component_area: int,
    closing_kernel_size: int,
) -> np.ndarray:
    processed = morphological_closing(mask.astype(bool), closing_kernel_size)
    processed = remove_small_components(processed, min_component_area)
    return processed.astype(np.uint8)


def morphological_closing(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask.astype(bool)
    return binary_erosion(binary_dilation(mask, kernel_size), kernel_size)


def binary_dilation(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    radius = kernel_size // 2
    padded = np.pad(mask.astype(bool), radius, mode="constant", constant_values=False)
    output = np.zeros_like(mask, dtype=bool)
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            output |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return output


def binary_erosion(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    radius = kernel_size // 2
    padded = np.pad(mask.astype(bool), radius, mode="constant", constant_values=False)
    output = np.ones_like(mask, dtype=bool)
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            output &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return output


def remove_small_components(mask: np.ndarray, min_component_area: int) -> np.ndarray:
    if min_component_area <= 1:
        return mask.astype(bool)

    mask = mask.astype(bool)
    visited = np.zeros_like(mask, dtype=bool)
    output = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue

            component = []
            stack = [(y, x)]
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            if len(component) >= min_component_area:
                for cy, cx in component:
                    output[cy, cx] = True

    return output
