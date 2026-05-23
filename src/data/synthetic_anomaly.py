"""DRAEM-style synthetic anomaly generation."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import functional as TF

from src.data.dtd_dataset import DTDTextureDataset


def generate_perlin_noise_2d(
    shape: tuple[int, int],
    res: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate 2D Perlin noise, adapted from the public DRAEM implementation."""

    height, width = shape
    res_y, res_x = res
    if height % res_y != 0 or width % res_x != 0:
        raise ValueError(f"Shape {shape} must be divisible by resolution {res}.")

    delta = (res_y / height, res_x / width)
    d = (height // res_y, width // res_x)
    grid = np.mgrid[0:res_y:delta[0], 0:res_x:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * rng.random((res_y + 1, res_x + 1))
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    gradients = gradients.repeat(d[0], axis=0).repeat(d[1], axis=1)

    g00 = gradients[:-d[0], :-d[1]]
    g10 = gradients[d[0] :, :-d[1]]
    g01 = gradients[:-d[0], d[1] :]
    g11 = gradients[d[0] :, d[1] :]

    n00 = np.sum(grid * g00, axis=2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, axis=2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, axis=2)
    n11 = np.sum((grid - 1) * g11, axis=2)

    t = _fade(grid)
    return math.sqrt(2) * _lerp(
        _lerp(n00, n10, t[:, :, 0]),
        _lerp(n01, n11, t[:, :, 0]),
        t[:, :, 1],
    )


class SyntheticAnomalyGenerator:
    """Creates synthetic DRAEM training tuples from clean images and DTD textures."""

    def __init__(
        self,
        dtd_path: str | Path,
        image_size: int,
        p_no_anomaly: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= p_no_anomaly <= 1.0:
            raise ValueError(f"p_no_anomaly must be between 0 and 1, got {p_no_anomaly}")

        self.dtd_dataset = DTDTextureDataset(dtd_path, image_size)
        self.image_size = image_size
        self.p_no_anomaly = p_no_anomaly
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.texture_transform = self._build_texture_transform()

    def __call__(self, clean_image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.rng.random() < self.p_no_anomaly:
            zero_mask = torch.zeros(
                (1, clean_image.shape[-2], clean_image.shape[-1]),
                dtype=clean_image.dtype,
                device=clean_image.device,
            )
            return clean_image, clean_image.clone(), zero_mask

        texture = self._sample_texture()
        synthetic_mask = self._generate_mask(
            clean_image.shape[-2],
            clean_image.shape[-1],
            dtype=clean_image.dtype,
            device=clean_image.device,
        )
        alpha = self.rng.uniform(0.1, 1.0)

        texture = texture.to(device=clean_image.device, dtype=clean_image.dtype)
        blended_region = (1.0 - alpha) * clean_image + alpha * texture
        synthetic_image = (
            clean_image * (1.0 - synthetic_mask) + blended_region * synthetic_mask
        )
        return clean_image, synthetic_image.clamp(0.0, 1.0), synthetic_mask

    def _sample_texture(self) -> Tensor:
        texture_index = self.rng.randrange(len(self.dtd_dataset))
        texture = self.dtd_dataset[texture_index]
        texture_image = TF.to_pil_image(texture)
        return self.texture_transform(texture_image)

    def _generate_mask(
        self,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        for _ in range(10):
            scale_y = 2 ** self.rng.randint(0, 4)
            scale_x = 2 ** self.rng.randint(0, 4)
            perlin = generate_perlin_noise_2d((height, width), (scale_y, scale_x), self.np_rng)
            perlin = _normalize(perlin)

            threshold = self.rng.uniform(0.45, 0.65)
            mask = (perlin > threshold).astype(np.float32)
            if mask.sum() > 0:
                break
        else:
            mask = np.zeros((height, width), dtype=np.float32)

        if self.rng.random() < 0.5:
            mask = np.fliplr(mask).copy()
        if self.rng.random() < 0.5:
            mask = np.flipud(mask).copy()

        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(dtype=dtype, device=device)
        return (mask_tensor > 0).float()

    def _build_texture_transform(self) -> transforms.Compose:
        augmentations: list[object] = [
            transforms.ColorJitter(
                brightness=(0.6, 1.4),
                contrast=(0.6, 1.4),
                saturation=(0.6, 1.4),
                hue=(-0.05, 0.05),
            ),
            transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),
            transforms.RandomRotation(
                degrees=180,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
        ]

        if hasattr(transforms, "RandomPosterize"):
            augmentations.append(transforms.RandomPosterize(bits=4, p=0.2))
        if hasattr(transforms, "RandomSolarize"):
            augmentations.append(transforms.RandomSolarize(threshold=128, p=0.2))

        augmentations.append(transforms.ToTensor())
        return transforms.Compose(augmentations)


def _fade(t: np.ndarray) -> np.ndarray:
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def _lerp(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    return a + x * (b - a)


def _normalize(values: np.ndarray) -> np.ndarray:
    min_value = values.min()
    max_value = values.max()
    if max_value == min_value:
        return np.zeros_like(values)
    return (values - min_value) / (max_value - min_value)
