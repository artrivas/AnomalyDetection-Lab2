"""Test dataset for MVTec AD anomaly detection and segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.mvtec_paths import MVTEC_CLASSES, list_test_images


class MVTecTestDataset(Dataset):
    """Loads normal and anomalous test images with binary masks."""

    def __init__(
        self,
        class_name: str,
        dataset_path: str | Path,
        image_size: int,
    ) -> None:
        if class_name not in MVTEC_CLASSES:
            valid_classes = ", ".join(MVTEC_CLASSES)
            raise ValueError(
                f"Unknown MVTec class '{class_name}'. Expected one of: {valid_classes}"
            )

        self.class_name = class_name
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        self.samples = list_test_images(class_name, self.dataset_path)

        if not self.samples:
            raise FileNotFoundError(
                f"No test images found for '{class_name}' under: "
                f"{self.dataset_path / class_name / 'test'}"
            )

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.NEAREST,
                ),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image_path = Path(sample["image_path"])
        defect_type = sample["defect_type"]
        is_anomalous = sample["is_anomalous"]

        image = self._load_image(image_path)
        if is_anomalous:
            mask_path = Path(sample["expected_mask_path"])
            mask = self._load_mask(mask_path)
        else:
            mask = torch.zeros((1, self.image_size, self.image_size), dtype=image.dtype)

        return {
            "image": image,
            "mask": mask,
            "image_path": str(image_path),
            "defect_type": defect_type,
            "is_anomalous": is_anomalous,
        }

    def _load_image(self, image_path: Path) -> Tensor:
        with Image.open(image_path) as image:
            return self.image_transform(image.convert("RGB"))

    def _load_mask(self, mask_path: Path) -> Tensor:
        if not mask_path.is_file():
            raise FileNotFoundError(f"Expected ground-truth mask is missing: {mask_path}")

        with Image.open(mask_path) as mask:
            mask_tensor = self.mask_transform(mask.convert("L"))
            return (mask_tensor > 0).float()
