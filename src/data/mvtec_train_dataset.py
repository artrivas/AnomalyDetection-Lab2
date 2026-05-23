"""Training and validation dataset for normal MVTec AD images."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.mvtec_paths import IMAGE_EXTENSIONS, MVTEC_CLASSES
from src.data.synthetic_anomaly import SyntheticAnomalyGenerator


class MVTecTrainDataset(Dataset):
    """Normal-image dataset with synthetic anomalies for the training split."""

    VALID_SPLITS = {"train", "val", "validation"}

    def __init__(
        self,
        class_name: str,
        dataset_path: str | Path,
        dtd_path: str | Path,
        image_size: int,
        split: str,
        train_val_split: float,
        seed: int,
        p_no_anomaly: float = 0.0,
    ) -> None:
        if class_name not in MVTEC_CLASSES:
            valid_classes = ", ".join(MVTEC_CLASSES)
            raise ValueError(
                f"Unknown MVTec class '{class_name}'. Expected one of: {valid_classes}"
            )
        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. Expected one of: {sorted(self.VALID_SPLITS)}"
            )
        if not 0.0 < train_val_split < 1.0:
            raise ValueError(
                f"train_val_split must be between 0 and 1, got: {train_val_split}"
            )

        self.class_name = class_name
        self.dataset_path = Path(dataset_path)
        self.dtd_path = Path(dtd_path)
        self.image_size = image_size
        self.split = "val" if split == "validation" else split
        self.seed = seed

        train_good_root = self.dataset_path / class_name / "train" / "good"
        if not train_good_root.is_dir():
            raise FileNotFoundError(
                f"Training normal-image folder is missing for '{class_name}': "
                f"{train_good_root}"
            )

        image_paths = sorted(
            path
            for path in train_good_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise FileNotFoundError(
                f"No normal training images found for '{class_name}' under: "
                f"{train_good_root}"
            )

        split_rng = random.Random(seed)
        shuffled_paths = image_paths[:]
        split_rng.shuffle(shuffled_paths)

        split_index = int(len(shuffled_paths) * train_val_split)
        split_index = max(1, min(split_index, len(shuffled_paths) - 1))

        if self.split == "train":
            self.image_paths = sorted(shuffled_paths[:split_index])
            self.synthetic_generator = SyntheticAnomalyGenerator(
                self.dtd_path,
                image_size,
                p_no_anomaly=p_no_anomaly,
                seed=seed,
            )
        else:
            self.image_paths = sorted(shuffled_paths[split_index:])
            self.synthetic_generator = None

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.image_paths[index]
        clean_image = self._load_image(image_path)

        if self.split == "val":
            return {
                "image": clean_image,
                "image_path": str(image_path),
                "class_name": self.class_name,
            }

        if self.synthetic_generator is None:
            raise RuntimeError("Synthetic anomaly generation is only available for train split.")

        clean_image, anomalous_image, anomaly_mask = self.synthetic_generator(clean_image)

        return {
            "clean_image": clean_image,
            "anomalous_image": anomalous_image,
            "anomaly_mask": anomaly_mask,
            "image_path": str(image_path),
            "class_name": self.class_name,
            "is_anomalous": True,
        }

    def _load_image(self, image_path: Path) -> Tensor:
        with Image.open(image_path) as image:
            return self.image_transform(image.convert("RGB"))
