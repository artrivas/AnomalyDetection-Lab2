"""DTD texture dataset used for synthetic anomaly generation."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


DTD_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class DTDTextureDataset(Dataset):
    """Recursively loads DTD texture images and returns random RGB crops."""

    def __init__(self, dtd_path: str | Path, image_size: int) -> None:
        self.dtd_path = Path(dtd_path)
        self.image_size = image_size

        if not self.dtd_path.is_dir():
            raise FileNotFoundError(f"DTD image path does not exist: {self.dtd_path}")

        self.image_paths = sorted(
            path
            for path in self.dtd_path.rglob("*")
            if path.is_file() and path.suffix.lower() in DTD_EXTENSIONS
        )
        if not self.image_paths:
            raise FileNotFoundError(
                f"No DTD texture images with extensions {sorted(DTD_EXTENSIONS)} "
                f"found under: {self.dtd_path}"
            )

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.2, 1.0),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tensor:
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            texture = image.convert("RGB")
            return self.transform(texture)
