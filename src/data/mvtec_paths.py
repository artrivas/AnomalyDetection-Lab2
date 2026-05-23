"""Path helpers for the MVTec AD dataset layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any


MVTEC_CLASSES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

IGNORE_FOLDERS = ["dtd", "mvtec_ad_evaluation"]

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
DEFAULT_DATASET_PATH = Path("./dataset")


def list_test_images(
    class_name: str,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
) -> list[dict[str, Any]]:
    """List test images and expected mask paths for one MVTec class."""

    if class_name not in MVTEC_CLASSES:
        valid_classes = ", ".join(MVTEC_CLASSES)
        raise ValueError(
            f"Unknown MVTec class '{class_name}'. Expected one of: {valid_classes}"
        )

    dataset_root = Path(dataset_path)
    class_root = dataset_root / class_name
    test_root = class_root / "test"
    ground_truth_root = class_root / "ground_truth"

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_root}")
    if not class_root.is_dir():
        raise FileNotFoundError(f"MVTec class folder is missing: {class_root}")
    if not test_root.is_dir():
        raise FileNotFoundError(f"Test folder is missing for '{class_name}': {test_root}")

    entries: list[dict[str, Any]] = []
    for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
        defect_type = defect_dir.name
        is_anomalous = defect_type != "good"

        if is_anomalous and not ground_truth_root.is_dir():
            raise FileNotFoundError(
                f"Ground-truth folder is missing for anomalous test images in "
                f"'{class_name}': {ground_truth_root}"
            )

        for image_path in sorted(defect_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            expected_mask_path = None
            if is_anomalous:
                expected_mask_path = _expected_mask_path(
                    ground_truth_root, defect_type, image_path
                )

            entries.append(
                {
                    "image_path": image_path,
                    "defect_type": defect_type,
                    "is_anomalous": is_anomalous,
                    "expected_mask_path": expected_mask_path,
                }
            )

    return entries


def _expected_mask_path(
    ground_truth_root: Path,
    defect_type: str,
    image_path: Path,
) -> Path:
    mask_dir = ground_truth_root / defect_type
    expected_png = mask_dir / f"{image_path.stem}_mask.png"
    if expected_png.exists():
        return expected_png

    matches = sorted(mask_dir.glob(f"{image_path.stem}_mask.*"))
    if matches:
        return matches[0]

    return expected_png
