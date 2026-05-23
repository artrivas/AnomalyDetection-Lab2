"""Threshold estimation from normal validation images only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.mvtec_paths import MVTEC_CLASSES
from src.data.mvtec_train_dataset import MVTecTrainDataset
from src.models import DRAEM


@torch.no_grad()
def compute_normal_validation_threshold(
    class_name: str,
    checkpoint_path: str | Path,
    config: dict[str, Any],
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Compute anomaly threshold from normal validation images.

    This function intentionally uses only `dataset/<class>/train/good` through
    `MVTecTrainDataset(split="val")`. It must not use test images or
    ground-truth masks for the main threshold.
    """

    if class_name not in MVTEC_CLASSES:
        valid_classes = ", ".join(MVTEC_CLASSES)
        raise ValueError(
            f"Unknown MVTec class '{class_name}'. Expected one of: {valid_classes}"
        )

    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_model_from_checkpoint(checkpoint_path, resolved_device)
    model.eval()

    val_dataset = MVTecTrainDataset(
        class_name=class_name,
        dataset_path=config.get("dataset_path", "./dataset"),
        dtd_path=config.get("dtd_path", "./dataset/dtd/images"),
        image_size=int(config.get("image_size", 256)),
        split="val",
        train_val_split=float(config.get("train_val_split", 0.8)),
        seed=int(config.get("seed", 42)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 4)),
        pin_memory=resolved_device.type == "cuda",
    )

    anomaly_values: list[torch.Tensor] = []
    for batch in val_loader:
        images = batch["image"].to(resolved_device, non_blocking=True)
        outputs = model(images)
        anomaly_values.append(outputs["anomaly_map"].detach().float().cpu().flatten())

    if not anomaly_values:
        raise RuntimeError(
            f"No normal validation images available for threshold computation: {class_name}"
        )

    all_values = torch.cat(anomaly_values, dim=0)
    threshold_percentile = float(config.get("threshold_percentile", 99.5))
    threshold = torch.quantile(all_values, threshold_percentile / 100.0).item()

    result = {
        "class_name": class_name,
        "threshold_method": "normal_validation_percentile",
        "threshold_percentile": threshold_percentile,
        "threshold": float(threshold),
        "num_validation_images": len(val_dataset),
    }
    save_threshold_json(result, config)
    return result


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> DRAEM:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model_state" not in checkpoint:
        raise KeyError(f"Checkpoint is missing 'model_state': {path}")

    model = DRAEM().to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model


def save_threshold_json(result: dict[str, Any], config: dict[str, Any]) -> Path:
    output_path = Path(config.get("output_path", "./outputs"))
    class_name = str(result["class_name"])
    threshold_path = output_path / "metrics" / class_name / "threshold.json"
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    with threshold_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    return threshold_path


def oracle_analysis_only_threshold(*_: Any, **__: Any) -> None:
    """Placeholder for explicitly labeled future oracle threshold analysis."""

    raise NotImplementedError(
        "Oracle thresholding is not implemented. If added later, it must be "
        "reported only as oracle_analysis_only and kept separate from the main "
        "normal-validation threshold."
    )
