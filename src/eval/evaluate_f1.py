"""F1 evaluation for DRAEM anomaly segmentation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.mvtec_paths import MVTEC_CLASSES
from src.data.mvtec_test_dataset import MVTecTestDataset
from src.eval.thresholding import load_model_from_checkpoint
from src.eval.visualize_results import (
    mark_visual_saved,
    save_result_visualization,
    should_save_visual,
)
from src.utils.metrics import build_metric_record, summarize_metric_records


PER_IMAGE_COLUMNS = [
    "class_name",
    "defect_type",
    "image_path",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
]


def evaluate_classes(
    class_name: str,
    config: dict[str, Any],
    checkpoint: str | None = None,
    checkpoint_type: str = "best",
    save_visuals: bool = False,
    gaussian_sigma: float | None = None,
    min_component_area: int | None = None,
    closing_kernel_size: int | None = None,
    max_visuals_per_defect: int | None = None,
) -> dict[str, Any]:
    classes = MVTEC_CLASSES if class_name == "all" else [class_name]
    invalid_classes = [name for name in classes if name not in MVTEC_CLASSES]
    if invalid_classes:
        valid = ", ".join(MVTEC_CLASSES + ["all"])
        raise ValueError(f"Invalid class_name {invalid_classes}. Expected one of: {valid}")
    if class_name == "all" and checkpoint:
        raise ValueError("--checkpoint is only supported when evaluating a single class.")

    all_records: list[dict[str, Any]] = []
    class_summaries: list[dict[str, Any]] = []
    for current_class in classes:
        checkpoint_path = resolve_checkpoint_path(
            current_class,
            config,
            checkpoint=checkpoint,
            checkpoint_type=checkpoint_type,
        )
        result = evaluate_one_class(
            current_class,
            config=config,
            checkpoint_path=checkpoint_path,
            save_visuals=save_visuals,
            gaussian_sigma=gaussian_sigma,
            min_component_area=min_component_area,
            closing_kernel_size=closing_kernel_size,
            max_visuals_per_defect=max_visuals_per_defect,
        )
        all_records.extend(result["records"])
        class_summaries.append(result["summary"])

    global_summary = summarize_metric_records(all_records)
    if class_name == "all":
        save_global_summary(class_summaries, global_summary, config)
    return {
        "class_summaries": class_summaries,
        "global_summary": global_summary,
    }


@torch.no_grad()
def evaluate_one_class(
    class_name: str,
    config: dict[str, Any],
    checkpoint_path: str | Path,
    save_visuals: bool = False,
    gaussian_sigma: float | None = None,
    min_component_area: int | None = None,
    closing_kernel_size: int | None = None,
    max_visuals_per_defect: int | None = None,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(config.get("output_path", "./outputs"))
    threshold = load_threshold(class_name, config)
    sigma = float(
        gaussian_sigma
        if gaussian_sigma is not None
        else config.get("gaussian_sigma", 4.0)
    )
    min_area = int(
        min_component_area
        if min_component_area is not None
        else config.get("min_component_area", 16)
    )
    close_kernel = int(
        closing_kernel_size
        if closing_kernel_size is not None
        else config.get("closing_kernel_size", 5)
    )

    dataset = MVTecTestDataset(
        class_name=class_name,
        dataset_path=config.get("dataset_path", "./dataset"),
        image_size=int(config.get("image_size", 256)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(config.get("eval_num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    model = load_model_from_checkpoint(checkpoint_path, device)
    model.eval()

    records: list[dict[str, Any]] = []
    saved_visual_counts: dict[str, int] = {}
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"]
        image_paths = list(batch["image_path"])
        defect_types = list(batch["defect_type"])

        outputs = model(images)
        anomaly_maps = gaussian_smooth(outputs["anomaly_map"].detach().float(), sigma)

        for index, image_path_text in enumerate(image_paths):
            image_path = Path(image_path_text)
            defect_type = str(defect_types[index])
            image_name = image_path.stem

            anomaly_map = anomaly_maps[index, 0].cpu().numpy().astype(np.float32)
            gt_mask = (masks[index, 0].cpu().numpy() > 0).astype(np.uint8)
            predicted_mask = (anomaly_map >= threshold).astype(np.uint8)
            predicted_mask = postprocess_mask(
                predicted_mask,
                min_component_area=min_area,
                closing_kernel_size=close_kernel,
            )

            save_anomaly_map(
                anomaly_map,
                output_path / "anomaly_maps" / class_name / defect_type / f"{image_name}.npy",
            )
            save_predicted_mask(
                predicted_mask,
                output_path
                / "predicted_masks"
                / class_name
                / defect_type
                / f"{image_name}.png",
            )

            record = build_metric_record(
                prediction=predicted_mask,
                target=gt_mask,
                class_name=class_name,
                defect_type=defect_type,
                image_path=image_path_text,
            )
            records.append(record)

            if save_visuals and should_save_visual(
                saved_visual_counts,
                defect_type,
                max_visuals_per_defect,
            ):
                save_result_visualization(
                    image=images[index].detach().cpu(),
                    reconstruction=outputs["reconstruction"][index].detach().cpu(),
                    anomaly_map=anomaly_map,
                    predicted_mask=predicted_mask,
                    ground_truth_mask=gt_mask,
                    class_name=class_name,
                    defect_type=defect_type,
                    image_name=image_name,
                    f1=float(record["f1"]),
                    output_path=output_path
                    / "visualizations"
                    / "eval"
                    / class_name
                    / defect_type
                    / f"{image_name}.png",
                )
                mark_visual_saved(saved_visual_counts, defect_type)

    summary = summarize_metric_records(records)
    summary["class_name"] = class_name
    summary["threshold"] = threshold
    summary["num_images"] = len(records)
    save_per_image_metrics(records, output_path / "metrics" / class_name / "per_image_metrics.csv")
    save_summary(summary, output_path / "metrics" / class_name / "summary.json")
    print(
        f"[{class_name}] mean_f1={summary['mean_f1_per_image']:.5f} "
        f"global_f1={summary['f1']:.5f} images={len(records)}"
    )
    return {"records": records, "summary": summary}


def resolve_checkpoint_path(
    class_name: str,
    config: dict[str, Any],
    checkpoint: str | None,
    checkpoint_type: str,
) -> Path:
    if checkpoint:
        return Path(checkpoint)
    output_path = Path(config.get("output_path", "./outputs"))
    return output_path / "checkpoints" / class_name / f"{checkpoint_type}.pth"


def load_threshold(class_name: str, config: dict[str, Any]) -> float:
    threshold_path = (
        Path(config.get("output_path", "./outputs"))
        / "metrics"
        / class_name
        / "threshold.json"
    )
    if not threshold_path.is_file():
        raise FileNotFoundError(
            f"Validation-normal threshold file is missing: {threshold_path}. "
            "Compute it with src.eval.thresholding before evaluation. Test images "
            "and ground-truth masks are not used to derive the main threshold."
        )

    with threshold_path.open("r", encoding="utf-8") as file:
        threshold_data = json.load(file)
    if threshold_data.get("threshold_method") != "normal_validation_percentile":
        raise ValueError(
            f"Unsupported threshold method in {threshold_path}: "
            f"{threshold_data.get('threshold_method')}"
        )
    return float(threshold_data["threshold"])


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


def save_anomaly_map(anomaly_map: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, anomaly_map.astype(np.float32))


def save_predicted_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    image.save(path)


def save_per_image_metrics(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=PER_IMAGE_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in PER_IMAGE_COLUMNS})


def save_summary(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")


def save_global_summary(
    class_summaries: list[dict[str, Any]],
    global_summary: dict[str, Any],
    config: dict[str, Any],
) -> None:
    metrics_root = Path(config.get("output_path", "./outputs")) / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)

    csv_path = metrics_root / "global_summary.csv"
    columns = [
        "class_name",
        "num_images",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "mean_f1_per_image",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for summary in class_summaries:
            writer.writerow({key: summary.get(key, "") for key in columns})
        global_row = {key: global_summary.get(key, "") for key in columns}
        global_row["class_name"] = "global"
        global_row["num_images"] = sum(int(summary.get("num_images", 0)) for summary in class_summaries)
        writer.writerow(global_row)

    json_path = metrics_root / "global_summary.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "classes": class_summaries,
                "global": global_summary,
            },
            file,
            indent=2,
        )
        file.write("\n")
