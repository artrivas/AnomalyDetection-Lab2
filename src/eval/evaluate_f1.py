"""F1 evaluation and debugging utilities for DRAEM anomaly segmentation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.mvtec_paths import MVTEC_CLASSES
from src.data.mvtec_test_dataset import MVTecTestDataset
from src.data.mvtec_train_dataset import MVTecTrainDataset
from src.eval.thresholding import load_model_from_checkpoint
from src.eval.visualize_results import (
    mark_visual_saved,
    save_debug_visualization,
    save_result_visualization,
    should_save_visual,
)
from src.utils.metrics import build_metric_record, precision_recall_f1


THRESHOLD_SWEEP_PERCENTILES = [95, 97, 98, 99, 99.5]
ORACLE_THRESHOLD_COUNT = 101

PER_IMAGE_COLUMNS = [
    "class_name",
    "defect_type",
    "image_path",
    "is_anomalous",
    "threshold_percentile",
    "threshold_value",
    "f1",
    "precision",
    "recall",
    "tp",
    "fp",
    "fn",
    "gt_area",
    "predicted_area",
    "raw_anomaly_min",
    "raw_anomaly_max",
    "raw_anomaly_mean",
    "raw_anomaly_p95",
    "raw_anomaly_p99",
    "raw_anomaly_p995",
]

SWEEP_COLUMNS = [
    "class_name",
    "threshold_percentile",
    "threshold_value",
    "anomaly_only_mean_f1",
    "anomaly_only_global_f1",
    "precision",
    "recall",
    "normal_false_positive_pixel_rate",
    "normal_images_with_any_prediction",
]

POSTPROCESS_ABLATION_COLUMNS = [
    "class_name",
    "mode",
    "gaussian_sigma",
    "min_component_area",
    "use_closing",
    "anomaly_only_mean_f1",
    "anomaly_only_global_f1",
    "precision",
    "recall",
]


@dataclass(frozen=True)
class EvaluationSample:
    class_name: str
    defect_type: str
    image_path: str
    image_name: str
    is_anomalous: bool
    gt_mask: np.ndarray
    raw_anomaly_map: np.ndarray
    smoothed_anomaly_map: np.ndarray
    image: torch.Tensor | None = None
    reconstruction: torch.Tensor | None = None


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
    threshold_sweep: bool = False,
    oracle_threshold_analysis: bool = False,
    postprocess_ablation: bool = False,
    debug_visuals: bool = False,
    max_debug_visuals: int = 20,
    check_masks: bool = False,
) -> dict[str, Any]:
    classes = MVTEC_CLASSES if class_name == "all" else [class_name]
    invalid_classes = [name for name in classes if name not in MVTEC_CLASSES]
    if invalid_classes:
        valid = ", ".join(MVTEC_CLASSES + ["all"])
        raise ValueError(f"Invalid class_name {invalid_classes}. Expected one of: {valid}")
    if class_name == "all" and checkpoint:
        raise ValueError("--checkpoint is only supported when evaluating a single class.")

    output_path = Path(config.get("output_path", "./outputs"))
    all_records: list[dict[str, Any]] = []
    class_summaries: list[dict[str, Any]] = []
    all_sweep_rows: list[dict[str, Any]] = []

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
            threshold_sweep=threshold_sweep,
            oracle_threshold_analysis=oracle_threshold_analysis,
            postprocess_ablation=postprocess_ablation,
            debug_visuals=debug_visuals,
            max_debug_visuals=max_debug_visuals,
            check_masks=check_masks,
        )
        all_records.extend(result["records"])
        class_summaries.append(result["summary"])
        all_sweep_rows.extend(result.get("threshold_sweep_rows", []))

    global_summary = summarize_records(all_records)
    if class_name == "all":
        save_global_summary(class_summaries, global_summary, config)
        if threshold_sweep:
            save_csv(all_sweep_rows, output_path / "metrics" / "threshold_sweep_global.csv", SWEEP_COLUMNS)
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
    threshold_sweep: bool = False,
    oracle_threshold_analysis: bool = False,
    postprocess_ablation: bool = False,
    debug_visuals: bool = False,
    max_debug_visuals: int = 20,
    check_masks: bool = False,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(config.get("output_path", "./outputs"))
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

    model = load_model_from_checkpoint(checkpoint_path, device)
    model.eval()
    threshold_info = load_or_compute_threshold_info(
        class_name=class_name,
        config=config,
        model=model,
        device=device,
        gaussian_sigma=sigma,
    )
    samples = collect_samples(
        class_name=class_name,
        config=config,
        model=model,
        device=device,
        gaussian_sigma=sigma,
        include_visual_data=debug_visuals or save_visuals,
    )

    records = evaluate_samples(
        samples,
        threshold_value=float(threshold_info["threshold"]),
        threshold_percentile=float(threshold_info["threshold_percentile"]),
        min_component_area=min_area,
        closing_kernel_size=close_kernel,
        use_closing=True,
        check_masks=check_masks,
    )

    save_standard_outputs(
        samples=samples,
        records=records,
        output_path=output_path,
        class_name=class_name,
    )
    if save_visuals:
        save_limited_eval_visuals(
            samples=samples,
            records=records,
            output_path=output_path,
            max_visuals_per_defect=max_visuals_per_defect,
        )

    summary = summarize_records(records)
    summary["class_name"] = class_name
    summary["threshold"] = float(threshold_info["threshold"])
    summary["threshold_percentile"] = float(threshold_info["threshold_percentile"])
    summary["num_images"] = len(records)
    save_per_image_metrics(records, output_path / "metrics" / class_name / "per_image_metrics.csv")
    save_summary(summary, output_path / "metrics" / class_name / "summary.json")

    threshold_sweep_rows: list[dict[str, Any]] = []
    if threshold_sweep:
        threshold_sweep_rows = run_threshold_sweep(
            class_name=class_name,
            config=config,
            model=model,
            device=device,
            samples=samples,
            gaussian_sigma=sigma,
            min_component_area=min_area,
            closing_kernel_size=close_kernel,
            check_masks=check_masks,
        )

    if oracle_threshold_analysis:
        run_oracle_threshold_analysis(
            class_name=class_name,
            samples=samples,
            output_path=output_path,
            min_component_area=min_area,
            closing_kernel_size=close_kernel,
            check_masks=check_masks,
        )

    if postprocess_ablation:
        run_postprocess_ablation(
            class_name=class_name,
            samples=samples,
            output_path=output_path,
            threshold_value=float(threshold_info["threshold"]),
            threshold_percentile=float(threshold_info["threshold_percentile"]),
            gaussian_sigma=sigma,
            check_masks=check_masks,
        )

    if debug_visuals:
        save_debug_visuals(
            samples=samples,
            records=records,
            output_path=output_path,
            max_debug_visuals=max_debug_visuals,
        )

    print(
        f"[{class_name}] anomaly_only_mean_f1={summary['anomaly_only_mean_f1']:.5f} "
        f"anomaly_only_global_f1={summary['anomaly_only_global_f1']:.5f} "
        f"all_images_mean_f1={summary['all_images_mean_f1']:.5f} images={len(records)}"
    )
    return {
        "records": records,
        "summary": summary,
        "threshold_sweep_rows": threshold_sweep_rows,
    }


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


def load_threshold_info(class_name: str, config: dict[str, Any]) -> dict[str, float]:
    threshold_path = (
        Path(config.get("output_path", "./outputs"))
        / "metrics"
        / class_name
        / "threshold.json"
    )
    if not threshold_path.is_file():
        raise FileNotFoundError(
            f"Validation-normal threshold file is missing: {threshold_path}. "
            "Thresholds for official evaluation must come from normal validation images."
        )

    with threshold_path.open("r", encoding="utf-8") as file:
        threshold_data = json.load(file)
    if threshold_data.get("threshold_method") != "normal_validation_percentile":
        raise ValueError(
            f"Unsupported threshold method in {threshold_path}: "
            f"{threshold_data.get('threshold_method')}"
        )
    return {
        "threshold": float(threshold_data["threshold"]),
        "threshold_percentile": float(threshold_data.get("threshold_percentile", config.get("threshold_percentile", 99.5))),
    }


def load_or_compute_threshold_info(
    class_name: str,
    config: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    gaussian_sigma: float,
) -> dict[str, float]:
    try:
        return load_threshold_info(class_name, config)
    except FileNotFoundError:
        percentile = float(config.get("threshold_percentile", 99.5))
        thresholds = compute_validation_thresholds(
            class_name=class_name,
            config=config,
            model=model,
            device=device,
            percentiles=[percentile],
            gaussian_sigma=gaussian_sigma,
        )
        return {
            "threshold": thresholds[percentile],
            "threshold_percentile": percentile,
        }


@torch.no_grad()
def collect_samples(
    class_name: str,
    config: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    gaussian_sigma: float,
    include_visual_data: bool,
) -> list[EvaluationSample]:
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

    samples: list[EvaluationSample] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"]
        image_paths = list(batch["image_path"])
        defect_types = list(batch["defect_type"])
        is_anomalous_values = [bool(value) for value in batch["is_anomalous"]]

        outputs = model(images)
        raw_maps = outputs["anomaly_map"].detach().float()
        smoothed_maps = gaussian_smooth(raw_maps, gaussian_sigma)

        for index, image_path_text in enumerate(image_paths):
            image_path = Path(image_path_text)
            gt_mask = (masks[index, 0].cpu().numpy() > 0).astype(np.uint8)
            raw_map = raw_maps[index, 0].cpu().numpy().astype(np.float32)
            smoothed_map = smoothed_maps[index, 0].cpu().numpy().astype(np.float32)
            samples.append(
                EvaluationSample(
                    class_name=class_name,
                    defect_type=str(defect_types[index]),
                    image_path=image_path_text,
                    image_name=image_path.stem,
                    is_anomalous=is_anomalous_values[index],
                    gt_mask=gt_mask,
                    raw_anomaly_map=raw_map,
                    smoothed_anomaly_map=smoothed_map,
                    image=images[index].detach().cpu() if include_visual_data else None,
                    reconstruction=outputs["reconstruction"][index].detach().cpu()
                    if include_visual_data
                    else None,
                )
            )
    return samples


def evaluate_samples(
    samples: list[EvaluationSample],
    threshold_value: float,
    threshold_percentile: float | str,
    min_component_area: int,
    closing_kernel_size: int,
    use_closing: bool,
    check_masks: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample in samples:
        predicted_mask = (sample.smoothed_anomaly_map >= threshold_value).astype(np.uint8)
        predicted_mask = postprocess_mask(
            predicted_mask,
            min_component_area=min_component_area,
            closing_kernel_size=closing_kernel_size if use_closing else 0,
        )
        if check_masks:
            validate_masks(predicted_mask, sample.gt_mask, sample)

        record = build_metric_record(
            prediction=predicted_mask,
            target=sample.gt_mask,
            class_name=sample.class_name,
            defect_type=sample.defect_type,
            image_path=sample.image_path,
        )
        record.update(
            {
                "is_anomalous": sample.is_anomalous,
                "threshold_percentile": threshold_percentile,
                "threshold_value": float(threshold_value),
                "gt_area": int(sample.gt_mask.sum()),
                "predicted_area": int(predicted_mask.sum()),
                "_mask_shape": tuple(int(value) for value in sample.gt_mask.shape),
                "_predicted_mask": predicted_mask,
                **anomaly_map_stats(sample.raw_anomaly_map),
            }
        )
        records.append(record)
    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    anomaly_records = [record for record in records if bool(record["is_anomalous"])]
    normal_records = [record for record in records if not bool(record["is_anomalous"])]

    all_summary = summarize_counts(records)
    anomaly_summary = summarize_counts(anomaly_records)
    per_defect_type = mean_f1_by_key(anomaly_records, "defect_type")
    per_class = mean_f1_by_key(anomaly_records, "class_name")

    if normal_records:
        pixels_per_image = [
            int(np.prod(record.get("_mask_shape", (0, 0)))) for record in normal_records
        ]
        pixel_total = sum(pixels_per_image)
    else:
        pixel_total = 0
    normal_predicted_pixels = int(sum(record["predicted_area"] for record in normal_records))
    normal_count = len(normal_records)

    return {
        "anomaly_only_mean_f1": mean_f1(anomaly_records),
        "anomaly_only_global_f1": anomaly_summary["f1"],
        "anomaly_only_precision": anomaly_summary["precision"],
        "anomaly_only_recall": anomaly_summary["recall"],
        "per_defect_type_mean_f1": per_defect_type,
        "per_class_anomaly_only_mean_f1": per_class,
        "normal_images_count": normal_count,
        "normal_false_positive_pixel_rate": float(normal_predicted_pixels / pixel_total) if pixel_total else 0.0,
        "normal_images_with_any_prediction": int(
            sum(1 for record in normal_records if int(record["predicted_area"]) > 0)
        ),
        "normal_mean_predicted_anomaly_area": float(normal_predicted_pixels / normal_count) if normal_count else 0.0,
        "all_images_tp": all_summary["tp"],
        "all_images_fp": all_summary["fp"],
        "all_images_fn": all_summary["fn"],
        "all_images_precision": all_summary["precision"],
        "all_images_recall": all_summary["recall"],
        "all_images_global_f1": all_summary["f1"],
        "all_images_mean_f1": mean_f1(records),
        "warning": (
            "Main lab metric should be anomaly_only_mean_f1 / anomaly_only_global_f1. "
            "all_images_mean_f1 may be inflated by normal images."
        ),
    }


def summarize_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        "tp": int(sum(record["tp"] for record in records)),
        "fp": int(sum(record["fp"] for record in records)),
        "fn": int(sum(record["fn"] for record in records)),
    }
    rates = precision_recall_f1(counts["tp"], counts["fp"], counts["fn"])
    return {**counts, **rates}


@torch.no_grad()
def compute_validation_thresholds(
    class_name: str,
    config: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    percentiles: list[float],
    gaussian_sigma: float,
) -> dict[float, float]:
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
        num_workers=int(config.get("eval_num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    values: list[torch.Tensor] = []
    for batch in val_loader:
        images = batch["image"].to(device, non_blocking=True)
        outputs = model(images)
        anomaly_maps = gaussian_smooth(outputs["anomaly_map"].detach().float(), gaussian_sigma)
        values.append(anomaly_maps.cpu().flatten())
    if not values:
        raise RuntimeError(f"No normal validation images available for {class_name}")
    all_values = torch.cat(values, dim=0)
    return {
        percentile: float(torch.quantile(all_values, percentile / 100.0).item())
        for percentile in percentiles
    }


def run_threshold_sweep(
    class_name: str,
    config: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    samples: list[EvaluationSample],
    gaussian_sigma: float,
    min_component_area: int,
    closing_kernel_size: int,
    check_masks: bool,
) -> list[dict[str, Any]]:
    output_path = Path(config.get("output_path", "./outputs"))
    thresholds = compute_validation_thresholds(
        class_name=class_name,
        config=config,
        model=model,
        device=device,
        percentiles=THRESHOLD_SWEEP_PERCENTILES,
        gaussian_sigma=gaussian_sigma,
    )
    rows: list[dict[str, Any]] = []
    for percentile in THRESHOLD_SWEEP_PERCENTILES:
        records = evaluate_samples(
            samples,
            threshold_value=thresholds[percentile],
            threshold_percentile=percentile,
            min_component_area=min_component_area,
            closing_kernel_size=closing_kernel_size,
            use_closing=True,
            check_masks=check_masks,
        )
        summary = summarize_records(records)
        rows.append(
            {
                "class_name": class_name,
                "threshold_percentile": percentile,
                "threshold_value": thresholds[percentile],
                "anomaly_only_mean_f1": summary["anomaly_only_mean_f1"],
                "anomaly_only_global_f1": summary["anomaly_only_global_f1"],
                "precision": summary["anomaly_only_precision"],
                "recall": summary["anomaly_only_recall"],
                "normal_false_positive_pixel_rate": summary["normal_false_positive_pixel_rate"],
                "normal_images_with_any_prediction": summary["normal_images_with_any_prediction"],
            }
        )

    save_csv(rows, output_path / "metrics" / class_name / "threshold_sweep.csv", SWEEP_COLUMNS)
    selected = max(rows, key=lambda row: float(row["anomaly_only_mean_f1"]))
    save_selected_threshold(class_name, selected, output_path)
    return rows


def save_selected_threshold(class_name: str, selected: dict[str, Any], output_path: Path) -> None:
    path = output_path / "metrics" / class_name / "selected_threshold.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "class_name": class_name,
        "selection_method": "validation_percentile_sweep_selected",
        "candidate_percentiles": THRESHOLD_SWEEP_PERCENTILES,
        "selected_percentile": float(selected["threshold_percentile"]),
        "selected_threshold": float(selected["threshold_value"]),
        "note": (
            "Threshold values are computed only from normal validation anomaly scores. "
            "Selection is used as a lab hyperparameter sweep."
        ),
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def run_oracle_threshold_analysis(
    class_name: str,
    samples: list[EvaluationSample],
    output_path: Path,
    min_component_area: int,
    closing_kernel_size: int,
    check_masks: bool,
) -> None:
    anomaly_values = np.concatenate(
        [sample.smoothed_anomaly_map.reshape(-1) for sample in samples if sample.is_anomalous]
    )
    thresholds = np.quantile(anomaly_values, np.linspace(0.0, 1.0, ORACLE_THRESHOLD_COUNT))
    rows: list[dict[str, Any]] = []
    for threshold in sorted(set(float(value) for value in thresholds)):
        records = evaluate_samples(
            samples,
            threshold_value=threshold,
            threshold_percentile="oracle_analysis_only",
            min_component_area=0,
            closing_kernel_size=0,
            use_closing=False,
            check_masks=check_masks,
        )
        summary = summarize_records(records)
        rows.append(
            {
                "class_name": class_name,
                "oracle_analysis_only": True,
                "threshold_value": threshold,
                "anomaly_only_mean_f1": summary["anomaly_only_mean_f1"],
                "anomaly_only_global_f1": summary["anomaly_only_global_f1"],
                "precision": summary["anomaly_only_precision"],
                "recall": summary["anomaly_only_recall"],
                "normal_false_positive_pixel_rate": summary["normal_false_positive_pixel_rate"],
                "normal_images_with_any_prediction": summary["normal_images_with_any_prediction"],
            }
        )
    columns = [
        "class_name",
        "oracle_analysis_only",
        "threshold_value",
        "anomaly_only_mean_f1",
        "anomaly_only_global_f1",
        "precision",
        "recall",
        "normal_false_positive_pixel_rate",
        "normal_images_with_any_prediction",
    ]
    save_csv(rows, output_path / "metrics" / class_name / "oracle_threshold_analysis.csv", columns)


def run_postprocess_ablation(
    class_name: str,
    samples: list[EvaluationSample],
    output_path: Path,
    threshold_value: float,
    threshold_percentile: float,
    gaussian_sigma: float,
    check_masks: bool,
) -> None:
    modes = [
        ("no_postprocessing", 0.0, 0, False),
        ("gaussian_only", gaussian_sigma, 0, False),
        ("gaussian_remove_small_5", gaussian_sigma, 5, False),
        ("gaussian_remove_small_10", gaussian_sigma, 10, False),
        ("gaussian_remove_small_20", gaussian_sigma, 20, False),
        ("gaussian_closing", gaussian_sigma, 0, True),
        ("gaussian_remove_small_5_closing", gaussian_sigma, 5, True),
        ("gaussian_remove_small_10_closing", gaussian_sigma, 10, True),
        ("gaussian_remove_small_20_closing", gaussian_sigma, 20, True),
    ]
    rows: list[dict[str, Any]] = []
    for mode, mode_sigma, min_area, use_closing in modes:
        mode_samples = [
            EvaluationSample(
                **{
                    **sample.__dict__,
                    "smoothed_anomaly_map": sample.raw_anomaly_map
                    if mode_sigma <= 0
                    else sample.smoothed_anomaly_map,
                }
            )
            for sample in samples
        ]
        records = evaluate_samples(
            mode_samples,
            threshold_value=threshold_value,
            threshold_percentile=threshold_percentile,
            min_component_area=min_area,
            closing_kernel_size=5,
            use_closing=use_closing,
            check_masks=check_masks,
        )
        summary = summarize_records(records)
        rows.append(
            {
                "class_name": class_name,
                "mode": mode,
                "gaussian_sigma": mode_sigma,
                "min_component_area": min_area,
                "use_closing": use_closing,
                "anomaly_only_mean_f1": summary["anomaly_only_mean_f1"],
                "anomaly_only_global_f1": summary["anomaly_only_global_f1"],
                "precision": summary["anomaly_only_precision"],
                "recall": summary["anomaly_only_recall"],
            }
        )
    save_csv(rows, output_path / "metrics" / class_name / "postprocess_ablation.csv", POSTPROCESS_ABLATION_COLUMNS)


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


def validate_masks(predicted_mask: np.ndarray, gt_mask: np.ndarray, sample: EvaluationSample) -> None:
    if predicted_mask.shape != gt_mask.shape:
        raise ValueError(
            f"Mask size mismatch for {sample.image_path}: prediction shape "
            f"{predicted_mask.shape}, ground truth shape {gt_mask.shape}."
        )
    for name, mask in (("prediction", predicted_mask), ("ground truth", gt_mask)):
        unique_values = set(np.unique(mask).tolist())
        if not unique_values.issubset({0, 1, False, True}):
            raise ValueError(
                f"Non-binary {name} mask for {sample.image_path}: values={sorted(unique_values)}"
            )
    gt_area = int(gt_mask.sum())
    total_pixels = int(gt_mask.size)
    if sample.is_anomalous and gt_area == 0:
        raise ValueError(f"Anomalous image has empty ground-truth mask: {sample.image_path}")
    if not sample.is_anomalous and gt_area != 0:
        raise ValueError(f"Good image must use a zero ground-truth mask: {sample.image_path}")
    if gt_area >= int(total_pixels * 0.95):
        raise ValueError(
            f"Ground-truth mask covers >=95% of pixels for {sample.image_path}; "
            "this is suspicious for an inverted mask."
        )


def anomaly_map_stats(anomaly_map: np.ndarray) -> dict[str, float]:
    return {
        "raw_anomaly_min": float(np.min(anomaly_map)),
        "raw_anomaly_max": float(np.max(anomaly_map)),
        "raw_anomaly_mean": float(np.mean(anomaly_map)),
        "raw_anomaly_p95": float(np.percentile(anomaly_map, 95)),
        "raw_anomaly_p99": float(np.percentile(anomaly_map, 99)),
        "raw_anomaly_p995": float(np.percentile(anomaly_map, 99.5)),
    }


def save_standard_outputs(
    samples: list[EvaluationSample],
    records: list[dict[str, Any]],
    output_path: Path,
    class_name: str,
) -> None:
    for sample, record in zip(samples, records):
        predicted_mask = record["_predicted_mask"]
        save_anomaly_map(
            sample.smoothed_anomaly_map,
            output_path / "anomaly_maps" / class_name / sample.defect_type / f"{sample.image_name}.npy",
        )
        save_predicted_mask(
            predicted_mask,
            output_path
            / "predicted_masks"
            / class_name
            / sample.defect_type
            / f"{sample.image_name}.png",
        )


def save_limited_eval_visuals(
    samples: list[EvaluationSample],
    records: list[dict[str, Any]],
    output_path: Path,
    max_visuals_per_defect: int | None,
) -> None:
    saved_visual_counts: dict[str, int] = {}
    for sample, record in zip(samples, records):
        if sample.image is None or sample.reconstruction is None:
            continue
        if not should_save_visual(saved_visual_counts, sample.defect_type, max_visuals_per_defect):
            continue
        save_result_visualization(
            image=sample.image,
            reconstruction=sample.reconstruction,
            anomaly_map=sample.smoothed_anomaly_map,
            predicted_mask=record["_predicted_mask"],
            ground_truth_mask=sample.gt_mask,
            class_name=sample.class_name,
            defect_type=sample.defect_type,
            image_name=sample.image_name,
            f1=float(record["f1"]),
            output_path=output_path
            / "visualizations"
            / "eval"
            / sample.class_name
            / sample.defect_type
            / f"{sample.image_name}.png",
        )
        mark_visual_saved(saved_visual_counts, sample.defect_type)


def save_debug_visuals(
    samples: list[EvaluationSample],
    records: list[dict[str, Any]],
    output_path: Path,
    max_debug_visuals: int,
) -> None:
    anomalous_pairs = [
        (sample, record)
        for sample, record in zip(samples, records)
        if sample.is_anomalous and sample.image is not None and sample.reconstruction is not None
    ]
    half = max(1, max_debug_visuals // 2)
    worst = sorted(anomalous_pairs, key=lambda pair: float(pair[1]["f1"]))[:half]
    best = sorted(anomalous_pairs, key=lambda pair: float(pair[1]["f1"]), reverse=True)[: max_debug_visuals - len(worst)]
    for rank, (sample, record) in enumerate(worst + best):
        save_debug_visualization(
            image=sample.image,
            reconstruction=sample.reconstruction,
            raw_anomaly_map=sample.raw_anomaly_map,
            smoothed_anomaly_map=sample.smoothed_anomaly_map,
            predicted_mask=record["_predicted_mask"],
            ground_truth_mask=sample.gt_mask,
            class_name=sample.class_name,
            defect_type=sample.defect_type,
            image_name=sample.image_name,
            threshold_percentile=record["threshold_percentile"],
            threshold_value=float(record["threshold_value"]),
            f1=float(record["f1"]),
            precision=float(record["precision"]),
            recall=float(record["recall"]),
            predicted_area=int(record["predicted_area"]),
            gt_area=int(record["gt_area"]),
            output_path=output_path
            / "visualizations"
            / "debug"
            / sample.class_name
            / f"{rank:03d}_{sample.defect_type}_{sample.image_name}.png",
        )


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
            writer.writerow({key: record.get(key, "") for key in PER_IMAGE_COLUMNS})


def save_summary(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")


def save_csv(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


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
        "anomaly_only_mean_f1",
        "anomaly_only_global_f1",
        "anomaly_only_precision",
        "anomaly_only_recall",
        "all_images_mean_f1",
        "all_images_global_f1",
        "normal_false_positive_pixel_rate",
        "normal_images_with_any_prediction",
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
        json.dump({"classes": class_summaries, "global": global_summary}, file, indent=2)
        file.write("\n")


def mean_f1(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return float(sum(float(record["f1"]) for record in records) / len(records))


def mean_f1_by_key(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record[key]), []).append(record)
    return {group: mean_f1(values) for group, values in sorted(grouped.items())}
