"""Pixel-level anomaly segmentation metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


EPSILON = 1e-8


def compute_pixel_counts(
    prediction: Any,
    target: Any,
) -> dict[str, int]:
    prediction_values = _to_binary_array(prediction)
    target_values = _to_binary_array(target)

    if prediction_values.shape != target_values.shape:
        raise ValueError(
            f"Prediction and target sizes must match, got {prediction_values.shape} "
            f"and {target_values.shape}."
        )

    true_positive = int(np.logical_and(prediction_values, target_values).sum())
    false_positive = int(np.logical_and(prediction_values, ~target_values).sum())
    false_negative = int(np.logical_and(~prediction_values, target_values).sum())

    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
    }


def precision_recall_f1(
    true_positive: int,
    false_positive: int,
    false_negative: int,
    eps: float = EPSILON,
) -> dict[str, float]:
    precision = true_positive / (true_positive + false_positive + eps)
    recall = true_positive / (true_positive + false_negative + eps)

    if true_positive == 0 and false_positive == 0 and false_negative == 0:
        f1 = 1.0
    elif true_positive == 0 and false_positive > 0 and false_negative == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall + eps)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_pixel_metrics(
    prediction: Any,
    target: Any,
    eps: float = EPSILON,
) -> dict[str, float | int]:
    counts = compute_pixel_counts(prediction, target)
    rates = precision_recall_f1(
        counts["tp"],
        counts["fp"],
        counts["fn"],
        eps=eps,
    )
    return {**counts, **rates}


def build_metric_record(
    prediction: Any,
    target: Any,
    class_name: str,
    defect_type: str,
    image_path: str,
    eps: float = EPSILON,
) -> dict[str, float | int | str]:
    metrics = compute_pixel_metrics(prediction, target, eps=eps)
    return {
        "class_name": class_name,
        "defect_type": defect_type,
        "image_path": image_path,
        **metrics,
    }


def summarize_metric_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "mean_f1_per_image": 0.0,
            "mean_f1_per_defect_type": {},
            "mean_f1_per_class": {},
            "global_mean_f1": 0.0,
        }

    mean_f1_per_image = _mean([float(record["f1"]) for record in records])
    mean_f1_per_defect_type = _mean_f1_by_key(records, "defect_type")
    mean_f1_per_class = _mean_f1_by_key(records, "class_name")

    global_counts = {
        "tp": int(sum(record["tp"] for record in records)),
        "fp": int(sum(record["fp"] for record in records)),
        "fn": int(sum(record["fn"] for record in records)),
    }
    global_rates = precision_recall_f1(
        global_counts["tp"],
        global_counts["fp"],
        global_counts["fn"],
    )

    return {
        "tp": global_counts["tp"],
        "fp": global_counts["fp"],
        "fn": global_counts["fn"],
        "precision": global_rates["precision"],
        "recall": global_rates["recall"],
        "f1": global_rates["f1"],
        "mean_f1_per_image": mean_f1_per_image,
        "mean_f1_per_defect_type": mean_f1_per_defect_type,
        "mean_f1_per_class": mean_f1_per_class,
        "global_mean_f1": mean_f1_per_image,
    }


def _mean_f1_by_key(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    grouped_values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped_values[str(record[key])].append(float(record["f1"]))
    return {
        group_name: _mean(values)
        for group_name, values in sorted(grouped_values.items())
    }


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _to_flat_binary_list(values: Any) -> list[bool]:
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [bool(value) for value in _flatten(values)]


def _to_binary_array(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values).astype(bool)


def _flatten(values: Any) -> list[Any]:
    if isinstance(values, (list, tuple)):
        flattened: list[Any] = []
        for value in values:
            flattened.extend(_flatten(value))
        return flattened
    return [values]
