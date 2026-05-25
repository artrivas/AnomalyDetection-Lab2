"""Single-image DRAEM inference and visualization."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from src.eval.evaluate_f1 import compute_validation_thresholds, resolve_evaluation_config
from src.eval.postprocessing import gaussian_smooth, postprocess_mask
from src.models import DRAEM
from src.utils.metrics import compute_pixel_counts, precision_recall_f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DRAEM inference and visualization on one image."
    )
    parser.add_argument("--image_path", required=True, help="Path to the input image.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained .pth checkpoint.")
    parser.add_argument(
        "--config",
        default="configs/draem_mvtec.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--class_name",
        default=None,
        help="Optional MVTec class name for threshold loading/computation.",
    )
    parser.add_argument("--output_path", default=None, help="Direct path for visualization PNG.")
    parser.add_argument("--output_dir", default=None, help="Directory for visualization and arrays.")
    parser.add_argument(
        "--threshold_value",
        type=float,
        default=None,
        help="Explicit anomaly threshold. Highest priority if provided.",
    )
    parser.add_argument(
        "--threshold_percentile",
        type=float,
        default=None,
        help="Normal-validation percentile for computed threshold.",
    )
    parser.add_argument(
        "--use_threshold_file",
        action="store_true",
        help="Load threshold from outputs/metrics/<class_name>/threshold*.json.",
    )
    parser.add_argument(
        "--checkpoint_type",
        default=None,
        help="Checkpoint type for threshold file lookup. Defaults to checkpoint stem if best/last.",
    )
    parser.add_argument("--gaussian_sigma", type=float, default=None, help="Override smoothing sigma.")
    parser.add_argument(
        "--min_component_area",
        type=int,
        default=None,
        help="Override minimum connected-component area.",
    )
    parser.add_argument(
        "--no_closing",
        action="store_true",
        help="Disable morphological closing for predicted mask.",
    )
    parser.add_argument(
        "--closing_kernel_size",
        type=int,
        default=None,
        help="Override morphological closing kernel size.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda, or a torch device string.",
    )
    parser.add_argument("--save_arrays", action="store_true", help="Save .npy/PNG outputs.")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively.")
    parser.add_argument("--title", default=None, help="Optional custom figure title.")
    parser.add_argument("--gt_mask_path", default=None, help="Optional ground-truth mask path.")
    return parser.parse_args()


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return config


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_model_and_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[DRAEM, dict[str, Any]]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = DRAEM().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata


def infer_checkpoint_type(checkpoint_path: str | Path, checkpoint_type: str | None) -> str:
    if checkpoint_type:
        return checkpoint_type
    stem = Path(checkpoint_path).stem.lower()
    if stem in {"best", "last"}:
        return stem
    return "best"


def load_image_tensor(image_path: str | Path, image_size: int, device: torch.device) -> tuple[Image.Image, torch.Tensor]:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    return image, tensor


def load_gt_mask(mask_path: str | Path, image_size: int) -> np.ndarray:
    path = Path(mask_path)
    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth mask does not exist: {path}")
    mask = Image.open(path).convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)
    return (np.asarray(mask) > 0).astype(np.uint8)


def resolve_threshold(
    args: argparse.Namespace,
    config: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    class_name: str | None,
    checkpoint_type: str,
) -> tuple[float, float | None, str]:
    evaluation_config = resolve_evaluation_config(
        config,
        threshold_percentile=args.threshold_percentile,
        gaussian_sigma=args.gaussian_sigma,
        min_component_area=args.min_component_area,
        no_closing=args.no_closing,
        closing_kernel_size=args.closing_kernel_size,
    )
    threshold_percentile = float(evaluation_config["threshold_percentile"])

    if args.threshold_value is not None:
        return float(args.threshold_value), threshold_percentile, "explicit_threshold_value"

    if args.use_threshold_file and class_name:
        threshold_info, source = load_threshold_from_file(
            class_name=class_name,
            config=config,
            checkpoint_type=checkpoint_type,
            checkpoint_path=args.checkpoint,
        )
        return (
            float(threshold_info["threshold"]),
            float(threshold_info.get("threshold_percentile", threshold_percentile)),
            source,
        )

    if class_name:
        thresholds = compute_validation_thresholds(
            class_name=class_name,
            config=config,
            model=model,
            device=device,
            percentiles=[threshold_percentile],
            gaussian_sigma=float(evaluation_config["gaussian_sigma"]),
        )
        return thresholds[threshold_percentile], threshold_percentile, "normal_validation_percentile"

    warnings.warn(
        "No threshold source provided. Using 0.5. For proper MVTec evaluation, "
        "provide --class_name or --threshold_value.",
        stacklevel=2,
    )
    return 0.5, threshold_percentile, "default_0.5"


def load_threshold_from_file(
    class_name: str,
    config: dict[str, Any],
    checkpoint_type: str,
    checkpoint_path: str | Path,
) -> tuple[dict[str, Any], str]:
    metrics_dir = Path(config.get("output_path", "./outputs")) / "metrics" / class_name
    candidates = [
        metrics_dir / f"threshold_{checkpoint_type}.json",
        metrics_dir / "threshold.json",
    ]
    threshold_path = next((path for path in candidates if path.is_file()), None)
    if threshold_path is None:
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"No threshold file found. Searched: {searched}")

    with threshold_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    threshold = data.get("threshold", data.get("threshold_value", data.get("selected_threshold")))
    if threshold is None:
        raise KeyError(f"Threshold file is missing threshold value: {threshold_path}")
    data["threshold"] = float(threshold)

    current_checkpoint = str(Path(checkpoint_path))
    file_checkpoint = data.get("checkpoint") or data.get("checkpoint_path")
    file_checkpoint_type = data.get("checkpoint_type")
    if file_checkpoint and Path(str(file_checkpoint)) != Path(current_checkpoint):
        warnings.warn(
            f"Threshold file checkpoint metadata does not match current checkpoint: "
            f"{file_checkpoint} != {current_checkpoint}",
            stacklevel=2,
        )
    if file_checkpoint_type and str(file_checkpoint_type) != checkpoint_type:
        warnings.warn(
            f"Threshold file checkpoint_type metadata does not match: "
            f"{file_checkpoint_type} != {checkpoint_type}",
            stacklevel=2,
        )
    return data, str(threshold_path)


def anomaly_stats(anomaly_map: np.ndarray) -> dict[str, float]:
    return {
        "anomaly_min": float(np.min(anomaly_map)),
        "anomaly_max": float(np.max(anomaly_map)),
        "anomaly_mean": float(np.mean(anomaly_map)),
        "anomaly_p95": float(np.percentile(anomaly_map, 95)),
        "anomaly_p99": float(np.percentile(anomaly_map, 99)),
        "anomaly_p995": float(np.percentile(anomaly_map, 99.5)),
    }


def make_overlay(original: np.ndarray, predicted_mask: np.ndarray) -> np.ndarray:
    overlay = original.copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 1.0
    mask = predicted_mask.astype(bool)
    overlay[mask] = (0.55 * overlay[mask]) + (0.45 * red[mask])
    return np.clip(overlay, 0.0, 1.0)


def save_visualization(
    output_path: Path,
    original: np.ndarray,
    reconstruction: np.ndarray,
    anomaly_map: np.ndarray,
    predicted_mask: np.ndarray,
    overlay: np.ndarray,
    gt_mask: np.ndarray | None,
    title: str | None,
    metrics: dict[str, float | int] | None,
    show: bool,
) -> None:
    try:
        import matplotlib
    except ModuleNotFoundError:
        warnings.warn(
            "matplotlib is not installed; saving a PIL-based visualization instead.",
            stacklevel=2,
        )
        save_pil_visualization(
            output_path=output_path,
            original=original,
            reconstruction=reconstruction,
            anomaly_map=anomaly_map,
            predicted_mask=predicted_mask,
            overlay=overlay,
            gt_mask=gt_mask,
            title=title,
            metrics=metrics,
            show=show,
        )
        return

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels: list[tuple[str, np.ndarray, str | None]] = [
        ("Original", original, None),
        ("Reconstruction", reconstruction, None),
        ("Anomaly map", anomaly_map, "magma"),
        ("Predicted mask", predicted_mask, "gray"),
        ("Overlay", overlay, None),
    ]
    if gt_mask is not None:
        panels.append(("Ground truth", gt_mask, "gray"))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (panel_title, panel_image, cmap) in zip(axes, panels):
        if cmap is None:
            ax.imshow(panel_image)
        else:
            ax.imshow(panel_image, cmap=cmap)
        ax.set_title(panel_title)
        ax.axis("off")

    if title:
        fig.suptitle(title)
    if metrics is not None:
        fig.text(
            0.5,
            0.01,
            "precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} "
            "tp={tp} fp={fp} fn={fn}".format(**metrics),
            ha="center",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def save_pil_visualization(
    output_path: Path,
    original: np.ndarray,
    reconstruction: np.ndarray,
    anomaly_map: np.ndarray,
    predicted_mask: np.ndarray,
    overlay: np.ndarray,
    gt_mask: np.ndarray | None,
    title: str | None,
    metrics: dict[str, float | int] | None,
    show: bool,
) -> None:
    panel_items: list[tuple[str, Image.Image]] = [
        ("Original", rgb_array_to_image(original)),
        ("Reconstruction", rgb_array_to_image(reconstruction)),
        ("Anomaly map", rgb_array_to_image(anomaly_map_to_heatmap(anomaly_map))),
        ("Predicted mask", mask_to_image(predicted_mask)),
        ("Overlay", rgb_array_to_image(overlay)),
    ]
    if gt_mask is not None:
        panel_items.append(("Ground truth", mask_to_image(gt_mask)))

    from PIL import ImageDraw

    panel_width, panel_height = panel_items[0][1].size
    title_height = 34 if title else 0
    label_height = 24
    metrics_height = 28 if metrics is not None else 0
    canvas = Image.new(
        "RGB",
        (panel_width * len(panel_items), panel_height + title_height + label_height + metrics_height),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((8, 8), title, fill=(0, 0, 0))
    y_offset = title_height
    for index, (label, image) in enumerate(panel_items):
        x = index * panel_width
        canvas.paste(image, (x, y_offset))
        draw.text((x + 8, y_offset + panel_height + 5), label, fill=(0, 0, 0))
    if metrics is not None:
        text = (
            "precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} "
            "tp={tp} fp={fp} fn={fn}"
        ).format(**metrics)
        draw.text((8, y_offset + panel_height + label_height + 5), text, fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    if show:
        warnings.warn("--show requires matplotlib in this environment.", stacklevel=2)


def rgb_array_to_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(array, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB")


def mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L").convert("RGB")


def anomaly_map_to_heatmap(anomaly_map: np.ndarray) -> np.ndarray:
    values = anomaly_map.astype(np.float32)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value <= min_value:
        normalized = np.zeros_like(values, dtype=np.float32)
    else:
        normalized = (values - min_value) / (max_value - min_value)
    red = np.clip(1.5 * normalized, 0.0, 1.0)
    green = np.clip(1.5 * normalized - 0.5, 0.0, 1.0)
    blue = np.clip(1.0 - 1.5 * normalized, 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


def save_arrays(
    output_dir: Path,
    anomaly_map: np.ndarray,
    predicted_mask: np.ndarray,
    reconstruction: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "anomaly_map.npy", anomaly_map.astype(np.float32))
    Image.fromarray((predicted_mask.astype(np.uint8) * 255), mode="L").save(
        output_dir / "predicted_mask.png"
    )
    Image.fromarray((np.clip(reconstruction, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB").save(
        output_dir / "reconstruction.png"
    )


def save_metadata(output_dir: Path, metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device)
    checkpoint_type = infer_checkpoint_type(args.checkpoint, args.checkpoint_type)
    evaluation_config = resolve_evaluation_config(
        config,
        threshold_percentile=args.threshold_percentile,
        gaussian_sigma=args.gaussian_sigma,
        min_component_area=args.min_component_area,
        no_closing=args.no_closing,
        closing_kernel_size=args.closing_kernel_size,
    )

    model, checkpoint_metadata = load_model_and_checkpoint(args.checkpoint, device)
    class_name = args.class_name or checkpoint_metadata.get("class_name")

    threshold_value, threshold_percentile, threshold_source = resolve_threshold(
        args=args,
        config=config,
        model=model,
        device=device,
        class_name=class_name,
        checkpoint_type=checkpoint_type,
    )

    image_size = int(config.get("image_size", 256))
    original_image, input_tensor = load_image_tensor(args.image_path, image_size, device)
    original = np.asarray(original_image, dtype=np.float32) / 255.0
    with torch.no_grad():
        outputs = model(input_tensor)
        reconstruction_tensor = outputs["reconstruction"].detach().float().clamp(0, 1)
        raw_anomaly_tensor = outputs["anomaly_map"].detach().float()
        smoothed_tensor = gaussian_smooth(
            raw_anomaly_tensor,
            float(evaluation_config["gaussian_sigma"]),
        )

    reconstruction = (
        reconstruction_tensor[0].cpu().permute(1, 2, 0).numpy().astype(np.float32)
    )
    raw_anomaly_map = raw_anomaly_tensor[0, 0].cpu().numpy().astype(np.float32)
    smoothed_anomaly_map = smoothed_tensor[0, 0].cpu().numpy().astype(np.float32)
    predicted_mask = (smoothed_anomaly_map >= threshold_value).astype(np.uint8)
    predicted_mask = postprocess_mask(
        predicted_mask,
        min_component_area=int(evaluation_config["min_component_area"]),
        closing_kernel_size=int(evaluation_config["closing_kernel_size"])
        if bool(evaluation_config["use_closing"])
        else 0,
    )
    overlay = make_overlay(original, predicted_mask)

    gt_mask = load_gt_mask(args.gt_mask_path, image_size) if args.gt_mask_path else None
    metrics = None
    if gt_mask is not None:
        counts = compute_pixel_counts(predicted_mask, gt_mask)
        metrics = {**counts, **precision_recall_f1(counts["tp"], counts["fp"], counts["fn"])}

    output_dir = Path(args.output_dir) if args.output_dir else None
    if args.output_path:
        visualization_path = Path(args.output_path)
    elif output_dir is not None:
        visualization_path = output_dir / "visualization.png"
    else:
        default_dir = Path(config.get("output_path", "./outputs")) / "single_inference"
        visualization_path = default_dir / f"{Path(args.image_path).stem}.png"

    save_visualization(
        output_path=visualization_path,
        original=original,
        reconstruction=reconstruction,
        anomaly_map=smoothed_anomaly_map,
        predicted_mask=predicted_mask,
        overlay=overlay,
        gt_mask=gt_mask,
        title=args.title,
        metrics=metrics,
        show=args.show,
    )

    if output_dir is not None and args.save_arrays:
        save_arrays(
            output_dir=output_dir,
            anomaly_map=smoothed_anomaly_map,
            predicted_mask=predicted_mask,
            reconstruction=reconstruction,
        )

    if output_dir is not None:
        metadata = {
            "image_path": str(args.image_path),
            "checkpoint": str(args.checkpoint),
            "class_name": class_name,
            "threshold_value": float(threshold_value),
            "threshold_percentile": threshold_percentile,
            "threshold_source": threshold_source,
            "gaussian_sigma": float(evaluation_config["gaussian_sigma"]),
            "min_component_area": int(evaluation_config["min_component_area"]),
            "use_closing": bool(evaluation_config["use_closing"]),
            "closing_kernel_size": int(evaluation_config["closing_kernel_size"]),
            "predicted_area": int(predicted_mask.sum()),
            **anomaly_stats(smoothed_anomaly_map),
        }
        if metrics is not None:
            metadata["gt_metrics"] = metrics
        save_metadata(output_dir, metadata)

    print(f"Saved visualization: {visualization_path}")


if __name__ == "__main__":
    main()
