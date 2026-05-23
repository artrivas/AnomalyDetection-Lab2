"""Readable evaluation visualizations for DRAEM results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


PANEL_TITLES = [
    "Original",
    "Reconstruction",
    "Anomaly map",
    "Predicted mask",
    "Ground truth",
]


def save_result_visualization(
    image: torch.Tensor,
    reconstruction: torch.Tensor,
    anomaly_map: np.ndarray,
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    class_name: str,
    defect_type: str,
    image_name: str,
    f1: float,
    output_path: str | Path,
) -> None:
    """Save original | reconstruction | heatmap | pred mask | GT mask."""

    original_image = tensor_rgb_to_image(image)
    panel_size = original_image.size
    panels = [
        original_image,
        tensor_rgb_to_image(reconstruction).resize(panel_size, Image.Resampling.BILINEAR),
        anomaly_map_to_heatmap(anomaly_map).resize(panel_size, Image.Resampling.BILINEAR),
        mask_to_image(predicted_mask).resize(panel_size, Image.Resampling.NEAREST),
        mask_to_image(ground_truth_mask).resize(panel_size, Image.Resampling.NEAREST),
    ]

    panel_width, panel_height = panel_size
    title_height = 42
    label_height = 28
    border_width = 1
    canvas_width = panel_width * len(panels) + border_width * (len(panels) - 1)
    canvas_height = title_height + label_height + panel_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = load_readable_font(size=14)
    title_font = load_readable_font(size=16)

    title = f"{class_name} | {defect_type} | {image_name} | F1={f1:.4f}"
    draw.text((10, 12), title, fill=(0, 0, 0), font=title_font)

    for index, panel in enumerate(panels):
        x = index * (panel_width + border_width)
        if index > 0:
            draw.line(
                [(x - border_width, title_height), (x - border_width, canvas_height)],
                fill=(220, 220, 220),
                width=border_width,
            )
        draw.text((x + 8, title_height + 6), PANEL_TITLES[index], fill=(0, 0, 0), font=font)
        canvas.paste(panel, (x, title_height + label_height))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def tensor_rgb_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().float().clamp(0.0, 1.0)
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Expected RGB tensor with shape [3,H,W], got {tuple(tensor.shape)}")

    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def anomaly_map_to_heatmap(anomaly_map: np.ndarray) -> Image.Image:
    normalized = normalize_array(anomaly_map.astype(np.float32))
    red = np.clip(1.5 * normalized, 0.0, 1.0)
    green = np.clip(1.5 - np.abs(2.0 * normalized - 1.0) * 1.5, 0.0, 1.0)
    blue = np.clip(1.5 * (1.0 - normalized), 0.0, 1.0)
    heatmap = np.stack([red, green, blue], axis=-1)
    return Image.fromarray((heatmap * 255.0).round().astype(np.uint8), mode="RGB")


def mask_to_image(mask: np.ndarray) -> Image.Image:
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError(f"Expected binary mask with shape [H,W], got {mask.shape}")
    mask = (mask.astype(bool).astype(np.uint8) * 255)
    return Image.fromarray(mask, mode="L").convert("RGB")


def normalize_array(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected anomaly map with shape [H,W], got {array.shape}")
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value <= min_value:
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_value) / (max_value - min_value)


def load_readable_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def should_save_visual(
    saved_counts: dict[str, int],
    defect_type: str,
    max_visuals_per_defect: int | None,
) -> bool:
    if max_visuals_per_defect is None:
        return True
    if max_visuals_per_defect <= 0:
        return False
    return saved_counts.get(defect_type, 0) < max_visuals_per_defect


def mark_visual_saved(saved_counts: dict[str, int], defect_type: str) -> None:
    saved_counts[defect_type] = saved_counts.get(defect_type, 0) + 1
