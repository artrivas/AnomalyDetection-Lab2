"""Evaluation entrypoint for DRAEM MVTec AD experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DRAEM F1 metrics")
    parser.add_argument(
        "--config",
        default="configs/draem_mvtec.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--class_name",
        required=True,
        help="MVTec class name or 'all'.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path for a single class. Defaults to output checkpoint_type.",
    )
    parser.add_argument(
        "--checkpoint_type",
        default="best",
        choices=["best", "last"],
        help="Checkpoint filename stem to use when --checkpoint is not provided.",
    )
    parser.add_argument(
        "--save_visuals",
        action="store_true",
        help="Save evaluation visual grids.",
    )
    parser.add_argument(
        "--max_visuals_per_defect",
        type=int,
        default=None,
        help="Maximum visualizations to save per defect type.",
    )
    parser.add_argument(
        "--gaussian_sigma",
        type=float,
        default=None,
        help="Override gaussian smoothing sigma.",
    )
    parser.add_argument(
        "--min_component_area",
        type=int,
        default=None,
        help="Override minimum connected-component area.",
    )
    parser.add_argument(
        "--closing_kernel_size",
        type=int,
        default=None,
        help="Override morphological closing kernel size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    from src.eval.evaluate_f1 import evaluate_classes

    evaluate_classes(
        class_name=args.class_name,
        config=config,
        checkpoint=args.checkpoint,
        checkpoint_type=args.checkpoint_type,
        save_visuals=args.save_visuals,
        gaussian_sigma=args.gaussian_sigma,
        min_component_area=args.min_component_area,
        closing_kernel_size=args.closing_kernel_size,
        max_visuals_per_defect=args.max_visuals_per_defect,
    )


if __name__ == "__main__":
    main()
