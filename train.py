"""Training entrypoint for the DRAEM MVTec AD lab scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import verify_dataset


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
    parser = argparse.ArgumentParser(description="DRAEM MVTec AD training scaffold")
    parser.add_argument(
        "--config",
        default="configs/draem_mvtec.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--verify_dataset",
        action="store_true",
        help="Validate dataset and DTD paths from the config, then exit.",
    )
    parser.add_argument(
        "--class_name",
        default=None,
        help="MVTec class name for class-specific commands.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint to resume from.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs from the config.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run a short debug training loop with fewer workers and batches.",
    )
    parser.add_argument(
        "--preview_synthetic",
        type=int,
        default=0,
        metavar="N",
        help="Save N synthetic anomaly preview examples, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.verify_dataset:
        verify_dataset(
            dataset_path=config.get("dataset_path", "./dataset"),
            dtd_path=config.get("dtd_path", "./dataset/dtd/images"),
        )
        return

    if args.preview_synthetic:
        if not args.class_name:
            raise ValueError("--class_name is required with --preview_synthetic")
        preview_synthetic_anomalies(args.class_name, args.preview_synthetic, config)
        return

    if not args.class_name:
        raise ValueError(
            "--class_name is required for training. Use a class name such as "
            "'bottle' or use 'all'."
        )

    from src.train.train_draem import train_classes

    train_classes(
        class_name=args.class_name,
        config=config,
        resume=args.resume,
        epochs_override=args.epochs,
        debug=args.debug,
    )


def preview_synthetic_anomalies(
    class_name: str,
    count: int,
    config: dict[str, Any],
) -> None:
    if count <= 0:
        raise ValueError(f"--preview_synthetic must be positive, got: {count}")

    import torch
    from torchvision.utils import save_image

    from src.data.mvtec_train_dataset import MVTecTrainDataset

    output_dir = (
        Path(config.get("output_path", "./outputs"))
        / "visualizations"
        / "synthetic_preview"
        / class_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MVTecTrainDataset(
        class_name=class_name,
        dataset_path=config.get("dataset_path", "./dataset"),
        dtd_path=config.get("dtd_path", "./dataset/dtd/images"),
        image_size=int(config.get("image_size", 256)),
        split="train",
        train_val_split=float(config.get("train_val_split", 0.8)),
        seed=int(config.get("seed", 42)),
        p_no_anomaly=float(config.get("p_no_anomaly", 0.0)),
    )

    for index in range(count):
        sample = dataset[index % len(dataset)]
        clean_image = sample["clean_image"]
        synthetic_image = sample["anomalous_image"]
        synthetic_mask = sample["anomaly_mask"].repeat(3, 1, 1)
        preview = torch.cat([clean_image, synthetic_image, synthetic_mask], dim=2)
        save_image(preview, output_dir / f"{index:03d}.png")

    print(f"Saved {count} synthetic previews to: {output_dir}")


if __name__ == "__main__":
    main()
