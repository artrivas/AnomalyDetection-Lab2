"""Training loop for one DRAEM model per MVTec AD class."""

from __future__ import annotations

import csv
import random
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.data.mvtec_paths import MVTEC_CLASSES
from src.data.mvtec_train_dataset import MVTecTrainDataset
from src.losses import DRAEMLoss, ReconstructionLoss
from src.models import DRAEM

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


LOG_COLUMNS = [
    "epoch",
    "train_loss",
    "train_mse_loss",
    "train_ssim_loss",
    "train_bce_loss",
    "train_dice_loss",
    "train_focal_loss",
    "train_recon_loss",
    "train_seg_loss",
    "train_total_loss",
    "val_loss",
    "val_recon_loss",
    "val_anomaly_mean",
    "val_anomaly_p995",
    "lr",
    "epoch_seconds",
]


def train_classes(
    class_name: str,
    config: dict[str, Any],
    resume: str | None = None,
    epochs_override: int | None = None,
    early_stopping_patience: int | None = None,
    debug: bool = False,
) -> None:
    """Train one class or all MVTec classes."""

    classes = MVTEC_CLASSES if class_name == "all" else [class_name]
    invalid_classes = [name for name in classes if name not in MVTEC_CLASSES]
    if invalid_classes:
        valid = ", ".join(MVTEC_CLASSES + ["all"])
        raise ValueError(f"Invalid class_name {invalid_classes}. Expected one of: {valid}")

    if class_name == "all" and resume:
        raise ValueError("--resume is only supported when training a single class.")

    for current_class in classes:
        train_one_class(
            current_class,
            config=config,
            resume=resume,
            epochs_override=epochs_override,
            early_stopping_patience=early_stopping_patience,
            debug=debug,
        )


def resolve_loss_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve nested loss config with old flat keys as fallbacks."""

    loss_section = config.get("loss", {})
    if not isinstance(loss_section, dict):
        loss_section = {}

    return {
        "reconstruction_weight": _float_config(
            loss_section, config, "reconstruction_weight", 1.0
        ),
        "segmentation_weight": _float_config(
            loss_section, config, "segmentation_weight", 1.0
        ),
        "mse_weight": _float_config(loss_section, config, "mse_weight", 1.0),
        "ssim_weight": _float_config(loss_section, config, "ssim_weight", 0.0),
        "bce_weight": _float_config(loss_section, config, "bce_weight", 1.0),
        "dice_weight": _float_config(loss_section, config, "dice_weight", 1.0),
        "use_focal": _bool_config(loss_section, config, "use_focal", False),
        "focal_weight": _float_config(loss_section, config, "focal_weight", 1.0),
        "focal_alpha": _float_config(loss_section, config, "focal_alpha", 0.25),
        "focal_gamma": _float_config(loss_section, config, "focal_gamma", 2.0),
    }


def resolve_training_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve nested training config with old flat keys as fallbacks."""

    training_section = config.get("training", {})
    if not isinstance(training_section, dict):
        training_section = {}

    return {
        "epochs": _int_config(training_section, config, "epochs", 100),
        "early_stopping_patience": _int_config(
            training_section, config, "early_stopping_patience", 10
        ),
        "early_stopping_min_delta": _float_config(
            training_section, config, "early_stopping_min_delta", 0.0
        ),
        "learning_rate": _float_config(training_section, config, "learning_rate", 1e-4),
        "batch_size": _int_config(training_section, config, "batch_size", 8),
        "num_workers": _int_config(training_section, config, "num_workers", 4),
        "use_amp": _bool_config(training_section, config, "use_amp", True),
        "gradient_clip_norm": float(
            training_section.get(
                "gradient_clip_norm",
                config.get("gradient_clip_norm", config.get("grad_clip", 1.0)),
            )
        ),
    }


def _float_config(
    nested: dict[str, Any],
    flat: dict[str, Any],
    key: str,
    default: float,
) -> float:
    return float(nested.get(key, flat.get(key, default)))


def _int_config(
    nested: dict[str, Any],
    flat: dict[str, Any],
    key: str,
    default: int,
) -> int:
    return int(nested.get(key, flat.get(key, default)))


def _bool_config(
    nested: dict[str, Any],
    flat: dict[str, Any],
    key: str,
    default: bool,
) -> bool:
    return bool(nested.get(key, flat.get(key, default)))


def _set_nested_config_value(
    config: dict[str, Any],
    section: str,
    key: str,
    value: Any,
) -> None:
    if not isinstance(config.get(section), dict):
        config[section] = {}
    config[section][key] = value


def train_one_class(
    class_name: str,
    config: dict[str, Any],
    resume: str | None = None,
    epochs_override: int | None = None,
    early_stopping_patience: int | None = None,
    debug: bool = False,
) -> None:
    """Train DRAEM for a single MVTec class."""

    config = dict(config)
    loss_config = resolve_loss_config(config)
    training_config = resolve_training_config(config)
    if epochs_override is not None:
        training_config["epochs"] = epochs_override
        config["epochs"] = epochs_override
        _set_nested_config_value(config, "training", "epochs", epochs_override)
    if early_stopping_patience is not None:
        training_config["early_stopping_patience"] = early_stopping_patience
        config["early_stopping_patience"] = early_stopping_patience
        _set_nested_config_value(
            config,
            "training",
            "early_stopping_patience",
            early_stopping_patience,
        )
    if debug:
        config["num_workers"] = 0
        training_config["num_workers"] = 0
        training_config["batch_size"] = min(int(training_config["batch_size"]), 2)
        config["batch_size"] = training_config["batch_size"]
        _set_nested_config_value(config, "training", "num_workers", 0)
        _set_nested_config_value(
            config,
            "training",
            "batch_size",
            training_config["batch_size"],
        )

    seed = int(config.get("seed", 42))
    set_deterministic_seed(seed)

    output_path = Path(config.get("output_path", "./outputs"))
    checkpoint_dir = output_path / "checkpoints" / class_name
    metrics_dir = output_path / "metrics" / class_name
    visual_dir = output_path / "visualizations" / "train_debug" / class_name
    log_dir = output_path / "logs" / class_name
    for directory in [checkpoint_dir, metrics_dir, visual_dir, log_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    save_config_snapshot(config, checkpoint_dir / "config_snapshot.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(training_config["use_amp"]) and device.type == "cuda"
    print(f"Training class '{class_name}' on {device} (AMP={use_amp}).")

    train_dataset = MVTecTrainDataset(
        class_name=class_name,
        dataset_path=config.get("dataset_path", "./dataset"),
        dtd_path=config.get("dtd_path", "./dataset/dtd/images"),
        image_size=int(config.get("image_size", 256)),
        split="train",
        train_val_split=float(config.get("train_val_split", 0.8)),
        seed=seed,
        p_no_anomaly=float(config.get("p_no_anomaly", 0.0)),
    )
    val_dataset = MVTecTrainDataset(
        class_name=class_name,
        dataset_path=config.get("dataset_path", "./dataset"),
        dtd_path=config.get("dtd_path", "./dataset/dtd/images"),
        image_size=int(config.get("image_size", 256)),
        split="val",
        train_val_split=float(config.get("train_val_split", 0.8)),
        seed=seed,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        num_workers=int(training_config["num_workers"]),
        pin_memory=device.type == "cuda",
        generator=generator,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        num_workers=int(training_config["num_workers"]),
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )

    model = DRAEM().to(device)
    criterion = DRAEMLoss(**loss_config)
    val_reconstruction_loss = ReconstructionLoss(
        mse_weight=float(loss_config["mse_weight"]),
        ssim_weight=float(loss_config["ssim_weight"]),
    )
    optimizer = Adam(model.parameters(), lr=float(training_config["learning_rate"]))
    epochs = int(training_config["epochs"])
    patience = int(training_config["early_stopping_patience"])
    min_delta = float(training_config["early_stopping_min_delta"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scaler = GradScaler(enabled=use_amp)

    start_epoch = 1
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    if resume:
        start_epoch, best_val_loss, epochs_without_improvement = load_checkpoint(
            Path(resume),
            model,
            optimizer,
            scheduler,
            scaler,
            device,
        )

    writer = SummaryWriter(str(log_dir)) if SummaryWriter is not None else None
    log_path = metrics_dir / "training_log.csv"
    ensure_log_header(log_path)

    max_train_batches = 2 if debug else None
    max_val_batches = 1 if debug else None

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            grad_clip=float(training_config["gradient_clip_norm"]),
            max_batches=max_train_batches,
        )
        val_metrics = validate_epoch(
            model=model,
            loader=val_loader,
            reconstruction_loss=val_reconstruction_loss,
            device=device,
            use_amp=use_amp,
            threshold_percentile=float(config.get("threshold_percentile", 99.5)),
            reconstruction_weight=float(loss_config["reconstruction_weight"]),
            segmentation_weight=float(loss_config["segmentation_weight"]),
            max_batches=max_val_batches,
        )
        scheduler.step()

        epoch_seconds = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["total_loss"],
            "train_mse_loss": train_metrics["mse_loss"],
            "train_ssim_loss": train_metrics["ssim_loss"],
            "train_bce_loss": train_metrics["bce_loss"],
            "train_dice_loss": train_metrics["dice_loss"],
            "train_focal_loss": train_metrics["focal_loss"],
            "train_recon_loss": train_metrics["reconstruction_loss"],
            "train_seg_loss": train_metrics["segmentation_loss"],
            "train_total_loss": train_metrics["total_loss"],
            "val_loss": val_metrics["loss"],
            "val_recon_loss": val_metrics["recon_loss"],
            "val_anomaly_mean": val_metrics["anomaly_mean"],
            "val_anomaly_p995": val_metrics["anomaly_p995"],
            "lr": lr,
            "epoch_seconds": epoch_seconds,
        }
        append_log_row(log_path, row)
        write_tensorboard(writer, row)

        improved = val_metrics["loss"] < (best_val_loss - min_delta)
        if improved:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        state = checkpoint_state(
            epoch=epoch,
            class_name=class_name,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            use_amp=use_amp,
            best_val_loss=best_val_loss,
            epochs_without_improvement=epochs_without_improvement,
            config=config,
            loss_config=loss_config,
            training_config=training_config,
        )
        save_checkpoint(state, checkpoint_dir / "last.pth")

        if improved:
            save_checkpoint(state, checkpoint_dir / "best.pth")

        checkpoint_every = int(config.get("checkpoint_every", 0))
        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            save_checkpoint(state, checkpoint_dir / f"epoch_{epoch:04d}.pth")

        if epoch % int(config.get("save_visual_every", 5)) == 0:
            save_training_visual(
                model,
                train_loader,
                device,
                visual_dir / f"epoch_{epoch:04d}.png",
            )

        print(
            f"[{class_name}] epoch {epoch}/{epochs} "
            f"total={row['train_total_loss']:.5f} mse={row['train_mse_loss']:.5f} "
            f"ssim={row['train_ssim_loss']:.5f} bce={row['train_bce_loss']:.5f} "
            f"dice={row['train_dice_loss']:.5f} focal={row['train_focal_loss']:.5f} "
            f"val={row['val_loss']:.5f} "
            f"lr={lr:.6g} time={epoch_seconds:.1f}s"
        )
        if patience > 0 and epochs_without_improvement >= patience:
            print(
                f"[{class_name}] early stopping at epoch {epoch}: "
                f"no val_loss improvement > {min_delta:g} for {patience} epochs."
            )
            break

    if writer is not None:
        writer.close()

    from src.eval.thresholding import compute_normal_validation_threshold

    threshold_result = compute_normal_validation_threshold(
        class_name=class_name,
        checkpoint_path=checkpoint_dir / "last.pth",
        config=config,
        device=device,
    )
    print(
        f"[{class_name}] threshold={threshold_result['threshold']:.6f} "
        f"p{threshold_result['threshold_percentile']}"
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: DRAEMLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    grad_clip: float,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    totals = {
        "total_loss": 0.0,
        "reconstruction_loss": 0.0,
        "segmentation_loss": 0.0,
        "mse_loss": 0.0,
        "ssim_loss": 0.0,
        "bce_loss": 0.0,
        "dice_loss": 0.0,
        "focal_loss": 0.0,
    }
    count = 0

    for batch_index, batch in enumerate(iter_progress(loader, desc="train"), start=1):
        clean = batch["clean_image"].to(device, non_blocking=True)
        synthetic = batch["anomalous_image"].to(device, non_blocking=True)
        mask = batch["anomaly_mask"].to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            outputs = model(synthetic)
            losses = criterion(
                reconstruction=outputs["reconstruction"],
                clean_image=clean,
                logits=outputs["logits"],
                synthetic_mask=mask,
            )

        scaler.scale(losses["total_loss"]).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

        batch_size = clean.shape[0]
        for key in totals:
            totals[key] += losses[key].detach().item() * batch_size
        count += batch_size

        if max_batches is not None and batch_index >= max_batches:
            break

    return {key: value / max(1, count) for key, value in totals.items()}


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    reconstruction_loss: ReconstructionLoss,
    device: torch.device,
    use_amp: bool,
    threshold_percentile: float,
    reconstruction_weight: float,
    segmentation_weight: float,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    recon_total = 0.0
    anomaly_sum = 0.0
    anomaly_pixels = 0
    p995_values: list[float] = []
    count = 0

    for batch_index, batch in enumerate(iter_progress(loader, desc="val"), start=1):
        clean = batch["image"].to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            outputs = model(clean)
            recon_terms = reconstruction_loss(outputs["reconstruction"], clean)

        anomaly_map = outputs["anomaly_map"].detach().float()
        batch_size = clean.shape[0]
        recon_total += recon_terms["reconstruction_loss"].detach().item() * batch_size
        anomaly_sum += anomaly_map.sum().item()
        anomaly_pixels += anomaly_map.numel()
        p995_values.append(percentile(anomaly_map, threshold_percentile))
        count += batch_size

        if max_batches is not None and batch_index >= max_batches:
            break

    recon_loss = recon_total / max(1, count)
    anomaly_mean = anomaly_sum / max(1, anomaly_pixels)
    anomaly_p995 = sum(p995_values) / max(1, len(p995_values))
    val_loss = reconstruction_weight * recon_loss + segmentation_weight * anomaly_mean
    return {
        "loss": val_loss,
        "recon_loss": recon_loss,
        "anomaly_mean": anomaly_mean,
        "anomaly_p995": anomaly_p995,
    }


@torch.no_grad()
def save_training_visual(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
) -> None:
    model.eval()
    batch = next(iter(loader))
    clean = batch["clean_image"].to(device)
    synthetic = batch["anomalous_image"].to(device)
    mask = batch["anomaly_mask"].to(device)
    outputs = model(synthetic)

    n = min(4, clean.shape[0])
    grid_items = []
    for index in range(n):
        anomaly_map = outputs["anomaly_map"][index].repeat(3, 1, 1)
        mask_rgb = mask[index].repeat(3, 1, 1)
        row = torch.cat(
            [
                clean[index].detach().cpu(),
                synthetic[index].detach().cpu(),
                outputs["reconstruction"][index].detach().cpu(),
                anomaly_map.detach().cpu(),
                mask_rgb.detach().cpu(),
            ],
            dim=2,
        )
        grid_items.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(torch.stack(grid_items, dim=0), output_path, nrow=1)


def checkpoint_state(
    epoch: int,
    class_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: GradScaler,
    use_amp: bool,
    best_val_loss: float,
    epochs_without_improvement: int,
    config: dict[str, Any],
    loss_config: dict[str, Any],
    training_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "class_name": class_name,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if use_amp else None,
        "best_val_loss": best_val_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "config": config,
        "loss_config": dict(loss_config),
        "training_config": dict(training_config),
        "rng_state": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": get_numpy_rng_state(),
        },
    }


def save_checkpoint(state: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[int, float, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    if checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])

    rng_state = checkpoint.get("rng_state", {})
    if rng_state.get("python") is not None:
        random.setstate(rng_state["python"])
    if rng_state.get("torch") is not None:
        torch.set_rng_state(rng_state["torch"])
    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng_state["cuda"])
    if rng_state.get("numpy") is not None:
        set_numpy_rng_state(rng_state["numpy"])

    start_epoch = int(checkpoint["epoch"]) + 1
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
    print(f"Resumed from {path} at epoch {start_epoch}.")
    return start_epoch, best_val_loss, epochs_without_improvement


def ensure_log_header(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        with log_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            existing_columns = reader.fieldnames or []
            rows = list(reader)
        if all(column in existing_columns for column in LOG_COLUMNS):
            return
        with log_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=LOG_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in LOG_COLUMNS})
        return
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LOG_COLUMNS)
        writer.writeheader()


def append_log_row(log_path: Path, row: dict[str, Any]) -> None:
    with log_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LOG_COLUMNS)
        writer.writerow({key: row.get(key, "") for key in LOG_COLUMNS})


def write_tensorboard(writer: Any, row: dict[str, Any]) -> None:
    if writer is None:
        return
    epoch = int(row["epoch"])
    for key in LOG_COLUMNS:
        if key == "epoch":
            continue
        writer.add_scalar(key, row[key], epoch)


def save_config_snapshot(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=True)
    output_config = path.parents[2] / "metrics" / path.parent.name / "config_snapshot.yaml"
    output_config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, output_config)


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset if worker_info is not None else None
    synthetic_generator = getattr(dataset, "synthetic_generator", None)
    if synthetic_generator is None:
        return

    synthetic_generator.rng = random.Random(worker_seed)
    try:
        import numpy as np

        synthetic_generator.np_rng = np.random.default_rng(worker_seed)
    except ImportError:
        pass


def get_numpy_rng_state() -> Any:
    try:
        import numpy as np

        return np.random.get_state()
    except ImportError:
        return None


def set_numpy_rng_state(state: Any) -> None:
    try:
        import numpy as np

        np.random.set_state(state)
    except ImportError:
        pass


def percentile(values: torch.Tensor, percentile_value: float) -> float:
    flattened = values.flatten()
    if flattened.numel() == 0:
        return 0.0
    quantile = torch.quantile(flattened, percentile_value / 100.0)
    return float(quantile.item())


def iter_progress(loader: DataLoader, desc: str) -> Any:
    if tqdm is None:
        return loader
    return tqdm(loader, desc=desc, leave=False)
