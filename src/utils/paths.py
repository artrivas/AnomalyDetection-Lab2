"""Filesystem validation helpers."""

from __future__ import annotations

from pathlib import Path

from src.data.mvtec_paths import IGNORE_FOLDERS, MVTEC_CLASSES


def verify_dataset(dataset_path: str | Path, dtd_path: str | Path) -> None:
    """Validate the expected MVTec AD and DTD folder structure."""

    dataset_root = Path(dataset_path)
    dtd_root = Path(dtd_path)
    errors: list[str] = []

    if not dataset_root.is_dir():
        errors.append(f"Dataset path does not exist: {dataset_root}")
    if not dtd_root.is_dir():
        errors.append(f"DTD image path does not exist: {dtd_root}")

    if dataset_root.is_dir():
        discovered_ignored = [
            folder for folder in IGNORE_FOLDERS if (dataset_root / folder).exists()
        ]
        if discovered_ignored:
            ignored = ", ".join(discovered_ignored)
            print(f"Ignoring non-MVTec folders under dataset: {ignored}")

        for class_name in MVTEC_CLASSES:
            class_root = dataset_root / class_name
            train_good_root = class_root / "train" / "good"
            test_root = class_root / "test"
            ground_truth_root = class_root / "ground_truth"

            if not class_root.is_dir():
                errors.append(f"MVTec class folder is missing: {class_root}")
                continue

            if not train_good_root.is_dir():
                errors.append(
                    f"Training normal-image folder is missing for '{class_name}': "
                    f"{train_good_root}"
                )

            if not test_root.is_dir():
                errors.append(f"Test folder is missing for '{class_name}': {test_root}")
            else:
                anomalous_test_dirs = [
                    path
                    for path in test_root.iterdir()
                    if path.is_dir() and path.name != "good"
                ]
                if anomalous_test_dirs and not ground_truth_root.is_dir():
                    defects = ", ".join(sorted(path.name for path in anomalous_test_dirs))
                    errors.append(
                        f"Ground-truth folder is missing for '{class_name}' even though "
                        f"anomalous test folders exist ({defects}): {ground_truth_root}"
                    )

    if errors:
        joined_errors = "\n".join(f"- {error}" for error in errors)
        raise FileNotFoundError(f"Dataset verification failed:\n{joined_errors}")

    print(f"Dataset verified: {dataset_root}")
    print(f"DTD images verified: {dtd_root}")
    print(f"MVTec classes verified: {len(MVTEC_CLASSES)}")
