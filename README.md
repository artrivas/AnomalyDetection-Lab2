# DRAEM for MVTec AD Anomaly Detection and Segmentation

## Objective

This laboratory implements a DRAEM-style anomaly detection and segmentation pipeline for MVTec AD. The model is trained only on normal images, generates synthetic anomalies with DTD textures, learns to reconstruct normal-looking images, produces anomaly maps, thresholds those maps into binary masks, and evaluates performance with pixel-level F1.

At evaluation time, each test image produces:

- a reconstruction
- an anomaly map
- a predicted binary anomaly mask
- pixel-level precision, recall, and F1 against the ground-truth mask
- optional visualizations for qualitative inspection

## Dataset Structure

The dataset root is configured as `./dataset` by default. Each MVTec class must use this layout:

```text
dataset/<class>/
  train/good/                         normal images used for training
  test/good/                          normal test images
  test/<defect_type>/                 anomalous test images
  ground_truth/<defect_type>/         binary masks for anomalous test images
```

DTD textures are stored separately and are used only to synthesize training anomalies:

```text
dataset/dtd/images/
```

The project intentionally ignores these dataset helper folders when validating or iterating over MVTec classes:

```text
dataset/dtd/
dataset/mvtec_ad_evaluation/
```

## MVTec Classes

The 15 valid MVTec AD classes are:

```text
bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut,
pill, screw, tile, toothbrush, transistor, wood, zipper
```

Use `--class_name all` to train or evaluate every valid class. The folders `dtd` and `mvtec_ad_evaluation` are not treated as object classes.

## Installation

Install the Python dependencies from the project root:

```bash
pip install -r requirements.txt
```

## Verify Dataset

Check that the MVTec and DTD paths are present and correctly structured:

```bash
python train.py --verify_dataset
```

## Preview Synthetic Anomalies

Save synthetic anomaly previews for one class:

```bash
python train.py --class_name bottle --preview_synthetic 16
```

Previews are written under:

```text
outputs/visualizations/synthetic_preview/<class_name>/
```

Each preview shows the clean image, synthetic anomalous image, and synthetic anomaly mask.

## Training

Train one class:

```bash
python train.py --class_name bottle
```

Resume training:

```bash
python train.py --class_name bottle --resume outputs/checkpoints/bottle/last.pth
```

Train all classes:

```bash
python train.py --class_name all
```

Training uses only `train/good` images. Real anomalous test images and `ground_truth` masks are not used for training.

## Evaluation

Evaluate one class with the best checkpoint and save visualizations:

```bash
python evaluate.py --class_name bottle --checkpoint_type best --save_visuals
```

Evaluate all classes:

```bash
python evaluate.py --class_name all --checkpoint_type best --save_visuals
```

Visualizations are saved as horizontal grids:

```text
original image | reconstruction | anomaly map heatmap | predicted binary mask | ground-truth mask
```

The output path is:

```text
outputs/visualizations/eval/<class_name>/<defect_type>/<image_name>.png
```

Limit saved visualizations per defect type with:

```bash
python evaluate.py --class_name bottle --save_visuals --max_visuals_per_defect 10
```

## Output Folders

```text
outputs/checkpoints/       best and last model checkpoints per class
outputs/logs/              TensorBoard logs and training analytics
outputs/metrics/           threshold files, per-image CSVs, and summary JSON files
outputs/visualizations/    synthetic previews, training debug images, and eval grids
outputs/anomaly_maps/      raw anomaly score maps saved as NumPy arrays
outputs/predicted_masks/   thresholded binary masks saved as PNG files
```

## Metrics

Metrics are computed at the pixel level for anomaly segmentation.

Precision measures how many predicted anomalous pixels are truly anomalous:

```text
precision = TP / (TP + FP)
```

Recall measures how many ground-truth anomalous pixels were found:

```text
recall = TP / (TP + FN)
```

F1 is the harmonic mean of precision and recall:

```text
F1 = 2 * precision * recall / (precision + recall)
```

The main reported metric for this lab is pixel-level segmentation F1.

## Thresholding

The main anomaly threshold is computed from normal validation images using the configured `threshold_percentile`, which defaults to `99.5`.

This means the model estimates a high normal anomaly-score percentile from images that should contain no real defects, then applies that threshold to test anomaly maps. This is fairer than choosing a threshold from test masks because it avoids tuning directly on the test ground truth.

The threshold file is saved under:

```text
outputs/metrics/<class_name>/threshold.json
```

## Training Analytics

Training writes class-specific analytics under `outputs/`.

- `training_log.csv` records epoch-level losses and validation statistics.
- TensorBoard logs are written under `outputs/logs/<class_name>/`.
- `best.pth` stores the best checkpoint according to validation behavior.
- `last.pth` stores the latest checkpoint for resume.
- Resume restores the model, optimizer, scheduler state when present, AMP scaler state when enabled, epoch, and class-specific metadata.

## Reproducibility

Experiment settings live in:

```text
configs/draem_mvtec.yaml
```

The config controls dataset paths, output paths, image size, batch size, epochs, learning rate, train/validation split, threshold percentile, AMP, and seed.

Training is class-wise: each MVTec class gets its own model and checkpoints. The normal-image train/validation split is deterministic for a fixed `seed`, which keeps thresholding and validation behavior reproducible across runs.

## Implementation Notes

This implementation is inspired by the official DRAEM repository and paper, but it is adapted for this laboratory's F1-based evaluation workflow. The lab focuses on normal-only training, DTD-based synthetic anomaly generation, reconstruction plus segmentation outputs, validation-derived thresholding, and pixel-level segmentation F1 reporting.
