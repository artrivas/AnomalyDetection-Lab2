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

## Training with stronger DRAEM-style losses

The default training config now uses a stronger DRAEM-style objective:

```text
reconstruction: MSE + SSIM
segmentation: BCEWithLogits + Dice + Focal
```

MSE keeps pixel-level reconstruction close to the clean image, while SSIM adds
pressure to preserve local image structure. For segmentation, BCE learns the
binary synthetic anomaly mask, Dice helps with foreground/background imbalance,
and Focal loss can put more weight on hard or small anomalous pixels.

These loss changes require retraining. They do not improve checkpoints that were
already trained with the old loss recipe.

Recommended first retrain only weak classes:

```text
capsule, screw, wood, toothbrush, cable, grid
```

Train from scratch:

```bash
python train.py --class_name capsule
python train.py --class_name screw
```

Resume weak-class retraining from the last checkpoint:

```bash
python train.py --class_name capsule --resume outputs/checkpoints/capsule/last.pth
python train.py --class_name screw --resume outputs/checkpoints/screw/last.pth
python train.py --class_name wood --resume outputs/checkpoints/wood/last.pth
```

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

## Evaluation Debugging Before Retraining

Do not retrain immediately when a class has low F1. First verify that the
evaluation pipeline is measuring the lab target: pixel-level segmentation F1 on
anomalous test images only. Good test images are useful for false-positive
diagnostics, but empty good-image masks should not inflate the main segmentation
F1.

Start by checking anomaly-only metrics:

```bash
python evaluate.py --class_name capsule --checkpoint_type best
```

The main lab metrics in `outputs/metrics/<class_name>/summary.json` are:

```text
anomaly_only_mean_f1
anomaly_only_global_f1
anomaly_only_precision
anomaly_only_recall
```

The old all-image metrics are still saved as `all_images_mean_f1` and
`all_images_global_f1`, but `all_images_mean_f1` may be inflated by normal
images.

Next, run a normal-validation threshold sweep without retraining:

```bash
python evaluate.py --class_name capsule --checkpoint_type best --threshold_sweep
```

This writes:

```text
outputs/metrics/<class_name>/threshold_sweep.csv
outputs/metrics/<class_name>/selected_threshold.json
```

For all classes:

```bash
python evaluate.py --class_name all --checkpoint_type best --threshold_sweep
```

This also writes:

```text
outputs/metrics/threshold_sweep_global.csv
```

Then inspect failed examples visually:

```bash
python evaluate.py --class_name capsule --checkpoint_type best --debug_visuals --max_debug_visuals 20
```

Debug grids are saved under:

```text
outputs/visualizations/debug/<class_name>/
```

Each grid shows the original image, reconstruction, raw anomaly map, smoothed
anomaly map, predicted mask, ground-truth mask, and overlay. Use these to decide
whether the model localizes defects but the threshold is too high, whether
post-processing removes small defects, or whether the raw anomaly map is flat.

Compare post-processing settings before changing training:

```bash
python evaluate.py --class_name capsule --checkpoint_type best --postprocess_ablation
```

The ablation includes no post-processing, Gaussian smoothing, connected
component filtering with small areas such as 5, 10, and 20 pixels, closing, and
combined modes. This matters for small-defect classes such as capsule, screw,
grid, and toothbrush.

Use mask checks when debugging possible dataset or resizing issues:

```bash
python evaluate.py --class_name capsule --check_masks
```

Retrain only after these checks show that the raw anomaly maps are flat,
reconstructions are poor, or the model genuinely fails to separate defective
regions from normal validation scores.

## Evaluation configuration and F1 sensitivity

Pixel-level F1 is computed after converting the continuous anomaly map into a
binary predicted mask. The threshold and post-processing settings can strongly
change F1 without changing the checkpoint.

Lower thresholds usually increase recall, but may reduce precision. Higher
thresholds usually increase precision, but may reduce recall or produce empty
masks. Gaussian smoothing can remove noise, but can also blur tiny defects.
Removing small connected components can remove false positives, but may also
remove real tiny defects. Morphological closing can fill holes, but may expand
predicted regions.

For weak classes such as capsule, screw, grid, toothbrush, and wood, try a lower
threshold and weaker post-processing before retraining:

```bash
python evaluate.py --class_name capsule --checkpoint_type best --threshold_sweep
python evaluate.py --class_name screw --checkpoint_type best --threshold_sweep
python evaluate.py --class_name capsule --checkpoint_type best --postprocess_ablation
python evaluate.py --class_name capsule --checkpoint_type best --use_weak_class_postprocessing
python evaluate.py --class_name screw --checkpoint_type best --use_weak_class_postprocessing
```

Manual override example:

```bash
python evaluate.py --class_name capsule --checkpoint_type best --threshold_percentile 95 --gaussian_sigma 1.0 --min_component_area 0 --no_closing
```

Official thresholds are computed from normal validation images and saved as
`outputs/metrics/<class_name>/threshold.json`. Threshold sweeps that select the
best candidate using test F1 are saved only as analysis outputs and should not be
reported as official metrics.

## Output Folders

```text
outputs/checkpoints/       best and last model checkpoints per class
outputs/logs/              TensorBoard logs and training analytics
outputs/metrics/           threshold files, per-image CSVs, and summary JSON files
outputs/visualizations/    synthetic previews, training debug images, and eval grids
outputs/anomaly_maps/      raw anomaly score maps saved as NumPy arrays
outputs/predicted_masks/   thresholded binary masks saved as PNG files
```

## Single-image inference and visualization

Use `infer_image.py` to run a trained DRAEM checkpoint on one image without
retraining and without requiring ground-truth masks.

Using an explicit threshold:

```bash
python infer_image.py --image_path ./my_image.png --checkpoint outputs/checkpoints/bottle/best.pth --threshold_value 0.05 --output_path outputs/single_inference/my_image.png
```

Using a class validation threshold:

```bash
python infer_image.py --image_path ./dataset/capsule/test/crack/000.png --checkpoint outputs/checkpoints/capsule/best.pth --class_name capsule --threshold_percentile 99.5 --output_path outputs/single_inference/capsule_000.png
```

Saving arrays and metadata:

```bash
python infer_image.py --image_path ./dataset/screw/test/scratch_head/001.png --checkpoint outputs/checkpoints/screw/best.pth --class_name screw --save_arrays --output_dir outputs/single_inference/screw_001/
```

The saved visualization shows:

```text
original image | reconstruction | anomaly map | predicted binary mask | overlay
```

The model outputs a continuous anomaly map. The binary predicted mask depends on
`--threshold_value` or on a validation-derived threshold from
`--threshold_percentile`. For fair MVTec evaluation, thresholds should come from
normal validation images, not test masks. For a custom external image, using
`--threshold_value` may be simpler.

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
