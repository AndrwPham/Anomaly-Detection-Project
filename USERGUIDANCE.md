# User Guidance: Running the Code (Post-Quantization)

This guide focuses on running the project after you already have `.hef` files.
Quantization and model compilation steps are intentionally omitted.

## Prerequisites
- Raspberry Pi 5 + Hailo-8
- HailoRT 4.22.0
- Python with required packages installed (see `README.md`)
- Quantized models:
  - `models/teacher.hef`
  - `models/student.hef`
  - `models/autoencoder.hef`
- Dataset layout:
```
data/
  train/
    normal/
      ...
  test/
    normal/
      ...
    anomaly/
      ...
```

## Environment Setup (venv + dependencies)
Create a virtual environment and install required packages:

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install requirements.txt
```

Note: `hailo_platform` is provided by the Hailo runtime installation.

## Step 1: Calibrate and Generate Specs
This step computes teacher statistics, quantile ranges, and a fixed decision threshold.

```
python -m src.scripts.calibrate_specs \
  --teacher models/teacher.hef \
  --student models/student.hef \
  --autoencoder models/autoencoder.hef \
  --train_dir data/train \
  --test_dir data/test \
  --out_specs specs.npz \
  --out_dir output/calibration
```

Outputs:
- `specs.npz` (used for inference)
- `output/calibration/metrics.txt`
- `output/calibration/roc_curve.png`
- `output/calibration/pr_curve.png`

## Step 2: Image Inference (Heatmaps + CSV)
Run inference on a directory of images and export heatmaps.

```
python -m src.scripts.img_test \
  --teacher models/teacher.hef \
  --student models/student.hef \
  --autoencoder models/autoencoder.hef \
  --specs specs.npz \
  --test_dir data/test \
  --out_dir output/png_infer \
  --save_overlay
```

Outputs:
- `output/png_infer/results.csv`
- `output/png_infer/heatmap/*.png`
- `output/png_infer/overlay/*.png` (when `--save_overlay` is set)

## Step 3: Video Inference (Sampled Frames)
Run inference on a video at a fixed sampling rate (default 1 fps).

```
python -m src.scripts.infer_video_1fps \
  --teacher models/teacher.hef \
  --student models/student.hef \
  --autoencoder models/autoencoder.hef \
  --specs specs.npz \
  --video <path/to/video.mp4> \
  --out_dir output/video_infer \
  --fps 1.0 \
  --save_overlay
```

Outputs:
- `output/video_infer/results.csv`
- `output/video_infer/heatmap/*.png`
- `output/video_infer/overlay/*.png` (when `--save_overlay` is set)

## Common Issues
- If inference fails, confirm the Hailo runtime is installed and the `.hef` files
  match the expected model inputs/outputs.
- If images have inconsistent sizes in a batch, the dataset loader will raise an error.
- Ensure the label folders are named `normal` and `anomaly` (or `good`/`defect`).
