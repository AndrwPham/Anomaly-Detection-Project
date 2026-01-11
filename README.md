# Application of Edge AI for Visual Anomaly Detection in Conveyor Belt Inspection (using Milkbox as dataset)

Edge-based visual anomaly detection for milk carton box inspection on a conveyor belt.
This project uses the EfficientAD teacher-student-autoencoder pipeline and runs on
Raspberry Pi 5 with a Hailo-8 accelerator for low-latency, on-device inference.

## Overview
- Purpose: detect surface defects and foreign objects on milk carton boxes in a controlled
  industrial inspection setup.
- Approach: unsupervised anomaly detection trained only on normal samples.
- Edge deployment: model inference is executed locally on the edge device to avoid
  cloud latency and network dependency.

## System Summary
- Camera: single RGB camera facing a controlled inspection area with black background
  and stable lighting.
- Products: milk carton boxes.
- Anomalies: dents, tears, and foreign objects (treated as anomalies).
- Pipeline separation:
  - Offline: training, quantization, calibration, and compilation.
  - On-device: preprocessing, inference, and threshold-based decision.

## Hardware and Software
- Hardware:
  - Raspberry Pi 5
  - Hailo-8 accelerator
- Hailo runtime:
  - HailoRT 4.22.0
- Python: 3.10+

## Dataset
Self-collected dataset captured under a conveyor-belt-like setup with controlled lighting.

### Split (from project report)
- Training set (normal only): 104 images
- Test set:
  - Normal: 39 images
  - Anomalous: 39 images

### Expected layout (used by scripts)
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
Notes:
- `src/scripts/calibrate_specs.py` expects class folders under `--test_dir`
  and treats any label other than `normal` as anomaly.
- `src/scripts/img_test.py` infers labels from folder names:
  `normal/good` -> normal, `anomaly/defect/bad` -> anomaly.

## Models
- Three models are used: teacher, student, autoencoder.
- Models are compiled to Hailo `.hef` format obtained through quantization.
- The runtime input size is inferred from the student `.hef` file.

## Project Structure
```
src/
  core/            Core EfficientAD math, scoring, evaluation, and IO utils
  common/          Hailo inference wrapper and utility helpers
  dataset/         Simple dataset and dataloader utilities
  scripts/         Calibration and inference scripts
```

## Dependencies (used by scripts)
- numpy
- opencv-python
- pillow
- pandas
- scikit-learn
- matplotlib
- loguru
- hailo_platform (Hailo runtime Python bindings)

No `requirements.txt` is included in the repo.

## Calibration (specs and threshold)
Generate calibration statistics and decision threshold (Youden's J):
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
- `specs.npz` with `teacher_mean`, `teacher_std`, quantile stats, and threshold.
- `output/calibration/metrics.txt`
- `output/calibration/roc_curve.png`
- `output/calibration/pr_curve.png`

## Image Inference (heatmaps and CSV)
Run inference on image folders:
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

## Video Inference (sampled frames)
Run inference on a video at a fixed sampling rate (default 1 fps):
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

## Results
- Image-level AUC: 0.97
- Test set: 78 images, 74 correct, overall accuracy 94.87%
- Normal samples: 36 correct, 3 false positives
- Anomalous samples: 38 correct, 1 false negative

## Notes and Limitations
- The dataset is relatively small and collected under controlled conditions.
- A fixed threshold (Youden's J) is used for deterministic runtime decisions.
- Additional evaluation metrics may be useful for deployment-specific tradeoffs.
