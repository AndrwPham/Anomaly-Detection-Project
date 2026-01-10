import os
import argparse
import numpy as np
import pandas as pd
import cv2
from PIL import Image

from src.common.infer_model import HailoInfer
from src.core.efficientad_core import EfficientADCore
from src.core.io_utils import load_specs_npz
from src.core.scoring import score_from_map


def onchip_transform(
        img: np.ndarray,
        image_size: int = 256
) -> np.ndarray:
    img_pil = Image.fromarray(img)
    img_pil = img_pil.resize((image_size, image_size), resample=Image.BILINEAR)

    img = np.array(img_pil).astype(np.uint8)
    return img


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(root: str):
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    return paths


def infer_label_from_path(path: str):
    """
    Assumes your test structure contains:
      data/test/normal/...
      data/test/anomaly/...
    """
    parts = os.path.normpath(path).split(os.sep)
    for p in parts:
        p_low = p.lower()
        if p_low in ("normal", "good"):
            return "normal"
        if p_low in ("anomaly", "defect", "bad"):
            return "anomaly"
    return "normal"


def preprocess_uint8_pil(path: str, size_hw):
    """
    PIL-open -> RGB -> resize bilinear -> uint8 HWC
    """
    h, w = size_hw
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = im.resize((w, h), resample=Image.BILINEAR)
        return np.array(im, dtype=np.uint8)


def save_heatmap_png(heatmap_float: np.ndarray, out_png: str):
    hm = heatmap_float.astype(np.float32)

    # Normalize to [0, 1]
    hm_norm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-12)

    # Convert to uint8 grayscale [0, 255]
    hm_gray = (hm_norm * 255).astype(np.uint8)

    # Save grayscale PNG
    cv2.imwrite(out_png, hm_gray)


def save_overlay_png(image_path: str, heatmap_float: np.ndarray, out_png: str, alpha: float = 0.4):
    """
    Overlay heatmap on original image and save PNG.
    """
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        return

    h_img, w_img = bgr.shape[:2]

    hm = heatmap_float.astype(np.float32)
    hm_norm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-12)
    hm_u8 = (hm_norm * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)

    # Ensure same size
    if hm_color.shape[0] != h_img or hm_color.shape[1] != w_img:
        hm_color = cv2.resize(hm_color, (w_img, h_img), interpolation=cv2.INTER_LINEAR)

    overlay = cv2.addWeighted(bgr, 1.0 - alpha, hm_color, alpha, 0)
    cv2.imwrite(out_png, overlay)


def main():
    ap = argparse.ArgumentParser("Infer EfficientAD and export heatmaps as PNG")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--autoencoder", required=True)
    ap.add_argument("--specs", required=True, help="specs.npz from calibrate_specs")
    ap.add_argument("--test_dir", required=True, help="data/test")
    ap.add_argument("--out_dir", default="output/png_infer")
    ap.add_argument("--save_overlay", action="store_true", help="also save overlay PNG")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    teacher = HailoInfer(args.teacher, input_type="UINT8", output_type="FLOAT32")
    student = HailoInfer(args.student, input_type="UINT8", output_type="FLOAT32")
    autoenc = HailoInfer(args.autoencoder, input_type="UINT8", output_type="FLOAT32")

    # input size
    h, w, _ = student.get_input_shape()
    size_hw = (h, w)

    # load specs
    specs = load_specs_npz(args.specs)
    teacher_mean = specs["teacher_mean"]
    teacher_std  = specs["teacher_std"]
    q_st_start   = specs["q_st_start"]
    q_st_end     = specs["q_st_end"]
    q_ae_start   = specs["q_ae_start"]
    q_ae_end     = specs["q_ae_end"]
    threshold    = float(specs["threshold"])

    core = EfficientADCore(teacher, student, autoenc)

    image_paths = list_images(args.test_dir)
    if not image_paths:
        raise RuntimeError(f"No images found under: {args.test_dir}")

    rows = []

    # Output folders
    hm_dir = os.path.join(args.out_dir, "heatmap")
    ov_dir = os.path.join(args.out_dir, "overlay")
    os.makedirs(hm_dir, exist_ok=True)
    if args.save_overlay:
        os.makedirs(ov_dir, exist_ok=True)

    # -----------------------------
    # Accuracy / metrics counters
    # -----------------------------
    total = 0
    correct = 0

    # Confusion matrix for binary classification (normal=0, anomaly=1)
    tn = fp = fn = tp = 0

    for path in image_paths:
        gt_label = infer_label_from_path(path)  # "normal" or "anomaly"
        gt = 0 if gt_label == "normal" else 1

        # preprocess
        img_u8 = preprocess_uint8_pil(path, size_hw)

        # predict map
        map_combined, _, _ = core.predict(
            image=img_u8,
            teacher_mean=teacher_mean,
            teacher_std=teacher_std,
            q_st_start=q_st_start, q_st_end=q_st_end,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        )

        # score + heatmap (resized for visualization)
        score, heatmap = score_from_map(map_combined, path)
        pred = 0 if score <= threshold else 1

        # -----------------------------
        # Update metrics
        # -----------------------------
        total += 1
        if pred == gt:
            correct += 1

        if gt == 0 and pred == 0:
            tn += 1
        elif gt == 0 and pred == 1:
            fp += 1
        elif gt == 1 and pred == 0:
            fn += 1
        else:
            tp += 1

        base = os.path.splitext(os.path.basename(path))[0]
        prefix = "anomaly" if pred == 1 else "normal"
        out_hm_png = os.path.join(hm_dir, f"{prefix}-img-{base}.png")

        save_heatmap_png(heatmap, out_hm_png)

        out_ov_png = None
        if args.save_overlay:
            out_ov_png = os.path.join(ov_dir, f"{base}_overlay.png")
            save_overlay_png(path, heatmap, out_ov_png)

        rows.append({
            "image_path": path,
            "ground_truth": gt_label,
            "score": float(score),
            "threshold": float(threshold),
            "predict": "normal" if pred == 0 else "anomaly",
            "is_correct": int(pred == gt),
            "heatmap_png": out_hm_png,
            "overlay_png": out_ov_png if out_ov_png else ""
        })

    # Save CSV
    results_csv = os.path.join(args.out_dir, "results.csv")
    pd.DataFrame(rows).to_csv(results_csv, index=False)

    # -----------------------------
    # Log accuracy summary
    # -----------------------------
    acc = (correct / total * 100.0) if total > 0 else 0.0
    normal_acc = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0   # accuracy on GT normal
    anomaly_acc = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0  # accuracy on GT anomaly

    print("Done.")
    print("Results:", results_csv)
    print("Heatmaps:", hm_dir)
    if args.save_overlay:
        print("Overlays:", ov_dir)

    print("\n=== Accuracy Summary ===")
    print(f"Total images: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.2f}%")
    print("\n=== Confusion Matrix (GT rows, Pred cols) ===")
    print("             Pred Normal   Pred Anomaly")
    print(f"GT Normal        {tn:>5}         {fp:>5}")
    print(f"GT Anomaly       {fn:>5}         {tp:>5}")
    print("\n=== Per-class accuracy (by GT class) ===")
    print(f"Normal:  {normal_acc:.2f}%")
    print(f"Anomaly: {anomaly_acc:.2f}%")

    teacher.close()
    student.close()
    autoenc.close()


if __name__ == "__main__":
    main()
