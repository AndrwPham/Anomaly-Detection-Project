import os
import argparse
import numpy as np
import pandas as pd
import cv2
from PIL import Image

from src.common.infer_model import HailoInfer
from src.core.efficientad_core import EfficientADCore
from src.core.io_utils import load_specs_npz
from src.core.scoring import score_from_map_with_size


def preprocess_bgr_to_uint8_rgb(bgr: np.ndarray, size_hw):
    """
    Video frame comes as BGR uint8.
    Convert to RGB, resize with PIL bilinear, return RGB uint8 HWC.
    """
    h, w = size_hw
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    im = im.resize((w, h), resample=Image.BILINEAR)
    return np.array(im, dtype=np.uint8)


def _normalize_heatmap_to_u8(heatmap_float: np.ndarray) -> np.ndarray:
    """
    Normalize float heatmap to uint8 grayscale:
    - 0   (black) = low anomaly (normal)
    - 255 (white) = high anomaly
    """
    hm = heatmap_float.astype(np.float32)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-12)
    hm_u8 = (hm * 255).astype(np.uint8)
    return hm_u8


def save_heatmap_png(heatmap_float: np.ndarray, out_png: str):
    """
    Save grayscale heatmap:
    Black = normal (low anomaly), White = anomaly (high anomaly)
    """
    hm_u8 = _normalize_heatmap_to_u8(heatmap_float)
    cv2.imwrite(out_png, hm_u8)


def save_overlay_png(
        frame_bgr: np.ndarray,
        heatmap_float: np.ndarray,
        out_png: str,
        alpha: float = 0.4
):
    """
    Overlay grayscale heatmap on original frame and save PNG.
    Black = normal, White = anomaly.
    """
    h_img, w_img = frame_bgr.shape[:2]

    hm_u8 = _normalize_heatmap_to_u8(heatmap_float)
    hm_gray_bgr = cv2.cvtColor(hm_u8, cv2.COLOR_GRAY2BGR)

    if hm_gray_bgr.shape[:2] != (h_img, w_img):
        hm_gray_bgr = cv2.resize(hm_gray_bgr, (w_img, h_img), interpolation=cv2.INTER_LINEAR)

    overlay = cv2.addWeighted(frame_bgr, 1.0 - alpha, hm_gray_bgr, alpha, 0)
    cv2.imwrite(out_png, overlay)


def save_side_by_side_png(
        frame_bgr: np.ndarray,
        heatmap_float: np.ndarray,
        out_png: str
):
    """
    Save a side-by-side image:
    [ original frame | grayscale heatmap ]
    Black = normal, White = anomaly.
    """
    hm_u8 = _normalize_heatmap_to_u8(heatmap_float)
    hm_gray_bgr = cv2.cvtColor(hm_u8, cv2.COLOR_GRAY2BGR)

    h, w = frame_bgr.shape[:2]
    if hm_gray_bgr.shape[:2] != (h, w):
        hm_gray_bgr = cv2.resize(hm_gray_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    combined = np.hstack([frame_bgr, hm_gray_bgr])
    cv2.imwrite(out_png, combined)


def main():
    ap = argparse.ArgumentParser("Infer EfficientAD on sampled frames from a video")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--autoencoder", required=True)
    ap.add_argument("--specs", required=True)
    ap.add_argument("--video", required=True, help="Path to video file (mp4/avi/...)")
    ap.add_argument("--out_dir", default="output/video_infer")
    ap.add_argument("--save_overlay", action="store_true")
    ap.add_argument("--fps", type=float, default=1.0, help="Sampling FPS (default 1.0 = one frame per second)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    heatmap_dir = os.path.join(args.out_dir, "heatmap")
    overlay_dir = os.path.join(args.out_dir, "overlay")
    os.makedirs(heatmap_dir, exist_ok=True)
    if args.save_overlay:
        os.makedirs(overlay_dir, exist_ok=True)

    # Load models
    teacher = HailoInfer(args.teacher, input_type="UINT8", output_type="FLOAT32")
    student = HailoInfer(args.student, input_type="UINT8", output_type="FLOAT32")
    autoenc = HailoInfer(args.autoencoder, input_type="UINT8", output_type="FLOAT32")

    # Model input size
    h, w, _ = student.get_input_shape()
    size_hw = (h, w)

    # Load specs
    specs = load_specs_npz(args.specs)
    threshold = float(specs["threshold"])

    core = EfficientADCore(teacher, student, autoenc)

    # Video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    if not vid_fps or vid_fps <= 0:
        vid_fps = 30.0  # fallback if metadata missing

    # We sample one frame every N frames
    step = max(1, int(round(vid_fps / args.fps)))

    rows = []
    frame_idx = 0
    saved_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        t_sec = frame_idx / vid_fps

        # preprocess to model input
        img_u8 = preprocess_bgr_to_uint8_rgb(frame, size_hw)

        # predict
        map_combined, _, _ = core.predict(
            image=img_u8,
            teacher_mean=specs["teacher_mean"],
            teacher_std=specs["teacher_std"],
            q_st_start=specs["q_st_start"], q_st_end=specs["q_st_end"],
            q_ae_start=specs["q_ae_start"], q_ae_end=specs["q_ae_end"],
        )

        orig_h, orig_w = frame.shape[:2]
        score, heatmap = score_from_map_with_size(map_combined, orig_w, orig_h)
        pred = "anomaly" if score > threshold else "normal"

        # filenames: anomaly-img-000012.png or normal-img-000012.png
        name = f"{pred}-img-{saved_idx:06d}"

        # Save side-by-side
        sbs_png = os.path.join(heatmap_dir, f"{name}.png")
        save_side_by_side_png(frame, heatmap, sbs_png)

        # Optionally save overlay
        ov_png = ""
        if args.save_overlay:
            ov_png = os.path.join(overlay_dir, f"{name}_overlay.png")
            save_overlay_png(frame, heatmap, ov_png)

        rows.append({
            "time_sec": round(t_sec, 3),
            "frame_idx": int(frame_idx),
            "score": float(score),
            "threshold": float(threshold),
            "predict": pred,
            "side_by_side_png": sbs_png,
            "overlay_png": ov_png,
        })

        saved_idx += 1
        frame_idx += 1

    cap.release()

    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "results.csv"), index=False)

    teacher.close()
    student.close()
    autoenc.close()

    print("Done.")
    print("Results:", os.path.join(args.out_dir, "results.csv"))
    print("Outputs (side-by-side):", heatmap_dir)
    if args.save_overlay:
        print("Overlays:", overlay_dir)


if __name__ == "__main__":
    main()
