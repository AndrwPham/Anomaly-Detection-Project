import os
import argparse
import numpy as np

from src.common.infer_model import HailoInfer
from src.dataset.dataset import Dataset, DataLoader
from src.core.efficientad_core import EfficientADCore
from src.core.evaluation import AD_Evaluation
from src.core.io_utils import save_specs_npz
from src.core.scoring import score_from_map
from PIL import Image

def onchip_transform(
    img: np.ndarray,
    image_size: int = 256
) -> np.ndarray:
    img_pil = Image.fromarray(img)
    img_pil = img_pil.resize((image_size, image_size), resample=Image.BILINEAR)

    # img = np.array(img_pil).astype(np.float32) / 255.0
    # img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.array(img_pil).astype(np.uint8)

    return img


def main():
    ap = argparse.ArgumentParser(description="Calibrate EfficientAD specs exactly like efficientAD.py")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--autoencoder", required=True)
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--out_specs", default="specs.npz")
    ap.add_argument("--out_dir", default="output/calibration")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    teacher = HailoInfer(args.teacher, input_type="UINT8", output_type="FLOAT32")
    student = HailoInfer(args.student, input_type="UINT8", output_type="FLOAT32")
    autoenc = HailoInfer(args.autoencoder, input_type="UINT8", output_type="FLOAT32")

    h, w, _ = student.get_input_shape()
    size_hw = (h, w)

    train_loader = DataLoader(Dataset(args.train_dir), shuffle=False, transform=onchip_transform)
    test_loader  = DataLoader(Dataset(args.test_dir), shuffle=False, transform=onchip_transform)

    core = EfficientADCore(teacher, student, autoenc)

    # EXACT efficientAD.py computations:
    teacher_mean, teacher_std = core.teacher_normalization(teacher, train_loader)
    q_st_start, q_st_end, q_ae_start, q_ae_end = core.map_normalization(teacher_mean, teacher_std, train_loader)

    y_true, y_score = [], []

    for images, labels, paths in test_loader:
        label = labels[0]
        path  = paths[0]

        # your repo uses 'normal/anomaly' labels, map to 0/1
        y_true.append(0 if label == "normal" else 1)

        map_combined, _, _ = core.predict(
            image=images[0],
            teacher_mean=teacher_mean,
            teacher_std=teacher_std,
            q_st_start=q_st_start, q_st_end=q_st_end,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end
        )

        score, _ = score_from_map(map_combined, path)
        y_score.append(score)

    # Debug prints (optional but helpful)
    y_score_np = np.array(y_score, dtype=np.float32)
    print("y_score count:", y_score_np.size)
    print("y_score min/max/std:", float(np.min(y_score_np)), float(np.max(y_score_np)), float(np.std(y_score_np)))

    evaluator = AD_Evaluation(y_true=y_true, y_score=y_score, plot_dir=args.out_dir)
    metrics = evaluator.evaluate()
    threshold = metrics["Youden's J threshold"]

    specs = {
        "teacher_mean": teacher_mean,
        "teacher_std": teacher_std,
        "q_st_start": q_st_start,
        "q_st_end": q_st_end,
        "q_ae_start": q_ae_start,
        "q_ae_end": q_ae_end,
        "threshold": np.float32(threshold),
    }
    save_specs_npz(args.out_specs, specs)

    print("Saved specs to:", args.out_specs)
    print("Threshold:", threshold)
    print("AUROC:", metrics.get("AUROC"))

    teacher.close()
    student.close()
    autoenc.close()


if __name__ == "__main__":
    main()
