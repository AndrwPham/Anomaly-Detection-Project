
import numpy as np
from PIL import Image

def score_from_map(map_combined: np.ndarray, image_path: str):
    """
    Practical EfficientAD scoring for your dataset:

    - Score is computed on the true anomaly map (before padding), so pad=0 never dominates.
    - Heatmap output follows efficientAD.py visualization steps: squeeze -> pad(4,0) -> PIL resize.
    """
    with Image.open(image_path) as img:
        orig_w, orig_h = img.width, img.height

    m = np.squeeze(map_combined, axis=2).astype(np.float32)

    # Critical: score BEFORE padding
    score = float(np.max(m))

    # EfficientAD visualization path
    m_vis = np.pad(m, pad_width=(4, 4), constant_values=0.0)
    m_vis_pil = Image.fromarray(m_vis)
    m_vis_pil = m_vis_pil.resize((orig_w, orig_h), resample=Image.BILINEAR)
    m_resized = np.array(m_vis_pil).astype(np.float32)

    return score, m_resized

def score_from_map_with_size(map_combined: np.ndarray, orig_w: int, orig_h: int):
    m = np.squeeze(map_combined, axis=2).astype(np.float32)
    score = float(np.max(m))

    m_vis = np.pad(m, pad_width=(4, 4), constant_values=0.0)
    m_vis_pil = Image.fromarray(m_vis)
    m_vis_pil = m_vis_pil.resize((orig_w, orig_h), resample=Image.BILINEAR)
    heatmap = np.array(m_vis_pil).astype(np.float32)

    return score, heatmap
