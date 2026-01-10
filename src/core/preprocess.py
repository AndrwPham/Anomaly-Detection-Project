import numpy as np
from PIL import Image
from typing import Tuple

def preprocess_uint8(path: str, size_hw: Tuple[int, int]) -> np.ndarray:
    """
    EXACT EfficientAD-style preprocessing (closest practical match):
      - PIL open
      - convert to RGB
      - PIL resize bilinear to model input size
      - return RGB uint8 HWC
    """
    h, w = size_hw
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = im.resize((w, h), resample=Image.BILINEAR)
        return np.array(im, dtype=np.uint8)
