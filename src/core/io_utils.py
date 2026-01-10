import os
import numpy as np
from typing import Dict, Any

def save_specs_npz(path: str, specs: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **specs)

def load_specs_npz(path: str) -> Dict[str, Any]:
    f = np.load(path)
    return {k: f[k] for k in f.files}
