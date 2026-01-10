import torch
import os
import onnx
import numpy as np
import onnxruntime as ort
from typing import Optional
from onnxsim import simplify

def convert(model, model_name: str, dummy_input: torch.tensor,
            output_dir: Optional[str] = None, **kwargs) -> None:
    model = model.float()
    model.eval()

    if output_dir is None:
        output_dir = os.curdir()
    os.makedirs(output_dir, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        os.path.join(output_dir, f"{model_name}_temp.onnx"),
        export_params=True,
        opset_version=17,
        **kwargs
    )
    print("ONNX model exported successfully!")

    simplified_model = simplify_onnx(os.path.join(output_dir, f"{model_name}_temp.onnx"))
    onnx.save(simplified_model, os.path.join(output_dir, f"{model_name}.onnx"))

    os.remove(os.path.join(output_dir, f"{model_name}_temp.onnx"))
    # test_onnx(os.path.join(output_dir, os.path.join(output_dir, f"{model_name}.onnx")), dummy_input.numpy())

def simplify_onnx(model_path):
    print("Simplifying ONNX model...")
    model = onnx.load(model_path)
    model_sim, check = simplify(model)

    assert check, "Simplified ONNX model could not be validated"

    return model_sim

# def test_onnx(onnx_path: str, dummy_input: np.ndarray):
#     assert os.path.isfile(onnx_path)
#
#     onnx_model = onnx.load(onnx_path)
#     onnx.checker.check_model(onnx_model)
#     print("ONNX model is valid!")

if __name__ == '__main__':
    checkpoint_path = './models/autoencoder_final.pth'
    output_dir = 'output1'
    input_names=['input']
    output_names=['output']
    model = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    convert(model, 'autoencoder', torch.randn(1, 3, 256, 256, dtype=torch.float),
            output_dir, input_names=input_names, output_names=output_names)
