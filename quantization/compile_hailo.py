from convert_onnx import convert
from parse import parse
from optimize import optimize
from compile import compile

import os
import argparse
import torch


def main():
    # ===============================
    # Argument parsing
    # ===============================
    parser = argparse.ArgumentParser(
        description="Convert and compile a PyTorch checkpoint into a Hailo-compatible file."
    )

    parser.add_argument("-cp", "--checkpoint-path", required=True, type=str,
                        help="Path to the PyTorch checkpoint file")  # [ADDED]
    parser.add_argument("-o", "--output-dir", required=True, type=str,
                        help="Directory where all outputs will be written")  # [ADDED]
    parser.add_argument("-n", "--model-name", required=True, type=str,
                        help="Base name of the model")  # [ADDED]
    parser.add_argument("--model-script-path", required=True, type=str,
                        help="Path to Hailo model script (.alls)")  # [ADDED]
    parser.add_argument("--calib-dataset-path", required=True, type=str,
                        help="Path to calibration dataset")  # [ADDED]

    args = parser.parse_args()  # [ADDED]

    # ===============================
    # Configuration
    # ===============================
    INPUT_NAMES = ["input"]
    OUTPUT_NAMES = ["output"]
    INPUT_SHAPE = (1, 3, 256, 256)

    checkpoint_path = os.path.abspath(args.checkpoint_path)  # [CHANGED]
    output_dir = os.path.abspath(args.output_dir)            # [CHANGED]
    model_name = args.model_name
    model_script_path = os.path.abspath(args.model_script_path)  # [CHANGED]
    calib_dataset_path = os.path.abspath(args.calib_dataset_path)  # [CHANGED]

    # ===============================
    # Path validation
    # ===============================
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path '{checkpoint_path}' not found.")

    if not os.path.isfile(model_script_path):
        raise FileNotFoundError(f"Model script path '{model_script_path}' not found.")

    if not os.path.isfile(calib_dataset_path):
        raise FileNotFoundError(f"Calibration dataset path '{calib_dataset_path}' not found.")

    os.makedirs(output_dir, exist_ok=True)  # [ADDED]

    # ===============================
    # Load model
    # ===============================
    model = torch.load(
        checkpoint_path,
        weights_only=False,
        map_location="cpu"
    )

    # ===============================
    # Convert to ONNX
    # ===============================
    dummy_input = torch.randn(INPUT_SHAPE, dtype=torch.float)  # [ADDED]

    convert(
        model,
        model_name,
        dummy_input,
        output_dir,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES
    )

    # ===============================
    # Parse ONNX
    # ===============================
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
    net_input_shapes = {"input": INPUT_SHAPE}

    parse(
        onnx_path,
        model_name,
        net_input_shapes,
        output_dir
    )

    # ===============================
    # Optimize
    # ===============================
    har_path = os.path.join(output_dir, f"{model_name}.har")

    optimize(
        har_path,
        model_name,
        calib_dataset_path,
        output_dir,
        model_script_path=model_script_path
    )

    # ===============================
    # Compile
    # ===============================
    quantized_har_path = os.path.join(
        output_dir, f"{model_name}_quantized.har"
    )

    compile(
        quantized_har_path,
        model_name,
        output_dir,
        model_script_path=model_script_path
    )


if __name__ == "__main__":
    main()  # [ADDED]
