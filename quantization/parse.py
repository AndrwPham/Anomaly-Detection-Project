from hailo_sdk_client import ClientRunner
from typing import List, Optional, Dict
from loguru import logger
import os


def parse(onnx_path: str, model_name: str, net_input_shapes: Dict | List,
          output_dir: Optional[str] = None, hw_arch: str = 'hailo8', **kwargs) -> None:

    if output_dir is None:
        output_dir = os.curdir()
    os.makedirs(output_dir, exist_ok=True)

    assert os.path.isfile(onnx_path)

    har_path = os.path.join(output_dir, f'{model_name}.har')
    runner = ClientRunner(hw_arch=hw_arch)

    try:
        hn, npz = runner.translate_onnx_model(
            model=onnx_path,
            net_name=model_name,
            net_input_shapes=net_input_shapes,
            **kwargs
        )
    except Exception as e:
        logger.error(f"Error during model translation: {e}")
        print(e)

    runner.save_har(har_path)


if __name__ == "__main__":
    model_name = 'student'
    onnx_path = './output1/student.onnx'
    net_input_shapes = {'input': [1, 3, 256, 256]}

    parse(onnx_path, model_name, net_input_shapes, 'output1')
