from hailo_sdk_client import ClientRunner
import os
from typing import Optional

from loguru import logger


def compile(har_path: str, model_name: str,
            output_dir: Optional[str] = None, **kwargs) -> None:
    assert os.path.isfile(har_path)

    if output_dir is None:
        output_dir = os.curdir()
    os.makedirs(output_dir, exist_ok=True)

    runner = ClientRunner(har=har_path)

    if 'model_script_path' in kwargs.keys():
        model_script_path = kwargs['model_script_path']
        assert os.path.isfile(model_script_path)

        runner.load_model_script(model_script_path)

    try:
        hef = runner.compile()
    except Exception as e:
        logger.error(f'Failed to compile the model: {e}')
        raise

    file_name = os.path.join(output_dir, f"{model_name}.hef")
    with open(file_name, "wb") as f:
        f.write(hef)


if __name__ == "__main__":
    har_path = 'output_16bit/autoencoder_quantized.har'
    model_name = 'autoencoder'
    output_dir = 'output_16bit'
    compile(har_path, model_name, output_dir, model_script_path='./model_script.alls')
