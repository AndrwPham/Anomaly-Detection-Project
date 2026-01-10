import numpy as np
from hailo_platform import HEF, VDevice, HailoSchedulingAlgorithm, FormatType
from typing import List, Optional

class HailoInfer:
    def __init__(self, hef_path: str, input_type: str = 'UINT8', output_type: str = 'FLOAT32'):
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.group_id = "SHARED"
        self.vdevice = VDevice(params)

        self.hef = HEF(hef_path)
        self.infer_model = self.vdevice.create_infer_model(hef_path)
        self.infer_model.set_batch_size(1)

        # Set input type (usually UINT8 for images)
        if input_type:
            self.infer_model.input().set_format_type(getattr(FormatType, input_type))

        # Set output type (usually FLOAT32 for math)
        if output_type:
            for output in self.infer_model.outputs:
                self.infer_model.output(output.name).set_format_type(getattr(FormatType, output_type))

        self.config_ctx = self.infer_model.configure()
        self.configured_model = self.config_ctx.__enter__()
        self.bindings = self.configured_model.create_bindings()

    def get_input_shape(self):
        return self.hef.get_input_vstream_infos()[0].shape

    def run_sync(self, input_data):
        # Set Input Buffer
        self.bindings.input().set_buffer(np.array(input_data))

        # Prepare Output Buffers
        output_buffers = {
            name: np.empty(self.infer_model.output(name).shape, dtype=np.float32)
            for name in self.infer_model.output_names
        }
        for name, buf in output_buffers.items():
            self.bindings.output(name).set_buffer(buf)

        # Run Inference
        self.configured_model.run([self.bindings], 1000)

        # Return result (assuming single output or handling specific logic outside)
        # For EfficientAD, we usually just need the buffer.
        # If multiple outputs, returns a dict or the first one.
        if len(output_buffers) == 1:
            return list(output_buffers.values())[0]
        return output_buffers # Returns dict if multiple outputs

    def close(self):
        self.config_ctx.__exit__(None, None, None)