#!/usr/bin/env python3

import os

from pathlib import Path
from optimum.onnxruntime import (
    AutoQuantizationConfig,
    ORTModelForSpeechSeq2Seq,
    ORTQuantizer,
    OptimizationConfig,
    ORTOptimizer,
)

model_id = "openai/whisper-small"
save_dir = "whisper-small"

model = ORTModelForSpeechSeq2Seq.from_pretrained(model_id, export=True)
model_dir = model.model_save_dir

optimizer = ORTOptimizer.from_pretrained(model)

# Define the optimization strategy by creating the appropriate configuration

optimization_config = OptimizationConfig(
    optimization_level=2,
    enable_transformers_specific_optimizations=True,
    optimize_for_gpu=False,
)
optimizer.optimize(save_dir=save_dir, optimization_config=optimization_config)

qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)

self_attn_layers = []
for i in range(0, 12):
    self_attn_layers.append(f'/model/decoder/layers.{i}/self_attn/Reshape_output_0')

qconfig.nodes_to_exclude = self_attn_layers + ['/Mul_1_output_0', '/Mul_3_output_0']

#m7i.2xlarge
#qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
#qconfig.nodes_to_exclude=['/conv1/Conv', '/conv2/Conv']

#m7g.2xlarge
#qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
#qconfig.nodes_to_exclude=['/conv1/Conv', '/conv2/Conv']

# model = ORTModelForSpeechSeq2Seq.from_pretrained(save_dir)

onnx_models = list(Path(save_dir).glob("*.onnx"))

# print(onnx_models)
quantizers = [
    ORTQuantizer.from_pretrained(save_dir, file_name=os.path.basename(onnx_model)) for onnx_model in onnx_models
]

for quantizer in quantizers:
    quantizer.quantize(save_dir=save_dir, quantization_config=qconfig)
