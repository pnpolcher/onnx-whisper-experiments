#!/usr/bin/env python3

from datetime import datetime
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import pipeline, AutoTokenizer, AutoFeatureExtractor

# Number of inferences for comparing timings
num_inferences = 10
save_dir = "whisper-small"
inference_file = "test2.wav"

# Create pipeline based on quantized ONNX model
model = ORTModelForSpeechSeq2Seq.from_pretrained(save_dir)
tokenizer = AutoTokenizer.from_pretrained(save_dir)
feature_extractor = AutoFeatureExtractor.from_pretrained(save_dir)
cls_pipeline_onnx = pipeline("automatic-speech-recognition", model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)

# Create pipeline with original model as baseline
cls_pipeline_original = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# Measure inference of quantized model
start_quantized = datetime.now()
for i in range(num_inferences):
    x = cls_pipeline_onnx(inference_file)
    print(x)
end_quantized = datetime.now()

# Measure inference of original model
start_original = datetime.now()
for i in range(num_inferences):
    cls_pipeline_original(inference_file)
end_original = datetime.now()

original_inference_time = (end_original - start_original).total_seconds() / num_inferences
print(f"Original inference time: {original_inference_time}")

quantized_inference_time = (end_quantized - start_quantized).total_seconds() / num_inferences
print(f"Quantized inference time: {quantized_inference_time}")