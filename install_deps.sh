#!/bin/bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -U optimum[exporters,onnxruntime] transformers
sudo apt install -y ffmpeg

wget https://huggingface.co/datasets/Narsil/candle-examples/resolve/main/samples_jfk.wav
