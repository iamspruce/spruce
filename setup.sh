#!/bin/bash
set -e

echo "--- 🚀 Starting Production Setup ---"
export PYTHONUNBUFFERED=1
export PIP_NO_CACHE_DIR=1

# --- System Dependencies ---
apt-get update
apt-get install -y ffmpeg git-lfs aria2 wget build-essential

# --- Python Env ---
cd /workspace
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools

# --- Cache Dirs ---
export HF_HOME='/workspace/cache/huggingface'
export TTS_HOME='/workspace/cache/tts'
export INSIGHTFACE_HOME='/workspace/cache/insightface'

# --- Core Dependencies ---
pip install fastapi uvicorn aiohttp websockets requests
pip install opencv-python-headless numpy==1.23.5 soundfile
pip install onnxruntime-gpu==1.17.1
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install aiortc av
pip install insightface==0.7.3
pip install TTS transformers accelerate optimum

# --- Download FaceSwap Model ---
aria2c -c -x 16 -s 16 -k 1M \
  https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx \
  -d /workspace/cache/insightface -o inswapper_128.onnx

# --- Pre-cache Models ---
cat <<EOF > cache_models.py
import torch
from transformers import pipeline
from TTS.api import TTS

print("Caching Whisper tiny.en...")
stt = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en", device=0 if torch.cuda.is_available() else -1)
print("✅ Whisper cached")

print("Caching TTS Tacotron2 + HiFiGAN...")
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=torch.cuda.is_available())
print("✅ TTS cached")
EOF

python cache_models.py

echo "--- ✅ Setup Complete ---"
