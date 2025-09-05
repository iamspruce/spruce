#!/bin/bash
set -e

echo "--- 🚀 Starting Final Zero-Shot WebRTC Avatar Setup ---"
export PYTHONUNBUFFERED=1
export PIP_NO_CACHE_DIR=1

# --- Part 1: System Dependencies ---
apt-get update
apt-get install -y ffmpeg git-lfs aria2 software-properties-common wget
add-apt-repository ppa:deadsnakes/ppa -y
apt-get update
apt-get install -y python3.10 python3.10-venv
cd /workspace
echo "--- Now in directory: $(pwd) ---"

# --- Part 2: Create Virtual Environment & Install Libraries ---
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip

export HF_HOME='/workspace/cache/huggingface'
export TTS_HOME='/workspace/cache/tts'
export INSIGHTFACE_HOME='/workspace/cache/insightface'

pip install uvicorn fastapi "websockets>=10.0" requests aiohttp
pip install opencv-python numpy==1.23.5 soundfile
pip install onnxruntime-gpu==1.17.1
pip install insightface==0.7.3
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
# Install WebRTC, TTS, and Transformers
pip install aiortc
pip install TTS
pip install transformers accelerate optimum

# --- Part 3: Download Face Swap Model ---
aria2c -c -x 16 -s 16 -k 1M https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx -d . -o inswapper_128.onnx

# --- Part 4: Pre-cache Voice Models ---
cat <<EOF > cache_models.py
import torch
from TTS.api import TTS
from transformers import pipeline

print("Caching Speech-to-Text model (Distil-Whisper)...")
try:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    stt_pipe = pipeline(
        "automatic-speech-recognition", model="distil-whisper/distil-medium.en",
        torch_dtype=torch.float16, device=device
    )
    print("STT Model cached.")
except Exception as e:
    print(f"Failed to cache STT model: {e}")

print("Caching Text-to-Speech model (Coqui XTTS)...")
try:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
    print("TTS Model cached.")
except Exception as e:
    print(f"Failed to cache TTS model: {e}")

print("Model caching complete.")
EOF

python cache_models.py

echo "--- ✅ Setup Complete! ---"
echo "--- To activate the environment, run: source /workspace/venv/bin/activate ---"
