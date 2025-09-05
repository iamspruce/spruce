#!/bin/bash
set -e

echo "--- Starting Full AI Avatar Setup ---"
export PYTHONUNBUFFERED=1

# 1. Update system packages
echo "--- Step 1: Installing system dependencies ---"
apt-get update
apt-get install -y ffmpeg git-lfs aria2
git lfs install

cd /workspace
echo "--- Now in directory: $(pwd) ---"

# --- Part A: FaceFusion Setup ---
echo ""
echo "--- Step 2: Setting up FaceFusion ---"
git clone https://github.com/facefusion/facefusion.git
cd facefusion
pip install --upgrade pip
pip install -r requirements.txt
# lock insightface version
pip install insightface==0.7.3 onnxruntime==1.16.3
cd /workspace

# --- Part B: RVC Setup ---
echo ""
echo "--- Step 3: Setting up RVC ---"
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git
cd Retrieval-based-Voice-Conversion-WebUI
pip install -r requirements.txt

echo "Downloading pretrained RVC models..."
mkdir -p pretrained_v2
aria2c -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt -d . -o hubert_base.pt
aria2c -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D48k.pth -d pretrained_v2 -o D48k.pth
aria2c -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G48k.pth -d pretrained_v2 -o G48k.pth
cd /workspace

# --- Part C: Backend Server ---
echo ""
echo "--- Step 4: Installing FastAPI backend dependencies ---"
pip install fastapi uvicorn "websockets>=10.0" python-multipart aiofiles opencv-python pydantic==1.10.14

echo ""
echo "--- Setup Complete! ---"
