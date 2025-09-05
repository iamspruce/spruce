#!/bin/bash

# This is the complete setup script for the AI Avatar project.
# It installs all necessary software, dependencies, and AI models.

echo "--- Starting Full AI Avatar Setup ---"
export PYTHONUNBUFFERED=1

# 1. Update package lists and install system dependencies
echo "--- Step 1: Installing system dependencies (ffmpeg, git-lfs) ---"
apt-get update
apt-get install -y ffmpeg git-lfs
git lfs install

# Move to the persistent workspace directory
cd /workspace
echo "--- Now in directory: $(pwd) ---"

# --- Part A: FaceFusion Setup ---
echo ""
echo "--- Step 2: Setting up FaceFusion for zero-shot video swapping ---"
git clone https://github.com/facefusion/facefusion.git
cd facefusion
echo "Installing FaceFusion Python dependencies..."
pip install -r requirements.txt
# Install the specific version of insightface needed for face analysis
pip install insightface==0.7.3 --no-cache-dir
cd /workspace # Return to the main workspace directory

# --- Part B: RVC Setup for Voice Conversion ---
echo ""
echo "--- Step 3: Setting up RVC for real-time voice conversion ---"
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git
cd Retrieval-based-Voice-Conversion-WebUI
echo "Installing RVC Python dependencies..."
pip install -r requirements.txt
echo "Downloading RVC's pre-trained base models... (This can take 10-15 minutes)"
apt install -y aria2
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt -d . -o hubert_base.pt
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D48k.pth -d ./pretrained_v2 -o D48k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G48k.pth -d ./pretrained_v2 -o G48k.pth
cd /workspace # Return to the main workspace directory

# --- Part C: Final Server Dependencies ---
echo ""
echo "--- Step 4: Installing FastAPI for our real-time backend server ---"
pip install fastapi uvicorn "websockets>=10.0" python-multipart aiofiles opencv-python

echo ""
echo "--- Setup Complete! ---"
echo "Your environment is ready. Both FaceFusion and RVC are installed in /workspace."
echo "You can now create your 'backend.py' file and start building the application logic."
```eof

4.  Press `Ctrl+O` (the letter O, not zero) to save the file, then press `Enter`.
5.  Press `Ctrl+X` to exit the `nano` editor.

#### **1.C: Execute the Setup Script**

Now, you just need to run the script you created.

1.  First, make the script executable by giving the system permission to run it:
    ```bash
    chmod +x runpod_setup.sh
    ```
2.  Next, execute the script:
    ```bash
    ./runpod_setup.sh
    ```

**This is the part that will take a while (20-30 minutes).** You will see a lot of text scrolling in your terminal as it downloads gigabytes of AI models and installs dozens of Python packages. **Do not close this terminal.**

#### **1.D: What to Expect & Next Steps**

You will know the script is finished when you see the final message:
`--- Setup Complete! ---`

Once you see this, your cloud server is fully prepared. It has all the complex AI software ready to go.

**Productivity Tip:** While this script is running, you can move on to the next steps in parallel! This is a great time to go to Vercel, create your new project, link it to a new GitHub repository, and get your serverless API functions set up.

When the script finishes, you will be perfectly positioned for **Step 2: Integrating the models with your `backend.py` script.**
