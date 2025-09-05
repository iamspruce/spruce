import asyncio
import base64
import os
import sys
import threading
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
import logging
import aiofiles
import subprocess # We will use this to call FaceFusion

# --- 1. SETUP LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 2. ADD RVC TO PYTHON PATH ---
# We only need RVC for direct import
sys.path.append('/workspace/Retrieval-based-Voice-Conversion-WebUI')

# --- 3. IMPORT RVC (FaceFusion will be called via command line) ---
try:
    # --- FIX: Corrected import paths ---
    # The RVC project's folders are 'infer' and 'configs' at the top level.
    from infer.modules.vc.modules import VC
    from configs.config import Config
    logging.info("Successfully imported RVC modules.")
except ImportError as e:
    logging.error(f"Failed to import RVC modules. Make sure the setup script ran correctly. Error: {e}")
    sys.exit(1)


# --- 4. IDLE AUTO-SHUTDOWN ---
class IdleShutdownManager:
    def __init__(self, timeout_minutes=15):
        self.timeout_seconds = timeout_minutes * 60
        self.last_active_time = time.time()
        self.shutdown_timer = None
        self.timer_lock = threading.Lock()
        logging.info(f"Idle shutdown manager initialized with a {timeout_minutes}-minute timeout.")

    def record_activity(self):
        with self.timer_lock:
            self.last_active_time = time.time()
            if self.shutdown_timer:
                self.shutdown_timer.cancel()
            self.shutdown_timer = threading.Timer(self.timeout_seconds, self._shutdown)
            self.shutdown_timer.start()

    def _shutdown(self):
        logging.warning("Server is idle. Initiating shutdown sequence.")
        os.system("runpodctl remove pod $RUNPOD_POD_ID")

# --- 5. AI WRAPPER CLASSES ---

class FaceFusionWrapper:
    def __init__(self):
        # Paths to the FaceFusion executable and our working files
        self.facefusion_path = "/workspace/facefusion/run.py"
        self.source_face_path = "/workspace/source_face.png"
        self.target_frame_path = "/workspace/target_frame.jpg"
        self.output_frame_path = "/workspace/output_frame.jpg"
        logging.info("FaceFusion Wrapper (Subprocess) initialized.")

    def set_source_face(self, image_bytes: bytes) -> bool:
        """Saves the source face image to a file."""
        try:
            with open(self.source_face_path, "wb") as f:
                f.write(image_bytes)
            logging.info(f"Source face saved to {self.source_face_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving source face: {e}")
            return False

    def swap_frame(self, target_frame_bytes: bytes) -> bytes:
        """
        Swaps face by calling FaceFusion's command-line interface.
        This is a robust, "black box" approach.
        """
        try:
            # 1. Save the incoming target frame to a temporary file
            with open(self.target_frame_path, "wb") as f:
                f.write(target_frame_bytes)

            # 2. Construct and run the command-line command
            command = [
                "python", self.facefusion_path,
                "-s", self.source_face_path,
                "-t", self.target_frame_path,
                "-o", self.output_frame_path,
                "--headless" # Important: runs without a GUI
            ]
            
            # This executes the command and waits for it to finish
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # 3. Read the resulting output image
            if os.path.exists(self.output_frame_path):
                with open(self.output_frame_path, "rb") as f:
                    result_bytes = f.read()
                return result_bytes
            else:
                logging.error("FaceFusion did not produce an output file.")
                return target_frame_bytes # Return original on error

        except subprocess.CalledProcessError as e:
            # This catches errors if the FaceFusion script itself fails
            logging.error(f"FaceFusion script failed with exit code {e.returncode}")
            logging.error(f"Stderr: {e.stderr.decode()}")
            return target_frame_bytes
        except Exception as e:
            logging.error(f"An error occurred in swap_frame: {e}")
            return target_frame_bytes


# --- RVC Wrapper (Placeholder - integration is complex) ---
# For the hackathon, we will focus on getting the video working first.
# Voice conversion adds significant latency and complexity.

# --- 6. INITIALIZE FASTAPI APP AND AI WRAPPERS ---
app = FastAPI()
shutdown_manager = IdleShutdownManager(timeout_minutes=15)
swapper = FaceFusionWrapper()

# --- 7. DEFINE THE WEBSOCKET ENDPOINT ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.info("Frontend connected.")
    shutdown_manager.record_activity()

    try:
        # -- Stage 1: Receive Source Face --
        logging.info("Waiting for source face...")
        face_msg = await websocket.receive_json()
        face_bytes = base64.b64decode(face_msg['data'])
        
        if not swapper.set_source_face(face_bytes):
            await websocket.send_json({"status": "error", "message": "Could not save source face."})
            return

        # -- Stage 2: Receive Target Voice (Skipped for now) --
        logging.info("Waiting for target voice (skipping for video-first demo)...")
        await websocket.receive_json() # Await message but do nothing with it

        # -- Stage 3: Signal Frontend to Go Live --
        await websocket.send_json({"status": "ready"})
        logging.info("FaceFusion is ready. Starting real-time video stream.")

        # -- Stage 4: Real-time Processing Loop --
        while websocket.client_state == WebSocketState.CONNECTED:
            shutdown_manager.record_activity()
            
            message = await websocket.receive_json()
            video_frame_bytes = base64.b64decode(message['video'].split(',')[1])
            
            # --- VIDEO PROCESSING ---
            swapped_frame_bytes = swapper.swap_frame(video_frame_bytes)

            # --- AUDIO PROCESSING (Passthrough) ---
            # We will just send the original audio back for now
            audio_chunk_b64 = message['audio'].split(',')[1]

            # Send results back to frontend
            response_video_b64 = base64.b64encode(swapped_frame_bytes).decode('utf-8')
            await websocket.send_json({
                "video": f"data:image/jpeg;base64,{response_video_b64}",
                "audio": f"data:audio/wav;base64,{audio_chunk_b64}" # Sending original audio back
            })

    except WebSocketDisconnect:
        logging.info("Frontend disconnected.")
    except Exception as e:
        logging.error(f"An error occurred in the websocket: {e}", exc_info=True)
    finally:
        logging.info("Closing websocket connection.")

