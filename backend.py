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
import cv2
import numpy as np
import torch

# --- 1. SETUP LOGGING ---
# This helps us see what's happening in the RunPod logs.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 2. ADD AI LIBRARIES TO PYTHON PATH ---
# This allows our script to find and import the code from FaceFusion and RVC.
sys.path.append('/workspace/facefusion')
sys.path.append('/workspace/Retrieval-based-Voice-Conversion-WebUI')

# --- 3. IMPORT CORE AI FUNCTIONALITY ---
# We wrap these imports in a try-except block to give helpful errors if they fail.
try:
    # FaceFusion core function for processing frames.
    from facefusion.core import process_image, process_frame, get_face_reference, set_face_reference
    from facefusion.utilities import resolve_relative_path
    
    # RVC core classes and functions for voice conversion.
    from RVC.infer.modules.vc.modules import VC
    from RVC.configs.config import Config
    logging.info("Successfully imported FaceFusion and RVC modules.")
except ImportError as e:
    logging.error(f"Failed to import AI modules. Make sure the setup script ran correctly. Error: {e}")
    # Exit if we can't import the core tools.
    sys.exit(1)


# --- 4. IDLE AUTO-SHUTDOWN ---
# This class will automatically stop the pod if it's idle for too long.
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
        # The 'runpodctl' command is a tool within RunPod to manage the pod.
        os.system("runpodctl remove pod $RUNPOD_POD_ID")

# --- 5. AI WRAPPER CLASSES ---
# These classes make it easier to interact with the AI models.

class FaceFusionWrapper:
    def __init__(self):
        self.source_face = None
        logging.info("FaceFusion Wrapper initialized.")

    def set_source_face(self, image_path: str):
        """Analyzes and sets the source face from an image file."""
        try:
            # This is a key FaceFusion function to get the face data.
            face_ref = get_face_reference(image_path)
            if face_ref:
                set_face_reference(image_path, face_ref)
                self.source_face = image_path
                logging.info(f"Successfully set source face from {image_path}")
                return True
            else:
                logging.error(f"No face found in the source image: {image_path}")
                return False
        except Exception as e:
            logging.error(f"Error setting source face: {e}")
            return False

    def swap_frame(self, target_frame_bytes: bytes) -> bytes:
        """Swaps the face in a single video frame."""
        if not self.source_face:
            logging.warning("Source face not set. Returning original frame.")
            return target_frame_bytes
        try:
            # Convert bytes to a NumPy array for image processing
            np_arr = np.frombuffer(target_frame_bytes, np.uint8)
            target_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # The core FaceFusion real-time processing function
            result_frame = process_frame(
                source_path=self.source_face,
                target_frame=target_frame
            )
            
            # Encode the result back to JPEG bytes to send over the network
            _, buffer = cv2.imencode('.jpg', result_frame)
            return buffer.tobytes()
        except Exception as e:
            logging.error(f"Error swapping frame: {e}")
            return target_frame_bytes # Return original on error

class RVCWrapper:
    def __init__(self):
        # Setup RVC configuration
        self.config = Config()
        self.vc = VC(self.config)
        self.target_voice_path = None
        logging.info("RVC Wrapper initialized.")

    def setup_voice(self, voice_model_path: str):
        """Loads the target voice model for conversion."""
        try:
            # TODO: Find the exact function in RVC to pre-load a model.
            # This step is crucial for low-latency conversion.
            self.target_voice_path = voice_model_path
            logging.info(f"RVC target voice model set to: {voice_model_path}")
            return True
        except Exception as e:
            logging.error(f"Error setting up RVC voice: {e}")
            return False

    def convert_audio(self, input_audio_path: str) -> bytes:
        """Converts an audio chunk using the target voice."""
        if not self.target_voice_path:
            logging.warning("RVC target voice not set. Returning original audio.")
            async def read_original():
                async with aiofiles.open(input_audio_path, "rb") as f:
                    return await f.read()
            return asyncio.run(read_original())
        try:
            # TODO: This is a conceptual call to the RVC inference.
            # You will need to find the specific function in the RVC library
            # that takes an input audio path and returns the converted audio bytes.
            # Example placeholder:
            # converted_audio_bytes = self.vc.vc_inference(
            #     model_path=self.target_voice_path,
            #     input_audio_path=input_audio_path
            # )
            # For the hackathon, we'll just pass through the original audio for now.
            async def read_original():
                async with aiofiles.open(input_audio_path, "rb") as f:
                    return await f.read()
            return asyncio.run(read_original())
        
        except Exception as e:
            logging.error(f"Error converting audio: {e}")
            # Fallback to original audio on error
            async def read_original():
                async with aiofiles.open(input_audio_path, "rb") as f:
                    return await f.read()
            return asyncio.run(read_original())

# --- 6. INITIALIZE FASTAPI APP AND AI WRAPPERS ---
app = FastAPI()
shutdown_manager = IdleShutdownManager(timeout_minutes=15)
swapper = FaceFusionWrapper()
# voice_converter = RVCWrapper() # TODO: Enable this once RVC integration is complete

# --- 7. DEFINE THE WEBSOCKET ENDPOINT ---
# This is the main entry point for the frontend connection.
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.info("Frontend connected.")
    shutdown_manager.record_activity() # Record activity to prevent shutdown

    try:
        # -- Stage 1: Receive Source Face --
        logging.info("Waiting for source face...")
        face_msg = await websocket.receive_json()
        face_bytes = base64.b64decode(face_msg['data'])
        source_face_path = "/workspace/source_face.png"
        async with aiofiles.open(source_face_path, "wb") as f:
            await f.write(face_bytes)
        
        if not swapper.set_source_face(source_face_path):
            await websocket.send_json({"status": "error", "message": "No face found in image."})
            return

        # -- Stage 2: Receive Target Voice (Placeholder) --
        logging.info("Waiting for target voice...")
        voice_msg = await websocket.receive_json()
        # TODO: Process the voice file and set up the RVC model.

        # -- Stage 3: Signal Frontend to Go Live --
        await websocket.send_json({"status": "ready"})
        logging.info("AI models are ready. Starting real-time stream.")

        # -- Stage 4: Real-time Processing Loop --
        while websocket.client_state == WebSocketState.CONNECTED:
            shutdown_manager.record_activity() # Keep pod alive
            
            message = await websocket.receive_json()
            video_frame_bytes = base64.b64decode(message['video'].split(',')[1])
            # audio_chunk_bytes = base64.b64decode(message['audio'].split(',')[1])

            # Process video
            swapped_frame_bytes = swapper.swap_frame(video_frame_bytes)

            # Process audio (TODO)
            # converted_audio_bytes = voice_converter.convert_audio(...)

            # Send results back to frontend
            response_video_b64 = base64.b64encode(swapped_frame_bytes).decode('utf-8')
            await websocket.send_json({
                "video": f"data:image/jpeg;base64,{response_video_b64}",
                # "audio": ...
            })

    except WebSocketDisconnect:
        logging.info("Frontend disconnected.")
    except Exception as e:
        logging.error(f"An error occurred in the websocket: {e}")
    finally:
        logging.info("Closing websocket connection.")

