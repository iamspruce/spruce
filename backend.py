import asyncio
import base64
import os
import sys
import threading
import time
import logging

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

# --- 1. LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- 2. RVC PATH ---
sys.path.append("/workspace/Retrieval-based-Voice-Conversion-WebUI")

# --- 3. IMPORT RVC (just to verify it works, not wired in yet) ---
try:
    from infer.modules.vc.modules import VC
    from configs.config import Config
    logging.info("✅ RVC modules imported successfully")
except ImportError as e:
    logging.error(f"❌ Failed to import RVC: {e}")
    sys.exit(1)

# --- 4. FACEFUSION IMPORT ---
sys.path.append("/workspace/facefusion")
try:
    from facefusion import inference
    from facefusion import choices
    from facefusion import process
    logging.info("✅ FaceFusion imported successfully")
except Exception as e:
    logging.error(f"❌ Could not import FaceFusion: {e}")
    sys.exit(1)


# --- 5. IDLE AUTO-SHUTDOWN ---
class IdleShutdownManager:
    def __init__(self, timeout_minutes=15):
        self.timeout_seconds = timeout_minutes * 60
        self.last_active_time = time.time()
        self.shutdown_timer = None
        self.timer_lock = threading.Lock()
        logging.info(f"Idle shutdown set to {timeout_minutes} minutes")

    def record_activity(self):
        with self.timer_lock:
            self.last_active_time = time.time()
            if self.shutdown_timer:
                self.shutdown_timer.cancel()
            self.shutdown_timer = threading.Timer(self.timeout_seconds, self._shutdown)
            self.shutdown_timer.start()

    def _shutdown(self):
        logging.warning("⚠️ Server idle, shutting down pod")
        os.system("runpodctl remove pod $RUNPOD_POD_ID")


# --- 6. FACEFUSION WRAPPER ---
class FaceFusionWrapper:
    def __init__(self):
        self.source_face = None
        logging.info("FaceFusion wrapper initialized")

    def set_source_face(self, image_bytes: bytes) -> bool:
        try:
            arr = np.frombuffer(image_bytes, np.uint8)
            self.source_face = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if self.source_face is None:
                raise ValueError("Decoded source face is None")
            logging.info("✅ Source face loaded in memory")
            return True
        except Exception as e:
            logging.error(f"❌ Failed to load source face: {e}")
            return False

    def swap_frame(self, frame_bytes: bytes) -> bytes:
        try:
            arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None or self.source_face is None:
                return frame_bytes  # fallback

            # --- Run FaceFusion pipeline directly ---
            swapped = process.process_frame(self.source_face, frame)

            _, encoded = cv2.imencode(".jpg", swapped)
            return encoded.tobytes()
        except Exception as e:
            logging.error(f"❌ Face swap failed: {e}")
            return frame_bytes


# --- 7. FASTAPI APP ---
app = FastAPI()
shutdown_manager = IdleShutdownManager()
swapper = FaceFusionWrapper()


# --- 8. WEBSOCKET ENDPOINT ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.info("Frontend connected")
    shutdown_manager.record_activity()

    try:
        # Stage 1: Receive source face
        face_msg = await websocket.receive_json()
        face_bytes = base64.b64decode(face_msg["data"])
        if not swapper.set_source_face(face_bytes):
            await websocket.send_json({"status": "error", "message": "Could not save source face"})
            return

        # Stage 2: Receive target voice (skipped)
        await websocket.receive_json()

        # Stage 3: Ready
        await websocket.send_json({"status": "ready"})
        logging.info("System ready, entering realtime loop")

        # Stage 4: Processing loop
        while websocket.client_state == WebSocketState.CONNECTED:
            shutdown_manager.record_activity()
            message = await websocket.receive_json()

            video_frame_bytes = base64.b64decode(message["video"].split(",")[1])
            swapped_frame_bytes = swapper.swap_frame(video_frame_bytes)

            audio_chunk_b64 = message["audio"].split(",")[1]  # passthrough

            response_video_b64 = base64.b64encode(swapped_frame_bytes).decode("utf-8")
            await websocket.send_json(
                {
                    "video": f"data:image/jpeg;base64,{response_video_b64}",
                    "audio": f"data:audio/wav;base64,{audio_chunk_b64}",
                }
            )

    except WebSocketDisconnect:
        logging.info("Frontend disconnected")
    except Exception as e:
        logging.error(f"❌ WebSocket error: {e}", exc_info=True)
    finally:
        logging.info("Closing connection")
