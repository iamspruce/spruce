#!/usr/bin/env python3
"""
Production-ready WebRTC avatar backend (FastAPI + aiortc).
(Video + FaceSwap only, no voice models)

Optimized for low latency and responsiveness.
"""

import os
import sys
import asyncio
import base64
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer,
    VideoStreamTrack,
)
from av import VideoFrame

from insightface.app import FaceAnalysis
import insightface.model_zoo

# ---------- Ensure cache path matches setup ----------
CACHE_HOME = "/workspace/cache"
os.environ["INSIGHTFACE_HOME"] = os.path.join(CACHE_HOME, "insightface")
INSIGHTFACE_HOME = os.environ["INSIGHTFACE_HOME"]
INSWAPPER_PATH = "inswapper_128.onnx"
INSWAPPER_FULLPATH = os.path.join(INSIGHTFACE_HOME, "models", INSWAPPER_PATH)

# ---------- Config ----------
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))
THREADPOOL_WORKERS = int(os.getenv("THREADPOOL_WORKERS", "4")) # Optimized for CPU-bound tasks
FRAME_PROCESS_INTERVAL = float(os.getenv("FRAME_PROCESS_INTERVAL", "0.066")) # Process a frame every ~66ms (~15 FPS)

ICE_SERVERS = [
    RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
    # Add TURN servers if needed, but they are costly. For development, STUN is often enough.
]
RTC_CONFIG = RTCConfiguration(iceServers=ICE_SERVERS)


# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("avatar-backend")

# ---------- Threadpool for blocking ML operations ----------
executor = ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS)

# ---------- FastAPI app ----------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

pcs = set()  # keep alive peer connections

# ---------- Model wrappers ----------
class FaceSwapService:
    def __init__(self):
        logger.info("Loading insightface models...")
        # Determine execution provider based on GPU availability
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        ctx_id = 0 if 'CUDAExecutionProvider' in providers else -1

        try:
            self.analyzer = FaceAnalysis(name="buffalo_s", root=INSIGHTFACE_HOME, providers=providers)
            self.analyzer.prepare(ctx_id=ctx_id, det_size=(640, 640))
            logger.info(f"FaceAnalysis ready (providers={providers})")
        except Exception:
            logger.exception("Failed to initialize FaceAnalysis.")
            self.analyzer = None

        try:
            self.inswapper = insightface.model_zoo.get_model(INSWAPPER_PATH, download=True, root=INSIGHTFACE_HOME, providers=providers)
            logger.info("Inswapper ready")
        except Exception:
            logger.exception("Inswapper load failed.")
            self.inswapper = None

        self.source_face = None

    def set_source_face(self, image_bytes: bytes) -> bool:
        try:
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                logger.error("Failed to decode source face image.")
                return False

            if self.analyzer:
                faces = self.analyzer.get(img)
                if faces:
                    # Sort by face area and pick the largest
                    self.source_face = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))[-1]
                    logger.info("Source face set successfully from analyzer.")
                    return True
            logger.warning("Analyzer not available or no face found in source image.")
            return False
        except Exception:
            logger.exception("Error setting source face.")
            return False

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Combines detection and swapping into one blocking function."""
        if not self.analyzer or not self.inswapper or self.source_face is None:
            return frame_bgr # Return original frame if models aren't ready

        try:
            # Resize for faster detection (optional but recommended)
            # scale = 640 / max(frame_bgr.shape[0], frame_bgr.shape[1])
            # if scale < 1.0:
            #     resized_frame = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            # else:
            #     resized_frame = frame_bgr
            # faces = self.analyzer.get(resized_frame)
            
            # For simplicity, we process the original frame. Uncomment above for optimization.
            faces = self.analyzer.get(frame_bgr)
            
            if faces:
                # if scale < 1.0: # Scale bbox back to original frame size
                #     for face in faces:
                #         face.bbox = face.bbox / scale
                
                # Swap the largest detected face
                target_face = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))[-1]
                return self.inswapper.get(frame_bgr, target_face, self.source_face, paste_back=True)
        except Exception:
            logger.exception("Error during frame processing.")
        
        return frame_bgr # Return original on failure

# ---------- Real-time Processing Logic ----------
class VideoProcessor:
    def __init__(self, face_service: FaceSwapService):
        self.face_service = face_service
        self.in_queue = asyncio.Queue(maxsize=1)
        self.out_queue = asyncio.Queue(maxsize=1)
        self.last_processed_frame: Optional[np.ndarray] = None
        self.processing_task = asyncio.create_task(self.processing_loop())

    async def processing_loop(self):
        """The core loop that processes frames from the queue."""
        loop = asyncio.get_running_loop()
        while True:
            frame_to_process = await self.in_queue.get()
            
            processed_frame = await loop.run_in_executor(
                executor, self.face_service.process_frame, frame_to_process
            )
            
            # Clear the output queue and put the new frame
            if not self.out_queue.empty():
                try: self.out_queue.get_nowait()
                except asyncio.QueueEmpty: pass
            await self.out_queue.put(processed_frame)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the most recently processed frame without blocking."""
        if not self.out_queue.empty():
            self.last_processed_frame = self.out_queue.get_nowait()
        return self.last_processed_frame
    
    def submit_frame(self, frame: np.ndarray):
        """Submit a new frame for processing without blocking."""
        if self.in_queue.empty():
            self.in_queue.put_nowait(frame)
        # If queue is full, just drop the frame to avoid latency buildup

    def stop(self):
        self.processing_task.cancel()


# ---------- Tracks ----------
class ProcessedVideoStreamTrack(VideoStreamTrack):
    def __init__(self, incoming_track, video_processor: VideoProcessor):
        super().__init__()
        self.track = incoming_track
        self.processor = video_processor
        self.last_process_time = 0

    async def recv(self):
        frame = await self.track.recv()
        current_time = frame.pts / frame.time_base

        # Submit frame for processing based on a time interval to control FPS
        if current_time - self.last_process_time > FRAME_PROCESS_INTERVAL:
            self.last_process_time = current_time
            img = frame.to_ndarray(format="bgr24")
            self.processor.submit_frame(img)

        # Get the latest available processed frame
        processed_img = self.processor.get_latest_frame()

        if processed_img is not None:
            # Create a new frame from the processed image
            new_frame = VideoFrame.from_ndarray(processed_img, format="bgr24")
        else:
            # If no processed frame is ready, send the original to avoid freezing
            new_frame = frame

        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

# ---------- App endpoints ----------
face_service = FaceSwapService()

@app.post("/prepare")
async def prepare(req: Request):
    data = await req.json()
    face_b64 = data.get("source_face")
    if not face_b64:
        raise HTTPException(status_code=400, detail="source_face required")
    try:
        face_bytes = base64.b64decode(face_b64.split(",")[1] if "," in face_b64 else face_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid base64 face")
    if not face_service.set_source_face(face_bytes):
        raise HTTPException(status_code=500, detail="failed to set source face, check image quality")
    return {"status": "success"}

@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection(configuration=RTC_CONFIG)
    pcs.add(pc)
    logger.info("Created PeerConnection %s", id(pc))

    # Create a dedicated processor for this connection
    video_processor = VideoProcessor(face_service)

    @pc.on("track")
    def on_track(track):
        logger.info("Track %s received", track.kind)
        if track.kind == "video":
            processed_track = ProcessedVideoStreamTrack(track, video_processor)
            pc.addTrack(processed_track)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
            video_processor.stop()

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutting down, closing peer connections")
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros, return_exceptions=True)
    pcs.clear()

if __name__ == "__main__":
    import uvicorn
    # Use the name of your file here, e.g., if your file is main.py, use "main:app"
    uvicorn.run("backend:app", host=HOST, port=PORT, log_level="info", reload=True)
