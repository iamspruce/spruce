#!/usr/bin/env python3
"""
A streamlined, production-ready WebRTC backend for real-time face swapping.
"""

import os
import asyncio
import base64
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor

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
from aiortc.rtp import PictureLossIndication
from av import VideoFrame

# AI model imports for face swapping
from insightface.app import FaceAnalysis
import insightface.model_zoo

# ---------- Config ----------
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))

INSIGHTFACE_HOME = os.getenv("INSIGHTFACE_HOME", "/workspace/cache/insightface")
INSWAPPER_PATH = os.getenv("INSWAPPER_PATH", "/workspace/inswapper_128.onnx")

# --- IMPORTANT: Use environment variables for credentials in production ---
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
    raise ValueError("TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN environment variables must be set")

ICE_SERVERS = [
    RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
    RTCIceServer(
        urls=["turn:global.turn.twilio.com:3478?transport=udp"],
        username=TWILIO_ACCOUNT_SID,
        credential=TWILIO_AUTH_TOKEN,
    ),
    RTCIceServer(
        urls=["turn:global.turn.twilio.com:3478?transport=tcp"],
        username=TWILIO_ACCOUNT_SID,
        credential=TWILIO_AUTH_TOKEN,
    ),
]
RTC_CONFIG = RTCConfiguration(iceServers=ICE_SERVERS)
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "5"))

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("avatar-backend")

# ---------- Threadpool for blocking operations ----------
executor = ThreadPoolExecutor(max_workers=int(os.getenv("THREADPOOL_WORKERS", "6")))

# ---------- FastAPI app ----------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

pcs = set()  # keep alive peer connections

# ---------- Model wrappers ----------
class FaceSwapService:
    def __init__(self):
        logger.info("Loading insightface FaceAnalysis...")
        try:
            # Forcing to CPU to save VRAM if needed, otherwise use GPU
            # ctx_id = -1 # Force CPU
            ctx_id = 0 if torch.cuda.is_available() else -1
            self.analyzer = FaceAnalysis(name="buffalo_s", root=INSIGHTFACE_HOME)
            self.analyzer.prepare(ctx_id=ctx_id, det_size=(640, 640))
            logger.info(f"FaceAnalysis ready (ctx={ctx_id})")
        except Exception:
            logger.exception("Failed to initialize FaceAnalysis.")
            self.analyzer = None

        try:
            self.inswapper = insightface.model_zoo.get_model(INSWAPPER_PATH, download=False, root=INSIGHTFACE_HOME)
            logger.info("Inswapper ready")
        except Exception:
            logger.exception("Inswapper load failed; falling back to naive paste")
            self.inswapper = None

        self.source_face = None

    def set_source_face(self, image_bytes: bytes) -> bool:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None or not self.analyzer:
            return False
        try:
            faces = self.analyzer.get(img)
            if faces:
                self.source_face = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))[-1]
                logger.info("Source face saved (insightface descriptor)")
                return True
        except Exception:
            logger.exception("Analyzer failed on source image")
        return False

    def detect(self, frame_bgr: np.ndarray):
        if not self.analyzer:
            return []
        try:
            return self.analyzer.get(frame_bgr)
        except Exception:
            logger.exception("Face detection error")
            return []

    def swap(self, frame_bgr: np.ndarray, dets):
        if self.inswapper and self.source_face and dets:
            try:
                return self.inswapper.get(frame_bgr, dets[0], self.source_face, paste_back=True)
            except Exception:
                logger.exception("Inswapper error")
        return frame_bgr

# ---------- Tracks ----------
class ProcessedVideoStreamTrack(VideoStreamTrack):
    def __init__(self, incoming_track, face_service: FaceSwapService):
        super().__init__()
        self.track = incoming_track
        self.face_service = face_service
        self.frame_index = 0
        self.cached_dets = []

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")
        self.frame_index += 1

        if (self.frame_index % FRAME_SKIP) == 0:
            loop = asyncio.get_running_loop()
            try:
                dets = await loop.run_in_executor(executor, self.face_service.detect, img.copy())
                self.cached_dets = dets
            except Exception:
                logger.exception("Detection executor failed")
                self.cached_dets = []

        loop = asyncio.get_running_loop()
        processed = await loop.run_in_executor(executor, self.face_service.swap, img.copy(), self.cached_dets)

        new_frame = VideoFrame.from_ndarray(processed, format="bgr24")
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
        face_bytes = base64.b64decode(face_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid base64 for face")
    if not face_service.set_source_face(face_bytes):
        raise HTTPException(status_code=400, detail="failed to set source face")
    return {"status": "success"}

@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection(configuration=RTC_CONFIG)
    pcs.add(pc)
    logger.info(f"Created PeerConnection {id(pc)}")

    @pc.on("track")
    def on_track(track):
        logger.info(f"Track {track.kind} received")
        if track.kind == "video":
            processed_track = ProcessedVideoStreamTrack(track, face_service)
            pc.addTrack(processed_track)

            async def force_keyframe():
                await asyncio.sleep(0.5)
                video_sender = next((s for s in pc.getSenders() if s.track and s.track.kind == "video"), None)
                if video_sender:
                    logger.info("Sending PLI to request a keyframe")
                    try:
                        await video_sender.transport.rtcp.send([PictureLossIndication(media_ssrc=track.ssrc)])
                    except Exception as e:
                        logger.error(f"Failed to send PLI request: {e}")
            
            asyncio.ensure_future(force_keyframe())

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state is {pc.connectionState}")
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutting down, closing peer connections")
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

if __name__ == "__main__":
    import uvicorn
    # Make sure this matches your filename. If your file is named 'backend.py', use "backend:app".
    uvicorn.run("servver:app", host=HOST, port=PORT, log_level="info")
