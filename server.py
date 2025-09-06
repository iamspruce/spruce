#!/usr/bin/env python3
"""
Production-ready WebRTC avatar backend (FastAPI + aiortc). 
(Video + FaceSwap only, no voice models)
"""

import os
import sys
import asyncio
import base64
import logging
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

ICE_SERVERS = [
    RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
    RTCIceServer(
        urls=["turn:global.turn.twilio.com:3478?transport=udp"],
        username="AC42be81518dbaa82d01a62f7f5adbf2cd",
        credential="3fbe69c38157120ce43f716729787b0f",
    ),
    RTCIceServer(
        urls=["turn:global.turn.twilio.com:3478?transport=tcp"],
        username="AC42be81518dbaa82d01a62f7f5adbf2cd",
        credential="3fbe69c38157120ce43f716729787b0f",
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
            ctx_id = 0 if torch.cuda.is_available() else -1
            self.analyzer = FaceAnalysis(name="buffalo_s", root=INSIGHTFACE_HOME)
            self.analyzer.prepare(ctx_id=ctx_id, det_size=(640, 640))
            logger.info("FaceAnalysis ready (ctx=%s)", ctx_id)
        except Exception:
            logger.exception("Failed to initialize FaceAnalysis; face detection will be limited.")
            self.analyzer = None

        # ---------- Load or download inswapper ----------
        try:
            os.makedirs(os.path.join(INSIGHTFACE_HOME, "models"), exist_ok=True)
            if os.path.exists(INSWAPPER_FULLPATH):
                logger.info(f"Inswapper model found at {INSWAPPER_FULLPATH}, loading locally")
                self.inswapper = insightface.model_zoo.get_model(INSWAPPER_FULLPATH, download=False)
            else:
                logger.info(f"Inswapper not found, downloading to {INSIGHTFACE_HOME}/models")
                self.inswapper = insightface.model_zoo.get_model(
                    INSWAPPER_PATH,
                    download=True,
                    root=INSIGHTFACE_HOME
                )
            logger.info("Inswapper ready")
        except Exception:
            logger.exception("Inswapper load failed; face swapping may be limited")
            self.inswapper = None

        self.source_face = None
        self.source_face_img = None

    def set_source_face(self, image_bytes: bytes) -> bool:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return False
        if self.analyzer:
            try:
                faces = self.analyzer.get(img)
                if faces:
                    self.source_face = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))[-1]
                    logger.info("Source face saved (insightface descriptor)")
                    return True
            except Exception:
                logger.exception("Analyzer failed on source img - will store fallback crop")
        h, w = img.shape[:2]
        s = min(h, w)
        cy, cx = h // 2, w // 2
        y1, x1 = max(0, cy - s // 2), max(0, cx - s // 2)
        self.source_face_img = img[y1:y1 + s, x1:x1 + s].copy()
        logger.info("Source face saved as fallback center-crop")
        return True

    def detect(self, frame_bgr: np.ndarray):
        if not self.analyzer:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            if not hasattr(self, "_cascade"):
                self._cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
            dets = []
            for (x, y, w, h) in faces:
                class D: pass
                d = D(); d.bbox = (x, y, x + w, y + h)
                dets.append(d)
            return dets
        try:
            return self.analyzer.get(frame_bgr)
        except Exception:
            logger.exception("Face detection error")
            return []

    def swap(self, frame_bgr: np.ndarray, dets):
        try:
            if self.inswapper and self.source_face and dets:
                return self.inswapper.get(frame_bgr, dets[0], self.source_face, paste_back=True)
        except Exception:
            logger.exception("Inswapper error, falling back")
        if self.source_face_img is not None and dets:
            try:
                d = dets[0]
                x1, y1, x2, y2 = map(int, d.bbox)
                w, h = x2 - x1, y2 - y1
                src = cv2.resize(self.source_face_img, (w, h))
                frame_bgr[y1:y2, x1:x2] = src
            except Exception:
                logger.exception("Fallback paste failed")
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
        raise HTTPException(status_code=400, detail="invalid base64 face")
    if not face_service.set_source_face(face_bytes):
        raise HTTPException(status_code=400, detail="failed to set source face")
    return {"status": "success"}

@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    if "sdp" not in params or "type" not in params:
        raise HTTPException(status_code=400, detail="sdp and type required")
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection(configuration=RTC_CONFIG)
    pcs.add(pc)
    logger.info("Created PeerConnection %s", id(pc))

    @pc.on("track")
    def on_track(track):
        logger.info("Track received: %s", track.kind)
        if track.kind == "video":
            processed_track = ProcessedVideoStreamTrack(track, face_service)
            pc.addTrack(processed_track)

            async def force_keyframe():
                await asyncio.sleep(0.5)
                video_sender = next(
                    (s for s in pc.getSenders() if s.track and s.track.kind == "video"),
                    None
                )
                if video_sender:
                    try:
                        await video_sender.send_rtcp_pli()
                        logger.info("PLI sent to request a keyframe from the client")
                    except Exception as e:
                        logger.error(f"Failed to send PLI request: {e}")
            
            asyncio.ensure_future(force_keyframe())

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state %s -> %s", id(pc), pc.connectionState)
        if pc.connectionState in ("failed", "closed", "disconnected"):
            try:
                await pc.close()
            except Exception: pass
            pcs.discard(pc)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    async def wait_for_ice():
        for _ in range(50):
            if pc.iceGatheringState == "complete":
                return
            await asyncio.sleep(0.1)
    await wait_for_ice()

    logger.info("Returning answer for PC %s", id(pc))
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutting down, closing peer connections")
    coros = [pc.close() for pc in pcs]
    if coros:
        await asyncio.gather(*coros)
    pcs.clear()

# ---------- Instantiate PeerConnection set ----------
pcs = set()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=HOST, port=PORT, log_level="info")
