#!/usr/bin/env python3
"""
Production-ready WebRTC avatar backend (FastAPI + aiortc).

- Correctly constructs RTCPeerConnection with ICE config (STUN by default)
- Passes swapper and voice services into processed tracks
- Uses run_in_executor for heavier frame ops (non-blocking)
- Waits for ICE gathering before returning answer SDP
- Graceful peer cleanup and shutdown
"""

import os
import sys
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
    AudioStreamTrack,
)
from av import VideoFrame, AudioFrame

# Optional model libs - ensure installed in your environment
from insightface.app import FaceAnalysis
import insightface.model_zoo
from transformers import pipeline
from TTS.api import TTS

# ---------- Config ----------
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))

INSIGHTFACE_HOME = os.getenv("INSIGHTFACE_HOME", "/workspace/cache/insightface")
INSWAPPER_PATH = os.getenv("INSWAPPER_PATH", "/workspace/inswapper_128.onnx")

WHISPER_MODEL_ID = os.getenv("WHISPER_MODEL_ID", "openai/whisper-tiny.en")
TTS_MODEL_ID = os.getenv("TTS_MODEL_ID", "tts_models/en/ljspeech/tacotron2-DDC")

ICE_SERVERS = [RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
RTC_CONFIG = RTCConfiguration(iceServers=ICE_SERVERS)

AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
AUDIO_CHUNK_SECONDS = float(os.getenv("AUDIO_CHUNK_SECONDS", "0.25"))
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

        # load inswapper ONNX if available
        try:
            if os.path.exists(INSWAPPER_PATH):
                self.inswapper = insightface.model_zoo.get_model(INSWAPPER_PATH, download=False, root=INSIGHTFACE_HOME)
            else:
                # try model zoo id
                self.inswapper = insightface.model_zoo.get_model("inswapper_128.onnx", download=False, root=INSIGHTFACE_HOME)
            logger.info("Inswapper ready")
        except Exception:
            logger.exception("Inswapper load failed; falling back to naive paste")
            self.inswapper = None

        self.source_face = None  # insightface face object
        self.source_face_img = None  # fallback BGR crop

    def set_source_face(self, image_bytes: bytes) -> bool:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return False
        # try analyzer
        if self.analyzer:
            try:
                faces = self.analyzer.get(img)
                if faces:
                    # pick largest
                    self.source_face = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))[-1]
                    logger.info("Source face saved (insightface descriptor)")
                    return True
            except Exception:
                logger.exception("Analyzer failed on source img - will store fallback crop")
        # fallback: store center crop
        h, w = img.shape[:2]
        s = min(h, w)
        cy, cx = h // 2, w // 2
        y1 = max(0, cy - s // 2); x1 = max(0, cx - s // 2)
        self.source_face_img = img[y1:y1 + s, x1:x1 + s].copy()
        logger.info("Source face saved as fallback center-crop")
        return True

    def detect(self, frame_bgr: np.ndarray):
        if not self.analyzer:
            # fallback detection with Haar cascade (fast, less accurate)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            if not hasattr(self, "_cascade"):
                self._cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
            dets = []
            for (x, y, w, h) in faces:
                # create a simple object with bbox attribute to mimic insightface result
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
        # try inswapper if available and we have a source descriptor
        try:
            if self.inswapper and self.source_face and dets:
                out = self.inswapper.get(frame_bgr, dets[0], self.source_face, paste_back=True)
                return out
        except Exception:
            logger.exception("Inswapper error, falling back")
        # fallback: naive paste of stored crop
        if self.source_face_img is not None and dets:
            try:
                d = dets[0]
                x1, y1, x2, y2 = map(int, d.bbox)
                w = x2 - x1; h = y2 - y1
                src = cv2.resize(self.source_face_img, (w, h))
                frame_bgr[y1:y2, x1:x2] = src
            except Exception:
                logger.exception("Fallback paste failed")
        return frame_bgr


class VoiceCloneService:
    def __init__(self):
        logger.info("Loading STT & TTS (lightweight models)...")
        try:
            self.stt = pipeline("automatic-speech-recognition", model=WHISPER_MODEL_ID, device=0 if torch.cuda.is_available() else -1)
            logger.info("STT pipeline ready")
        except Exception:
            logger.exception("STT load failed; speech -> text will be disabled")
            self.stt = None
        try:
            self.tts = TTS(TTS_MODEL_ID, gpu=torch.cuda.is_available())
            logger.info("TTS ready")
        except Exception:
            logger.exception("TTS load failed; generation disabled")
            self.tts = None
        self.voice_ref = None

    def prepare(self, audio_bytes: bytes):
        path = os.path.join(tempfile.gettempdir(), f"voice_ref_{uuid_safe()}.wav")
        with open(path, "wb") as f:
            f.write(audio_bytes)
        self.voice_ref = path
        logger.info("Saved voice_ref at %s", path)

    def transcribe_and_synthesize(self, float32_audio: np.ndarray):
        """
        Blocking: run STT then TTS. We call this in executor.
        Input: float32 numpy in -1..1
        Output: float32 numpy -1..1
        """
        if self.stt is None or self.tts is None:
            # passthrough
            return float32_audio
        # write chunk to temp file for pipeline if needed
        tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            sf_write(tmp_in.name, float32_audio, AUDIO_SAMPLE_RATE)
            tmp_in.close()
            # STT
            try:
                res = self.stt(tmp_in.name)
                text = res.get("text", "") if isinstance(res, dict) else str(res)
                text = text.strip()
            except Exception:
                logger.exception("STT call failed"); text = ""
            if not text:
                return float32_audio
            # TTS to file
            tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_out.close()
            try:
                kwargs = {}
                if self.voice_ref:
                    kwargs["speaker_wav"] = self.voice_ref
                self.tts.tts_to_file(text=text, file_path=tmp_out.name, **kwargs)
                arr, sr = sf_read(tmp_out.name)
                if sr != AUDIO_SAMPLE_RATE:
                    arr = simple_resample(arr, sr, AUDIO_SAMPLE_RATE)
                if arr.ndim > 1:
                    arr = arr.mean(axis=1)
                return arr.astype(np.float32)
            except Exception:
                logger.exception("TTS call failed")
                return float32_audio
            finally:
                safe_remove(tmp_out.name)
        finally:
            safe_remove(tmp_in.name)


# ---------- small helpers ----------
import uuid as _uuid
def uuid_safe(): return _uuid.uuid4().hex
def safe_remove(p):
    try: os.remove(p)
    except Exception: pass

# use soundfile but import lazily for graceful failure
def sf_write(path, data, sr):
    import soundfile as sf
    sf.write(path, data, sr)
def sf_read(path):
    import soundfile as sf
    return sf.read(path, dtype="float32")

def simple_resample(x, old_sr, new_sr):
    if old_sr == new_sr: return x
    old_len = x.shape[0]
    new_len = int(old_len * new_sr / old_sr)
    return np.interp(np.linspace(0, old_len, new_len, endpoint=False), np.arange(old_len), x).astype(np.float32)

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

        # run detection periodically in executor
        if (self.frame_index % FRAME_SKIP) == 0:
            loop = asyncio.get_running_loop()
            try:
                dets = await loop.run_in_executor(executor, self.face_service.detect, img.copy())
                self.cached_dets = dets
            except Exception:
                logger.exception("Detection executor failed")
                self.cached_dets = []

        # perform swap in executor (non-blocking)
        loop = asyncio.get_running_loop()
        processed = await loop.run_in_executor(executor, self.face_service.swap, img.copy(), self.cached_dets)

        new_frame = VideoFrame.from_ndarray(processed, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

class ProcessedAudioStreamTrack(AudioStreamTrack):
    def __init__(self, incoming_track, voice_service: VoiceCloneService):
        super().__init__()
        self.track = incoming_track
        self.voice_service = voice_service
        self.buffer = np.array([], dtype=np.float32)
        self.sample_rate = AUDIO_SAMPLE_RATE
        # small queues for background processing
        self._process_q = asyncio.Queue(maxsize=8)
        self._out_q = asyncio.Queue(maxsize=16)
        self._worker = asyncio.create_task(self._audio_worker())

    async def _audio_worker(self):
        loop = asyncio.get_running_loop()
        while True:
            chunk = await self._process_q.get()
            if chunk is None:
                break
            # blocking STT+TTS in executor
            processed = await loop.run_in_executor(executor, self.voice_service.transcribe_and_synthesize, chunk)
            int16 = np.clip(processed * 32767.0, -32768, 32767).astype(np.int16)
            await self._out_q.put(int16.tobytes())

    async def recv(self):
        frame = await self.track.recv()
        arr = frame.to_ndarray().flatten().astype(np.int16)
        float32 = (arr.astype(np.float32) / 32768.0).copy()
        self.buffer = np.concatenate([self.buffer, float32])

        chunk_size = int(self.sample_rate * AUDIO_CHUNK_SECONDS)
        if len(self.buffer) >= chunk_size:
            chunk = self.buffer[:chunk_size].copy()
            self.buffer = self.buffer[chunk_size:]
            try:
                self._process_q.put_nowait(chunk)
            except asyncio.QueueFull:
                logger.warning("Audio process queue full; dropping chunk")

        # if processed available, return it
        try:
            processed_bytes = self._out_q.get_nowait()
            samples = np.frombuffer(processed_bytes, dtype=np.int16)
            new_frame = AudioFrame(format="s16", layout="mono", samples=len(samples))
            new_frame.planes[0].update(samples.tobytes())
            new_frame.sample_rate = self.sample_rate
            return new_frame
        except asyncio.QueueEmpty:
            return frame

    async def stop(self):
        try:
            await self._process_q.put(None)
        except Exception:
            pass
        if not self._worker.done():
            self._worker.cancel()
        await super().stop()

# ---------- App endpoints ----------
face_service = FaceSwapService()
voice_service = VoiceCloneService()

@app.post("/prepare")
async def prepare(req: Request):
    data = await req.json()
    face_b64 = data.get("source_face")
    audio_b64 = data.get("source_audio")
    if not face_b64:
        raise HTTPException(status_code=400, detail="source_face required")
    try:
        face_bytes = base64.b64decode(face_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid base64 face")
    ok = face_service.set_source_face(face_bytes)
    if not ok:
        raise HTTPException(status_code=400, detail="failed to set source face")
    if audio_b64:
        try:
            audio_bytes = base64.b64decode(audio_b64)
            voice_service.prepare(audio_bytes)
        except Exception:
            logger.exception("Failed to store voice sample (continuing)")
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
            pc.addTrack(ProcessedVideoStreamTrack(track, face_service))
        elif track.kind == "audio":
            pc.addTrack(ProcessedAudioStreamTrack(track, voice_service))

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state %s -> %s", id(pc), pc.connectionState)
        if pc.connectionState in ("failed", "closed", "disconnected"):
            try:
                await pc.close()
            except Exception:
                pass
            pcs.discard(pc)

    # set remote & create answer
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # wait for ICE gathering to complete (so returned SDP contains candidates)
    async def wait_for_ice():
        # wait up to 5s
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host=HOST, port=PORT, log_level="info")
