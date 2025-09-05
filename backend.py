import asyncio
import base64
import logging
import os
import sys
import requests
import json
import uuid
import tempfile

import cv2
import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, Request, Body
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, AudioStreamTrack
from av import VideoFrame, AudioFrame

# Set Cache Directories
os.environ['HF_HOME'] = '/workspace/cache/huggingface'
os.environ['TTS_HOME'] = '/workspace/cache/tts'
os.environ['INSIGHTFACE_HOME'] = '/workspace/cache/insightface'

# --- Logging and Config ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# --- FaceSwapService Class ---
class FaceSwapService:
    def __init__(self):
        logging.info("Loading Face Analysis models...")
        from insightface.app import FaceAnalysis
        import insightface.model_zoo
        self.face_analyzer = FaceAnalysis(name='buffalo_s', root='/workspace/cache/insightface', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        self.face_swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, root='/workspace/cache/insightface')
        self.source_face = None
        logging.info("✅ Face models loaded.")

    def set_source_face(self, image_bytes: bytes):
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        faces = self.face_analyzer.get(img)
        if not faces: return False
        self.source_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))[-1]
        logging.info("✅ Source face set.")
        return True

    def swap_frame_numpy(self, frame_np: np.ndarray) -> np.ndarray:
        if self.source_face is None: return frame_np
        target_faces = self.face_analyzer.get(frame_np)
        if target_faces:
            frame_np = self.face_swapper.get(frame_np, target_faces[0], self.source_face, paste_back=True)
        return frame_np

# --- VoiceCloneService Class ---
class VoiceCloneService:
    def __init__(self):
        logging.info("Loading STT and TTS models...")
        from transformers import pipeline
        from TTS.api import TTS
        self.stt_pipe = pipeline("automatic-speech-recognition", model="distil-whisper/distil-medium.en", torch_dtype=TORCH_DTYPE, device=DEVICE)
        self.tts_pipe = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
        self.speaker_embedding = None
        self.gpt_cond_latent = None
        logging.info("✅ Voice models loaded.")

    def prepare_voice(self, audio_bytes: bytes):
        logging.info("Analyzing source audio to create voice fingerprint...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            tmpfile.write(audio_bytes)
            tmp_path = tmpfile.name
        self.gpt_cond_latent, self.speaker_embedding = self.tts_pipe.synthesizer.tts_model.get_speaker_embedding(tmp_path)
        os.remove(tmp_path)
        logging.info("✅ Voice fingerprint created.")

    def change_voice(self, audio_data: np.ndarray) -> np.ndarray:
        if self.speaker_embedding is None or self.gpt_cond_latent is None: return audio_data
        
        transcription = self.stt_pipe(audio_data)["text"]
        if not transcription.strip(): return np.array([], dtype=np.int16)

        wav_chunks = []
        for chunk in self.tts_pipe.tts_stream(
            text=transcription, language="en",
            speaker_embedding=self.speaker_embedding,
            gpt_cond_latent=self.gpt_cond_latent,
        ):
            wav_chunks.append(chunk.cpu().numpy())
        
        return np.concatenate(wav_chunks)

# --- WebRTC Media Tracks ---
class ProcessedVideoStreamTrack(VideoStreamTrack):
    def __init__(self, track, face_swapper):
        super().__init__()
        self.track = track
        self.face_swapper = face_swapper

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")
        new_img = self.face_swapper.swap_frame_numpy(img)
        new_frame = VideoFrame.from_ndarray(new_img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

class ProcessedAudioStreamTrack(AudioStreamTrack):
    def __init__(self, track, voice_changer):
        super().__init__()
        self.track = track
        self.voice_changer = voice_changer
        self.buffer = np.array([], dtype=np.int16)
        self.sample_rate = 16000 # Whisper's required sample rate

    async def recv(self):
        frame = await self.track.recv()
        frame_data = frame.to_ndarray(format="s16", layout="mono").flatten()
        self.buffer = np.append(self.buffer, frame_data)
        
        chunk_duration_seconds = 2
        chunk_size = self.sample_rate * chunk_duration_seconds
        
        if len(self.buffer) >= chunk_size:
            chunk_to_process = self.buffer[:chunk_size].astype(np.float32) / 32768.0
            self.buffer = self.buffer[chunk_size:]
            
            converted_chunk = self.voice_changer.change_voice(chunk_to_process)
            converted_chunk = (converted_chunk * 32767).astype(np.int16)
            
            new_frame = AudioFrame(format='s16', layout='mono', samples=len(converted_chunk))
            new_frame.planes[0].update(converted_chunk.tobytes())
            new_frame.sample_rate = 24000 # XTTS output sample rate
            return new_frame
        
        # To avoid dead air, we could pass through audio, but for simplicity we wait for a full chunk
        # This is a source of latency. A more complex implementation would stream this better.
        return await self.track.recv() # Pass-through for now

# --- FastAPI Application ---
app = FastAPI()
face_swapper = FaceSwapService()
voice_changer = VoiceCloneService()
pcs = set()

@app.on_event("shutdown")
async def on_shutdown():
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            pc.addTrack(ProcessedVideoStreamTrack(track, face_swapper))
        elif track.kind == "audio":
            pc.addTrack(ProcessedAudioStreamTrack(track, voice_changer))

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            if pc in pcs: pcs.discard(pc)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

@app.post("/prepare")
async def prepare(request: Request):
    body = await request.json()
    face_bytes = base64.b64decode(body["source_face"])
    audio_bytes = base64.b64decode(body["source_audio"])
    
    face_success = face_swapper.set_source_face(face_bytes)
    if not face_success:
        return {"status": "error", "message": "No face found in source image."}
        
    voice_changer.prepare_voice(audio_bytes)
    return {"status": "success"}
