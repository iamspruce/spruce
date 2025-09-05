import asyncio
import base64
import logging
import os
import sys
import requests
import io

import cv2
import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

# --- ⚙️ Global Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# --- 🤖 RunPod Auto-Shutdown Service ---
class RunPodManager:
    def __init__(self):
        self.api_key = os.getenv("RUNPOD_API_KEY")
        self.pod_id = os.getenv("RUNPOD_POD_ID")
        if not self.api_key or not self.pod_id:
            logging.warning("⚠️ RUNPOD_API_KEY or RUNPOD_POD_ID not set. Auto-shutdown is disabled.")

    def stop_pod(self):
        if not self.api_key or not self.pod_id:
            return

        logging.info(f"🛑 Stopping RunPod instance {self.pod_id}...")
        url = f"https://api.runpod.io/graphql?api_key={self.api_key}"
        query = f"""
        mutation {{
            podStop(input: {{ podId: "{self.pod_id}" }}) {{
                id
                desiredStatus
            }}
        }}
        """
        try:
            requests.post(url, json={'query': query})
            logging.info("✅ Stop command sent to RunPod API.")
        except Exception as e:
            logging.error(f"❌ Failed to send stop command: {e}")

# --- 🧑‍🎨 Face Swap Service (In-Memory) ---
class FaceSwapService:
    def __init__(self):
        logging.info("Loading Face Analysis and Swapper models...")
        from insightface.app import FaceAnalysis
        import insightface.model_zoo
        self.face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        self.face_swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False)
        self.source_face = None
        logging.info("✅ Face models loaded.")

    def set_source_face(self, image_bytes: bytes):
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        faces = self.face_analyzer.get(img)
        if not faces:
            logging.error("❌ No face found in source image.")
            return False
        self.source_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))[-1]
        logging.info("✅ Source face set.")
        return True

    def swap_frame(self, frame_bytes: bytes) -> bytes:
        if not self.source_face: return frame_bytes
        frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        target_faces = self.face_analyzer.get(frame)
        if target_faces:
            frame = self.face_swapper.get(frame, target_faces[0], self.source_face, paste_back=True)
        _, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()

# --- 🎤 Zero-Shot Voice Clone Service (In-Memory) ---
class VoiceCloneService:
    def __init__(self):
        logging.info("Loading STT and TTS models...")
        from transformers import pipeline
        from TTS.api import TTS
        self.stt_pipe = pipeline("automatic-speech-recognition", model="distil-whisper/distil-large-v2", torch_dtype=TORCH_DTYPE, device=DEVICE)
        self.tts_pipe = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
        self.speaker_latents = None
        logging.info("✅ Voice models loaded.")

    def prepare_voice(self, audio_bytes: bytes):
        logging.info("Analyzing source audio to create voice fingerprint...")
        with io.BytesIO(audio_bytes) as audio_file:
            # Coqui TTS can compute latents directly from a file path-like object
            self.speaker_latents = self.tts_pipe.get_speaker_latents(audio_file)
        logging.info("✅ Voice fingerprint created.")

    def convert_voice(self, audio_bytes: bytes) -> bytes:
        if not self.speaker_latents: return audio_bytes
        
        # 1. Speech-to-Text (STT)
        with io.BytesIO(audio_bytes) as audio_file:
            transcription = self.stt_pipe(audio_file.read())["text"]

        if not transcription.strip():
            return audio_bytes # Return original audio if no speech is detected

        # 2. Text-to-Speech (TTS) with cloned voice
        wav_chunks = []
        # XTTS generates audio in chunks
        for chunk in self.tts_pipe.tts_stream(
            text=transcription,
            language="en",
            speaker_latents=self.speaker_latents,
            speed=1.0,
        ):
            wav_chunks.append(chunk.cpu().numpy())
        
        # Combine chunks and convert back to bytes
        full_audio = np.concatenate(wav_chunks)
        with io.BytesIO() as out_wav_file:
            sf.write(out_wav_file, full_audio, 24000, format='WAV')
            return out_wav_file.getvalue()

# --- 🚀 FastAPI Application ---
app = FastAPI()
runpod_manager = RunPodManager()
face_swapper = FaceSwapService()
voice_cloner = VoiceCloneService()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.info("✅ Frontend connected.")
    try:
        # Stage 1: Preparation
        config_msg = await websocket.receive_json()
        face_bytes = base64.b64decode(config_msg["source_face"])
        audio_bytes = base64.b64decode(config_msg["source_audio"])

        if not face_swapper.set_source_face(face_bytes):
            await websocket.send_json({"status": "error", "message": "Could not set source face."})
            return
        
        voice_cloner.prepare_voice(audio_bytes)
        await websocket.send_json({"status": "ready"})
        logging.info("🚀 System ready. Starting stream processing...")

        # Stage 2: Real-time loop
        while True:
            message = await websocket.receive_json()
            
            video_bytes = base64.b64decode(message["video"].split(",")[1])
            swapped_bytes = face_swapper.swap_frame(video_bytes)
            response_video = base64.b64encode(swapped_bytes).decode("utf-8")
            
            audio_bytes = base64.b64decode(message["audio"].split(",")[1])
            converted_audio = voice_cloner.convert_voice(audio_bytes)
            response_audio = base64.b64encode(converted_audio).decode("utf-8")

            await websocket.send_json({
                "video": f"data:image/jpeg;base64,{response_video}",
                "audio": f"data:audio/wav;base64,{response_audio}",
            })

    except WebSocketDisconnect:
        logging.info("Frontend disconnected.")
    except Exception as e:
        logging.error(f"❌ WebSocket error: {e}", exc_info=True)
    finally:
        # This is crucial for cost-saving
        runpod_manager.stop_pod()
