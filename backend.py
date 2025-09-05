import asyncio
import base64
import logging
import os
import sys
import requests
import io
import tempfile

# Set Cache Directories to a Persistent Location
os.environ['HF_HOME'] = '/workspace/cache/huggingface'
os.environ['TTS_HOME'] = '/workspace/cache/tts'
os.environ['INSIGHTFACE_HOME'] = '/workspace/cache/insightface'

import cv2
import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

# Global Configuration & Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# RunPod Auto-Shutdown Service
class RunPodManager:
    # ... (no changes in this class)
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

# Face Swap Service (In-Memory)
class FaceSwapService:
    # ... (no changes in this class)
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

# Zero-Shot Voice Clone Service (In-Memory)
class VoiceCloneService:
    # ... (no changes in __init__ or convert_voice)
    def __init__(self):
        logging.info("Loading STT and TTS models...")
        from transformers import pipeline
        from TTS.api import TTS
        self.stt_pipe = pipeline("automatic-speech-recognition", model="distil-whisper/distil-large-v2", torch_dtype=TORCH_DTYPE, device=DEVICE)
        self.tts_pipe = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
        self.speaker_embedding = None
        self.gpt_cond_latent = None
        logging.info("✅ Voice models loaded.")

    def prepare_voice(self, audio_bytes: bytes):
        logging.info("Analyzing source audio to create voice fingerprint...")
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                tmpfile.write(audio_bytes)
                tmp_path = tmpfile.name
            
            # **FIX:** Load the audio file into a tensor before passing it to the function
            audio_data, sample_rate = sf.read(tmp_path)
            audio_tensor = torch.FloatTensor(audio_data).unsqueeze(0)
            
            # **FIX:** Call the function with the audio tensor and sample rate
            self.gpt_cond_latent, self.speaker_embedding = self.tts_pipe.synthesizer.tts_model.get_speaker_embedding(audio_tensor, sample_rate)
            
            os.remove(tmp_path)
            logging.info("✅ Voice fingerprint created.")
        except Exception as e:
            logging.error(f"❌ Failed to prepare voice: {e}", exc_info=True)

    def convert_voice(self, audio_bytes: bytes) -> bytes:
        if self.speaker_embedding is None or self.gpt_cond_latent is None:
            return audio_bytes
        
        try:
            with io.BytesIO(audio_bytes) as audio_file:
                transcription = self.stt_pipe(audio_file.read())["text"]

            if not transcription.strip():
                return audio_bytes

            wav_chunks = []
            for chunk in self.tts_pipe.tts_stream(
                text=transcription,
                language="en",
                speaker_embedding=self.speaker_embedding,
                gpt_cond_latent=self.gpt_cond_latent,
                speed=1.0,
            ):
                wav_chunks.append(chunk.cpu().numpy())
            
            full_audio = np.concatenate(wav_chunks)
            with io.BytesIO() as out_wav_file:
                sf.write(out_wav_file, full_audio, 24000, format='WAV')
                return out_wav_file.getvalue()
        except Exception as e:
            logging.error(f"❌ Voice conversion failed: {e}", exc_info=True)
            return audio_bytes

# FastAPI Application
app = FastAPI()
runpod_manager = RunPodManager()
face_swapper = FaceSwapService()
voice_cloner = VoiceCloneService()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # ... (no changes in this function)
    await websocket.accept()
    logging.info("✅ Frontend connected.")
    try:
        config_msg = await websocket.receive_json()
        face_bytes = base64.b64decode(config_msg["source_face"])
        audio_bytes = base64.b64decode(config_msg["source_audio"])

        if not face_swapper.set_source_face(face_bytes):
            await websocket.send_json({"status": "error", "message": "Could not set source face."})
            return
        
        voice_cloner.prepare_voice(audio_bytes)
        await websocket.send_json({"status": "ready"})
        logging.info("🚀 System ready. Starting stream processing...")

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
        pass

@app.post("/shutdown")
async def shutdown_pod():
    # ... (no changes in this function)
    logging.info("Received shutdown request from client.")
    runpod_manager.stop_pod()
    return {"status": "shutdown sequence initiated"}
