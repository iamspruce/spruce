import os, asyncio, base64, logging
import cv2, numpy as np, torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, AudioStreamTrack, RTCConfiguration, RTCIceServer
from av import VideoFrame, AudioFrame
from transformers import pipeline
from TTS.api import TTS
from insightface.app import FaceAnalysis
import insightface.model_zoo

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# --- FaceSwap Service ---
class FaceSwapService:
    def __init__(self):
        self.analyzer = FaceAnalysis(name="buffalo_s", root="/workspace/cache/insightface",
                                     providers=["CUDAExecutionProvider","CPUExecutionProvider"])
        self.analyzer.prepare(ctx_id=0, det_size=(640, 640))
        self.swapper = insightface.model_zoo.get_model(
            "/workspace/cache/insightface/inswapper_128.onnx", download=False
        )
        self.source_face = None
    def set_source_face(self, image_bytes: bytes):
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        faces = self.analyzer.get(img)
        if not faces: return False
        self.source_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))[-1]
        return True
    def swap_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self.source_face: return frame
        targets = self.analyzer.get(frame)
        if targets:
            frame = self.swapper.get(frame, targets[0], self.source_face, paste_back=True)
        return frame

# --- Voice Clone Service ---
class VoiceCloneService:
    def __init__(self):
        self.stt = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en", device=0 if torch.cuda.is_available() else -1)
        self.tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=torch.cuda.is_available())
        self.voice_ref = None
    def prepare(self, audio_bytes: bytes):
        path = "/workspace/source_voice.wav"
        with open(path,"wb") as f: f.write(audio_bytes)
        self.voice_ref = path
    def change(self, audio_np: np.ndarray) -> np.ndarray:
        text = self.stt(audio_np)["text"].strip()
        if not text: return np.array([], dtype=np.int16)
        wav = self.tts.tts(text)
        return (wav * 32767).astype(np.int16)

# --- Tracks ---
class ProcessedVideoStreamTrack(VideoStreamTrack):
    def __init__(self, track, swapper): super().__init__(); self.track=track; self.swapper=swapper
    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")
        new = self.swapper.swap_frame(img)
        f2 = VideoFrame.from_ndarray(new, format="bgr24")
        f2.pts, f2.time_base = frame.pts, frame.time_base
        return f2

class ProcessedAudioStreamTrack(AudioStreamTrack):
    def __init__(self, track, vc): super().__init__(); self.track=track; self.vc=vc; self.buffer=np.array([],dtype=np.int16); self.sr=16000
    async def recv(self):
        frame = await self.track.recv()
        data = frame.to_ndarray(format="s16", layout="mono").flatten()
        self.buffer = np.append(self.buffer, data)
        chunk = self.sr*2
        if len(self.buffer)>=chunk:
            seg=self.buffer[:chunk].astype(np.float32)/32768.0; self.buffer=self.buffer[chunk:]
            out=self.vc.change(seg); 
            if out.size>0:
                af=AudioFrame(format="s16", layout="mono", samples=len(out)); af.planes[0].update(out.tobytes()); af.sample_rate=24000; return af
        return frame

# --- App ---
app=FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
swapper=FaceSwapService(); vc=VoiceCloneService(); pcs=set()
config=RTCConfiguration([RTCIceServer(urls=["stun:stun.l.google.com:19302"])])

@app.on_event("shutdown")
async def shutdown(): await asyncio.gather(*[pc.close() for pc in pcs]); pcs.clear()

@app.post("/prepare")
async def prepare(req:Request):
    body=await req.json()
    face=base64.b64decode(body["source_face"]); audio=base64.b64decode(body["source_audio"])
    if not swapper.set_source_face(face): return {"status":"error","message":"No face detected"}
    vc.prepare(audio); return {"status":"success"}

@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("track")
    def on_track(track):
        logger.info("Track received: %s", track.kind)
        if track.kind == "video":
            pc.addTrack(ProcessedVideoStreamTrack(track))
        elif track.kind == "audio":
            pc.addTrack(ProcessedAudioStreamTrack(track))

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # Wait until ICE gathering is complete
    async def wait_for_ice():
        while pc.iceGatheringState != "complete":
            await asyncio.sleep(0.1)
    await wait_for_ice()

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    }
