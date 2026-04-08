import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
import uvicorn

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config import WS_HZ, OFFLINE_MODE, HF_HOME

if OFFLINE_MODE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", HF_HOME)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from state import state
from audio import start_mic
# Import the load functions directly!
from segmenter import start_segmenter, _load_vad
from asr import start_asr, _load_asr
from translation import start_translation, _load_translation
from tts import start_tts, _load_tts

DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀 [SYSTEMS BOOT] Loading models securely on MAIN THREAD...")
    state["offline_mode"] = OFFLINE_MODE

    # 1. Load all ML models sequentially on the main OS thread (Linux/Mac safe)
    print("⏳ Loading VAD...")
    _load_vad()
    
    print("⏳ Loading ASR (Whisper)...")
    _load_asr()
    
    print("⏳ Loading Translation...")
    _load_translation()
    
    print("⏳ Loading TTS...")
    _load_tts()
    
    print("\n✅ All models loaded into memory safely.")
    print("🚀 Igniting concurrent worker threads...")

    # 2. Now it is 100% safe to start the background threads
    start_mic()
    start_segmenter()
    start_asr()
    start_translation()
    start_tts()

    asyncio.create_task(broadcast_state())
    print("🎙️ Pipeline fully operational! You can now speak.\n")
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=DASHBOARD_DIR), name="static")

_clients: list[WebSocket] = []

@app.get("/")
async def index():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(DASHBOARD_DIR, "index.html"))

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        if ws in _clients:
            _clients.remove(ws)

async def broadcast_state():
    interval = 1.0 / WS_HZ
    while True:
        await asyncio.sleep(interval)
        payload = {
            **state,
            "runtime": int(time.time() - state["start_time"]),
            "speech_onset_time": None,
        }
        msg = json.dumps(payload)
        dead = []
        for ws in _clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in _clients:
                _clients.remove(ws)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
