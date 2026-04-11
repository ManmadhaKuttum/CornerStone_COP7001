import asyncio
import json
import os
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config import WS_HZ, OFFLINE_MODE, HF_HOME, PROJECT_ROOT
from state import state
from audio import start_mic
from segmenter import start_segmenter, _load_vad
from asr import start_asr, _load_asr
from translation import start_translation, _load_translation
from tts import start_tts, _load_tts

if OFFLINE_MODE:
    os.environ.setdefault("HF_HUB_OFFLINE",      "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", HF_HOME)

DASHBOARD_DIR = os.path.join(PROJECT_ROOT, "dashboard")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀 [BOOT] Loading all ML models on main thread...")
    state["offline_mode"] = OFFLINE_MODE

    # All models loaded sequentially on the main thread before any worker
    # thread starts — prevents PyTorch multi-threaded init crashes on Mac ARM.
    _load_vad()
    _load_asr()
    _load_translation()
    _load_tts()

    print("🚀 [BOOT] Starting pipeline threads...")
    start_mic()
    start_segmenter()
    start_asr()
    start_translation()
    start_tts()

    asyncio.create_task(broadcast_state())
    print("🎙️  Pipeline live — start speaking!\n")
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=DASHBOARD_DIR), name="static")

_clients: list[WebSocket] = []

@app.get("/")
async def index():
    return FileResponse(os.path.join(DASHBOARD_DIR, "index.html"))

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
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
            "runtime":           int(time.time() - state["start_time"]),
            "speech_onset_time": None,
        }
        msg  = json.dumps(payload)
        dead = []
        for ws in list(_clients):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in _clients:
                _clients.remove(ws)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)