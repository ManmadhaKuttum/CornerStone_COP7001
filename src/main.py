import asyncio
import json
import os
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from transformers.utils import logging as hf_logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

from config import WS_HZ, OFFLINE_MODE, HF_HOME, PROJECT_ROOT
from state import state, reset_session_state
from audio import start_mic_background
from audio import audio_queue
from segmenter import start_segmenter, _load_vad, asr_queue
from asr import start_asr, _load_asr, trans_text_queue
from translation import start_translation, _load_translation, tts_text_queue
from tts import start_tts, _load_tts, stop_playback

if OFFLINE_MODE:
    os.environ.setdefault("HF_HUB_OFFLINE",      "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", HF_HOME)

DASHBOARD_DIR = os.path.join(PROJECT_ROOT, "dashboard")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀 [BOOT] Loading all ML models on main thread...")

    _load_vad()
    try:
        _load_asr()
    except Exception as exc:
        state["asr_status"] = "unavailable"
        print(f"⚠️  ASR failed to load: {exc}")
    try:
        _load_translation()
    except Exception as exc:
        state["trans_status"] = "unavailable"
        print(f"⚠️  Translation failed to load: {exc}")
    try:
        _load_tts()
    except Exception as exc:
        state["tts_status"] = "unavailable"
        print(f"⚠️  TTS failed to load: {exc}")

    state["offline_mode"] = True

    print("🚀 [BOOT] Starting pipeline threads...")
    start_mic_background()
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

def _drain_queue(q):
    while True:
        try:
            q.get_nowait()
        except Exception:
            break

@app.get("/")
async def index():
    return FileResponse(os.path.join(DASHBOARD_DIR, "index.html"))

@app.post("/api/reset-session")
async def reset_session():
    stop_playback()
    reset_session_state()
    _drain_queue(audio_queue)
    _drain_queue(asr_queue)
    _drain_queue(trans_text_queue)
    _drain_queue(tts_text_queue)
    return JSONResponse({"ok": True, "session_id": state["session_id"]})

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
