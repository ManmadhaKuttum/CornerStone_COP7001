import sounddevice as sd
import numpy as np
import queue
import threading
import time
import asyncio
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")

# ---------- AUDIO CONFIG ----------
SAMPLE_RATE = 16000
FRAME_DURATION = 0.02
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)

audio_queue = queue.Queue(maxsize=100)

state = {
    "energy": 0.0,
    "frames": 0,
    "start_time": time.time(),
    "alive": True
}

# ---------- AUDIO CALLBACK ----------
def audio_callback(indata, frames, time_info, status):
    try:
        audio_queue.put_nowait(indata.copy())
    except queue.Full:
        pass

# ---------- MONITOR THREAD ----------
def monitor_audio():
    while True:
        frame = audio_queue.get()
        energy = float(np.linalg.norm(frame) / len(frame))

        state["energy"] = energy
        state["frames"] += 1

# ---------- START AUDIO ----------
def start_audio():
    monitor_thread = threading.Thread(target=monitor_audio, daemon=True)
    monitor_thread.start()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=FRAME_SIZE,
        callback=audio_callback
    ):
        while True:
            time.sleep(1)

# ---------- FASTAPI ----------
app = FastAPI()

app.mount("/static", StaticFiles(directory=DASHBOARD_DIR), name="static")

@app.get("/")
async def serve_dashboard():
    return FileResponse(os.path.join(DASHBOARD_DIR, "index.html"))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            runtime = time.time() - state["start_time"]

            await websocket.send_json({
                "energy": state["energy"],
                "frames": state["frames"],
                "runtime": round(runtime, 2),
                "alive": state["alive"]
            })

            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        print("Client disconnected cleanly.")


# ---------- MAIN ----------
if __name__ == "__main__":
    audio_thread = threading.Thread(target=start_audio, daemon=True)
    audio_thread.start()

    uvicorn.run(app, host="0.0.0.0", port=8000)