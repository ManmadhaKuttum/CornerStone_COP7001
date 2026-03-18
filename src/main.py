import os
import time
import asyncio
import threading
import uvicorn

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from audio import start_audio_stream
from segmenter import start_segmenter
from state import state

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")

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
                "alive": state["alive"],
                "speech_active": state["speech_active"],
                "partial_transcript": state["partial_transcript"],
                "final_transcript": state["final_transcript"],
                "asr_latency_ms": state["asr_latency_ms"],
            })
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        print("Client disconnected cleanly.")

if __name__ == "__main__":
    start_segmenter()

    with start_audio_stream():
        uvicorn.run(app, host="0.0.0.0", port=8000)