import os
import time
import asyncio
import uvicorn

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from audio import start_audio_stream
from segmenter import start_segmenter
from translation import start_translation
from state import state

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
            await websocket.send_json({
                # Phase 1
                "energy":               round(state["energy"], 5),
                "frames":               state["frames"],
                "runtime":              round(time.time() - state["start_time"], 2),
                "alive":                state["alive"],
                # Phase 2
                "speech_active":        state["speech_active"],  # matches script.js
                "is_speaking":          state["speech_active"],  # alias for dashboard compat
                "partial_transcript":   state["partial_transcript"],
                "final_transcript":     state["final_transcript"],
                "asr_latency_ms":       state["asr_latency_ms"],
                "asr_status":           state["asr_status"],
                "total_words_asr":      state["total_words_asr"],
                # Phase 3
                "partial_translation":  state["partial_translation"],
                "final_translation":    state["final_translation"],
                "trans_latency_ms":     state["trans_latency_ms"],
                "trans_status":         state["trans_status"],
                "trans_backend":        state["trans_backend"],
                "total_words_trans":    state["total_words_trans"],
                # E2E
                "e2e_latency_ms":       state["e2e_latency_ms"],
                # Phase 4
                "tts_status":           state["tts_status"],
            })
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        print("Client disconnected cleanly.")


if __name__ == "__main__":
    start_translation()   # load IndicTrans2 first
    start_segmenter()     # load Silero-VAD + start pipeline

    with start_audio_stream():
        print("[*] Pipeline running → http://localhost:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000)