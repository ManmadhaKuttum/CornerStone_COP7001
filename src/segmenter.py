import os
import queue
import threading
import time
import numpy as np
import torch

from audio import audio_queue
from config import (
    SAMPLE_RATE, VAD_THRESHOLD,
    SILENCE_FRAMES, MAX_FRAMES, TRANS_QUEUE_SIZE,
    PARTIAL_INTERVAL, OFFLINE_MODE, SILERO_VAD_PATH,
)
from state import state

asr_queue = queue.Queue(maxsize=TRANS_QUEUE_SIZE)

_vad_model = None
_vad_mode = "silero"

def _load_vad():
    global _vad_model, _vad_mode
    path = SILERO_VAD_PATH if (
        SILERO_VAD_PATH and
        os.path.isfile(os.path.join(SILERO_VAD_PATH, "hubconf.py"))
    ) else _find_silero_in_hub_cache()

    if path:
        model, _ = torch.hub.load(
            repo_or_dir=path,
            model="silero_vad",
            source="local",
            force_reload=False,
            onnx=False,
            verbose=False,
        )
        _vad_model = model
        _vad_mode = "silero"
        state["vad_backend"] = "silero-vad"
        print("  ✅ VAD loaded (Silero)")
        return

    if not OFFLINE_MODE:
        try:
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
                verbose=False,
            )
            _vad_model = model
            _vad_mode = "silero"
            state["vad_backend"] = "silero-vad"
            print("  ✅ VAD loaded (Silero)")
            return
        except Exception as exc:
            print(f"⚠️  Silero VAD unavailable, falling back to energy VAD: {exc}")

    _vad_model = None
    _vad_mode = "energy"
    state["vad_backend"] = "energy"
    print("  ⚠️  Using simple energy-based VAD fallback")

def _find_silero_in_hub_cache():
    try:
        hub_dir = torch.hub.get_dir()
        for name in os.listdir(hub_dir):
            if "silero-vad" in name.lower():
                cand = os.path.join(hub_dir, name)
                if os.path.isfile(os.path.join(cand, "hubconf.py")):
                    return cand
    except Exception:
        pass
    return None

def _vad_conf(frame: np.ndarray) -> float:
    if _vad_mode == "energy":
        rms = float(np.sqrt(np.mean(frame ** 2)))
        return min(1.0, rms * 40.0)
    with torch.no_grad():
        return _vad_model(torch.from_numpy(frame).unsqueeze(0), SAMPLE_RATE).item()

def _enqueue(tag: str, buffer: list, onset_time: float):
    if not buffer:
        return
    audio = np.concatenate(buffer)
    state["last_audio_duration"] = len(audio) / SAMPLE_RATE
    try:
        asr_queue.put_nowait((state["session_id"], tag, audio, onset_time))
    except queue.Full:
        pass

def run_segmenter():
    buffer        = []
    silence_count = 0
    in_speech     = False
    onset_time    = None
    last_partial  = 0.0
    active_session = state["session_id"]

    while True:
        if active_session != state["session_id"]:
            buffer = []
            silence_count = 0
            in_speech = False
            onset_time = None
            last_partial = 0.0
            active_session = state["session_id"]

        try:
            frame = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        state["frames"] += 1
        state["energy"]  = float(np.sqrt(np.mean(frame ** 2)))

        is_speech = _vad_conf(frame) >= VAD_THRESHOLD

        if is_speech:
            if not in_speech:
                in_speech    = True
                onset_time   = time.time()
                last_partial = onset_time
                state["speech_active"]     = True
                state["speech_onset_time"] = onset_time
            buffer.append(frame)
            silence_count = 0

            now = time.time()
            if now - last_partial >= PARTIAL_INTERVAL:
                _enqueue("partial", list(buffer), onset_time)
                last_partial = now

        else:
            if in_speech:
                silence_count += 1
                buffer.append(frame)

                if silence_count >= SILENCE_FRAMES or len(buffer) >= MAX_FRAMES:
                    _enqueue("final", buffer, onset_time)
                    buffer        = []
                    silence_count = 0
                    in_speech     = False
                    onset_time    = None
                    state["speech_active"]      = False
                    state["partial_transcript"] = ""
            else:
                state["speech_active"] = False

def start_segmenter():
    t = threading.Thread(target=run_segmenter, daemon=True, name="segmenter")
    t.start()
    return t
