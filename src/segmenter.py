import queue
import threading
import time
import os
import numpy as np
import torch

from audio import audio_queue
from config import (
    SAMPLE_RATE, FRAME_SAMPLES, VAD_THRESHOLD,
    SILENCE_FRAMES, MAX_FRAMES, TRANS_QUEUE_SIZE,
    OFFLINE_MODE, SILERO_VAD_PATH
)
from state import state

asr_to_trans_queue = queue.Queue(maxsize=TRANS_QUEUE_SIZE)

_vad_model = None
_vad_utils = None

def _resolve_silero_path():
    # Prefer explicit path
    if os.path.isdir(SILERO_VAD_PATH):
        return SILERO_VAD_PATH

    # Fallback: auto-detect Torch Hub cache
    try:
        hub_dir = torch.hub.get_dir()
        if os.path.isdir(hub_dir):
            for name in os.listdir(hub_dir):
                if "silero-vad" in name.lower():
                    cand = os.path.join(hub_dir, name)
                    if os.path.isfile(os.path.join(cand, "hubconf.py")):
                        return cand
    except Exception:
        pass

    return None

def _load_vad():
    global _vad_model, _vad_utils
    if OFFLINE_MODE:
        silero_path = _resolve_silero_path()
        if not silero_path:
            raise RuntimeError(
                "Offline mode enabled but local Silero VAD repo was not found. "
                "Set SILERO_VAD_PATH to your downloaded Silero repo or ensure the "
                "Torch Hub cache already contains silero-vad."
            )
        model, utils = torch.hub.load(
            repo_or_dir=silero_path,
            model="silero_vad",
            source="local",
            force_reload=False,
            onnx=False,
        )
    else:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
    _vad_model = model
    _vad_utils = utils

def _vad_confidence(frame: np.ndarray) -> float:
    tensor = torch.from_numpy(frame).unsqueeze(0)
    with torch.no_grad():
        conf = _vad_model(tensor, SAMPLE_RATE).item()
    return conf

def run_segmenter():
   
    buffer = []
    silence_count = 0
    in_speech = False
    onset_time = None

    while True:
        try:
            frame = audio_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        state["frames"] += 1
        energy = float(np.sqrt(np.mean(frame ** 2)))
        state["energy"] = energy

        conf = _vad_confidence(frame)
        is_speech = conf >= VAD_THRESHOLD

        if is_speech:
            if not in_speech:
                in_speech = True
                onset_time = time.time()
                state["speech_active"] = True
                state["speech_onset_time"] = onset_time
            buffer.append(frame)
            silence_count = 0
        else:
            if in_speech:
                silence_count += 1
                buffer.append(frame)
                if silence_count >= SILENCE_FRAMES or len(buffer) >= MAX_FRAMES:
                    utterance = np.concatenate(buffer)
                    audio_seconds = len(utterance) / SAMPLE_RATE
                    state["last_audio_duration"] = audio_seconds
                    try:
                        asr_to_trans_queue.put_nowait(("asr", utterance, onset_time))
                    except queue.Full:
                        pass
                    buffer = []
                    silence_count = 0
                    in_speech = False
                    onset_time = None
                    state["speech_active"] = False
            else:
                state["speech_active"] = False

def start_segmenter():
    t = threading.Thread(target=run_segmenter, daemon=True)
    t.start()
    return t
