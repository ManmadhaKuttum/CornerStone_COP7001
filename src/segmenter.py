import queue
import threading
import time
import numpy as np
import torch

from audio import audio_queue
from config import (
    SAMPLE_RATE, FRAME_SAMPLES, VAD_THRESHOLD,
    SILENCE_FRAMES, MAX_FRAMES, TRANS_QUEUE_SIZE
)
from state import state

asr_to_trans_queue = queue.Queue(maxsize=TRANS_QUEUE_SIZE)

_vad_model = None
_vad_utils = None

def _load_vad():
    global _vad_model, _vad_utils
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
                state["speech_active"] = True
                state["speech_onset_time"] = time.time()
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
                        asr_to_trans_queue.put_nowait(("asr", utterance))
                    except queue.Full:
                        pass
                    buffer = []
                    silence_count = 0
                    in_speech = False
                    state["speech_active"] = False
            else:
                state["speech_active"] = False

def start_segmenter():
    t = threading.Thread(target=run_segmenter, daemon=True)
    t.start()
    return t