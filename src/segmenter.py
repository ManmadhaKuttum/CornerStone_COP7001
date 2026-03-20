"""
segmenter.py — Phase 2
Silero-VAD for speech detection (replaces energy threshold).
On utterance end → ASR → translation queue.
"""

import time
import threading
import numpy as np
import torch

from audio import audio_queue
from state import state
from config import (
    SAMPLE_RATE, VAD_THRESHOLD,
    SILENCE_DURATION, MAX_SPEECH_DURATION, MIN_SPEECH_DURATION,
)
from asr import transcribe_segment
from translation import asr_to_trans_queue

# ── Load Silero-VAD ───────────────────────────────────────────────────────────
print("Loading Silero VAD model...")
_vad_model, _vad_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    trust_repo=True,
)
print("Silero VAD loaded.")


def _normalize(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim > 1:
        frame = frame.mean(axis=1)
    if np.issubdtype(frame.dtype, np.integer):
        frame = frame.astype(np.float32) / np.iinfo(frame.dtype).max
    else:
        frame = frame.astype(np.float32)
    return frame


def _vad_confidence(frame: np.ndarray) -> float:
    audio = _normalize(frame)
    tensor = torch.from_numpy(audio)
    with torch.no_grad():
        return _vad_model(tensor, SAMPLE_RATE).item()


def _flush(speech_buffer: list):
    """Run ASR and push result to translation queue."""
    text = transcribe_segment(speech_buffer)
    if text:
        state["final_transcript"] = text
        try:
            asr_to_trans_queue.put_nowait({"text": text})
        except Exception:
            pass
    else:
        state["final_transcript"] = ""
    state["partial_transcript"] = ""


def run_segmenter():
    speech_buffer  = []
    silence_start  = None
    speech_start   = None
    speaking       = False

    while True:
        frame = audio_queue.get()
        normed = _normalize(frame)

        # Energy for dashboard meter only
        energy = float(np.sqrt(np.mean(normed ** 2)))
        state["energy"] = energy
        state["frames"] += 1

        now        = time.time()
        confidence = _vad_confidence(normed)
        is_speech  = confidence >= VAD_THRESHOLD

        if not speaking:
            if is_speech:
                speaking      = True
                speech_start  = now
                silence_start = None
                speech_buffer = [frame]
                state["_e2e_start"]         = now
                state["speech_active"]      = True
                state["partial_transcript"] = "Listening..."
                print(f"🎙️ Speech detected! (Confidence: {confidence:.2f})")
            else:
                state["speech_active"] = False

        else:
            speech_buffer.append(frame)
            state["speech_active"]      = True
            state["partial_transcript"] = "Listening..."

            if not is_speech:
                if silence_start is None:
                    silence_start = now
                elif now - silence_start >= SILENCE_DURATION:
                    duration = now - speech_start
                    print(f"Speech ended. Handing off to ASR worker...")
                    if duration >= MIN_SPEECH_DURATION:
                        _flush(speech_buffer)
                    # reset
                    speech_buffer  = []
                    speaking       = False
                    silence_start  = None
                    speech_start   = None
                    state["speech_active"]      = False
                    state["partial_transcript"] = ""
            else:
                silence_start = None

            # Force flush at max duration
            if speech_start and (now - speech_start) >= MAX_SPEECH_DURATION:
                print("Max speech duration. Force flush.")
                _flush(speech_buffer)
                speech_buffer  = []
                speaking       = False
                silence_start  = None
                speech_start   = None
                state["speech_active"]      = False
                state["partial_transcript"] = ""


def start_segmenter():
    threading.Thread(target=run_segmenter, daemon=True).start()