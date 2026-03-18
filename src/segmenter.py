import time
import numpy as np
import threading

from audio import audio_queue
from state import state
from config import (
    START_THRESHOLD,
    STOP_THRESHOLD,
    SILENCE_DURATION,
    MAX_SPEECH_DURATION,
    MIN_SPEECH_DURATION,
)
from asr import transcribe_segment


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)

    if frame.ndim > 1:
        frame = frame.mean(axis=1)

    if np.issubdtype(frame.dtype, np.integer):
        max_val = np.iinfo(frame.dtype).max
        frame = frame.astype(np.float32) / max_val
    else:
        frame = frame.astype(np.float32)

    return frame


def frame_energy(frame: np.ndarray) -> float:
    frame = normalize_frame(frame)
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(frame ** 2)))


def start_segmenter():
    thread = threading.Thread(target=run_segmenter, daemon=True)
    thread.start()


def run_segmenter():
    speech_buffer = []
    silence_start = None
    speech_start = None
    speaking = False

    while True:
        frame = audio_queue.get()
        energy = frame_energy(frame)

        state["energy"] = energy
        state["frames"] += 1

        # print(f"energy={energy:.6f}, speaking={speaking}")

        now = time.time()

        if not speaking:
            if energy >= START_THRESHOLD:
                speaking = True
                speech_start = now
                silence_start = None
                speech_buffer = [frame]

                state["speech_active"] = True
                state["partial_transcript"] = "Listening..."
                print("Speech started")
            else:
                state["speech_active"] = False

        else:
            speech_buffer.append(frame)
            state["speech_active"] = True
            state["partial_transcript"] = "Listening..."

            if energy < STOP_THRESHOLD:
                if silence_start is None:
                    silence_start = now
                elif now - silence_start >= SILENCE_DURATION:
                    duration = now - speech_start if speech_start else 0.0
                    print("Speech ended. Transcribing...")

                    if duration >= MIN_SPEECH_DURATION and speech_buffer:
                        text = transcribe_segment(speech_buffer)
                        state["final_transcript"] = text
                    else:
                        state["final_transcript"] = ""

                    speech_buffer = []
                    speaking = False
                    silence_start = None
                    speech_start = None
                    state["speech_active"] = False
                    state["partial_transcript"] = ""

            else:
                silence_start = None

            if speech_start is not None and (now - speech_start) >= MAX_SPEECH_DURATION:
                print("Max speech duration reached. Transcribing...")

                text = transcribe_segment(speech_buffer) if speech_buffer else ""
                state["final_transcript"] = text

                speech_buffer = []
                speaking = False
                silence_start = None
                speech_start = None
                state["speech_active"] = False
                state["partial_transcript"] = ""