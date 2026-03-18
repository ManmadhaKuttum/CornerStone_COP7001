import time
import numpy as np
from faster_whisper import WhisperModel

from config import ASR_MODEL_SIZE, DEVICE, COMPUTE_TYPE, SOURCE_LANG
from state import state

model = WhisperModel(
    ASR_MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE
)


def preprocess_audio(audio_frames):
    if not audio_frames:
        return None

    audio = np.concatenate(audio_frames, axis=0)

    if audio.ndim > 1:
        audio = audio.squeeze()

    audio = audio.astype("float32")
    audio = np.ascontiguousarray(audio)
    return audio


def transcribe_segment(audio_frames):
    audio = preprocess_audio(audio_frames)
    if audio is None or len(audio) == 0:
        return ""

    start_time = time.time()

    segments, info = model.transcribe(
        audio,
        language=SOURCE_LANG,
        task="transcribe",
        beam_size=1,
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=False
    )

    text_parts = []
    for seg in segments:
        seg_text = seg.text.strip()
        if seg_text:
            text_parts.append(seg_text)

    final_text = " ".join(text_parts).strip()
    state["asr_latency_ms"] = round((time.time() - start_time) * 1000, 2)

    print("Detected language:", info.language, "prob:", info.language_probability)
    print("Transcript:", final_text)

    return final_text