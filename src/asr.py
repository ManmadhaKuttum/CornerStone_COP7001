import time
import numpy as np
from faster_whisper import WhisperModel

from config import ASR_MODEL_SIZE, DEVICE, COMPUTE_TYPE, SOURCE_LANG, ASR_BEAM_SIZE
from state import state

print(f"[asr] Loading faster-whisper '{ASR_MODEL_SIZE}' on {DEVICE}/{COMPUTE_TYPE}...")
model = WhisperModel(ASR_MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
print("[asr] Model ready.")


def preprocess_audio(audio_frames: list) -> np.ndarray | None:
    if not audio_frames:
        return None
    audio = np.concatenate(audio_frames, axis=0)
    if audio.ndim > 1:
        audio = audio.squeeze()
    return np.ascontiguousarray(audio.astype("float32"))


def transcribe_segment(audio_frames: list) -> str:
    audio = preprocess_audio(audio_frames)
    if audio is None or len(audio) == 0:
        return ""

    state["asr_status"] = "processing"
    t0 = time.time()

    segments, info = model.transcribe(
        audio,
        language=SOURCE_LANG,
        task="transcribe",
        beam_size=ASR_BEAM_SIZE,
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=False,
        # ── Fixes तो तो तो / बच्च्च hallucinations ──
        no_speech_threshold=0.6,         # drop if Whisper says no speech
        log_prob_threshold=-1.0,         # drop very low confidence
        compression_ratio_threshold=2.4, # drop repetitive output
        # ── Internal VAD as second pass ──────────────
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 300,
            "threshold": 0.45,
        },
    )

    text_parts = [seg.text.strip() for seg in segments if seg.text.strip()]
    final_text = " ".join(text_parts).strip()

    state["asr_latency_ms"]   = round((time.time() - t0) * 1000, 2)
    state["asr_status"]       = "idle"
    state["total_words_asr"] += len(final_text.split()) if final_text else 0

    print(f"[asr] lang={info.language}({info.language_probability:.2f})  "
          f"lat={state['asr_latency_ms']}ms  text={final_text}")
    return final_text