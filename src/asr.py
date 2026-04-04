import queue
import threading
import time

from faster_whisper import WhisperModel

from config import (
    ASR_MODEL, DEVICE, COMPUTE_TYPE, ASR_LANGUAGE,
    BEAM_SIZE, NO_SPEECH_THRESHOLD,
    COMPRESSION_RATIO_THRESHOLD, LOG_PROB_THRESHOLD,
)
from segmenter import asr_to_trans_queue
from state import state

trans_text_queue = queue.Queue(maxsize=20)

_model = None

def _load_asr():
    global _model
    _model = WhisperModel(
        ASR_MODEL,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )
    state["asr_status"] = "ready"
def _transcribe(audio):
    t0 = time.time()
    segments, info = _model.transcribe(
        audio,
        language=ASR_LANGUAGE,
        task="transcribe", # <--- FIX 1: Explicitly tell it NOT to translate to English
        initial_prompt="नमस्ते, आप कैसे हैं? यह हिंदी है।", # <--- FIX 2: Force Devanagari script via prompt
        beam_size=BEAM_SIZE,
        no_speech_threshold=NO_SPEECH_THRESHOLD,
        compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
        log_prob_threshold=LOG_PROB_THRESHOLD,
        vad_filter=True,
    )
    text = " ".join(s.text for s in segments).strip()
    latency = int((time.time() - t0) * 1000)
    return text, latency

def run_asr():
    
    while True:
        try:
            tag, audio = asr_to_trans_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        if tag != "asr":
            continue

        text, latency = _transcribe(audio)
        if not text:
            continue

        state["final_transcript"] = state["final_transcript"] + " " + text if state["final_transcript"] else text
        state["asr_latency_ms"] = latency
        state["total_words_asr"] += len(text.split())
        state["partial_transcript"] = ""

        try:
            trans_text_queue.put_nowait(text)
        except queue.Full:
            pass

def start_asr():
    t = threading.Thread(target=run_asr, daemon=True)
    t.start()
    return t