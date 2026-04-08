import queue
import threading
import time
import os

from faster_whisper import WhisperModel

from config import (
    ASR_MODEL, DEVICE, COMPUTE_TYPE, ASR_LANGUAGE,
    BEAM_SIZE, NO_SPEECH_THRESHOLD,
    COMPRESSION_RATIO_THRESHOLD, LOG_PROB_THRESHOLD,
    OFFLINE_MODE,
)
from segmenter import asr_to_trans_queue
from state import state

# Queue carries tuples: ("partial"|"final", text)
trans_text_queue = queue.Queue(maxsize=20)

_model = None

def _load_asr():
    global _model
    if OFFLINE_MODE:
        if not (os.path.isdir(ASR_MODEL) or os.path.isfile(ASR_MODEL)):
            raise RuntimeError(
                "Offline mode enabled but ASR_MODEL is not a local path. "
                "Set WHISPER_MODEL_PATH to a downloaded Faster-Whisper model directory."
            )
    _model = WhisperModel(
        ASR_MODEL,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )
    state["asr_status"] = "ready"
def _transcribe(audio, onset_time):
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
    partial_chunks = []
    last_partial = ""
    last_emit = 0.0
    min_emit_interval = 0.3  # seconds, avoid spamming dashboard/translation

    for s in segments:
        seg_text = (s.text or "").strip()
        if not seg_text:
            continue

        partial_chunks.append(seg_text)
        partial_text = " ".join(partial_chunks).strip()
        if not partial_text or partial_text == last_partial:
            continue

        now = time.time()
        if now - last_emit >= min_emit_interval:
            state["partial_transcript"] = partial_text
            try:
                trans_text_queue.put_nowait(("partial", partial_text, onset_time))
            except queue.Full:
                pass
            last_partial = partial_text
            last_emit = now

    text = " ".join(partial_chunks).strip()
    latency = int((time.time() - t0) * 1000)
    return text, latency

def run_asr():
    
    while True:
        try:
            item = asr_to_trans_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        if isinstance(item, tuple) and len(item) == 3:
            tag, audio, onset_time = item
        elif isinstance(item, tuple) and len(item) == 2:
            tag, audio = item
            onset_time = None
        else:
            continue

        if tag != "asr":
            continue

        # Reset partials at the start of a new utterance
        state["partial_transcript"] = ""
        state["partial_translation"] = ""

        text, latency = _transcribe(audio, onset_time)
        if not text:
            continue

        state["final_transcript"] = state["final_transcript"] + " " + text if state["final_transcript"] else text
        state["asr_latency_ms"] = latency
        state["total_words_asr"] += len(text.split())
        state["partial_transcript"] = ""

        try:
            trans_text_queue.put_nowait(("final", text, onset_time))
        except queue.Full:
            # Make room for final output by dropping one queued item
            try:
                trans_text_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                trans_text_queue.put_nowait(("final", text, onset_time))
            except queue.Full:
                pass

def start_asr():
    t = threading.Thread(target=run_asr, daemon=True)
    t.start()
    return t
