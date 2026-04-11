import queue
import threading
import time

from faster_whisper import WhisperModel

from config import (
    ASR_MODEL, DEVICE, COMPUTE_TYPE, ASR_LANGUAGE,
    BEAM_SIZE, PARTIAL_BEAM_SIZE,
    NO_SPEECH_THRESHOLD, COMPRESSION_RATIO_THRESHOLD, LOG_PROB_THRESHOLD,
    OFFLINE_MODE,
)
from segmenter import asr_queue
from state import state

# Output queue to translation thread.
# Protocol: ("partial", text, onset_time) | ("final", text, onset_time)
trans_text_queue = queue.Queue(maxsize=20)

_model = None

# Forces Devanagari script output — prevents Whisper romanizing Hindi words.
_HINDI_PROMPT = "नमस्ते, यह हिंदी में बातचीत है।"

def _load_asr():
    global _model
    _model = WhisperModel(ASR_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
    state["asr_status"] = "ready"
    print(f"  ✅ ASR loaded — model: {ASR_MODEL!r}, device: {DEVICE}, compute: {COMPUTE_TYPE}")

def _run_whisper(audio, beam_size: int) -> str:
    segments, _ = _model.transcribe(
        audio,
        language=ASR_LANGUAGE,
        task="transcribe",
        initial_prompt=_HINDI_PROMPT,
        beam_size=beam_size,
        no_speech_threshold=NO_SPEECH_THRESHOLD,
        compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
        log_prob_threshold=LOG_PROB_THRESHOLD,
        vad_filter=True,
    )
    # Consume the generator fully — faster-whisper is lazy.
    return " ".join(s.text for s in segments).strip()

def run_asr():
    last_partial_enqueued = ""

    while True:
        try:
            tag, audio, onset_time = asr_queue.get(timeout=1.0)
        except (queue.Empty, ValueError):
            continue

        if tag == "partial":
            # ── Fast path: beam=1 ────────────────────────────────────────────
            # Goal: show live Hindi text on dashboard while user is still talking.
            # Accuracy is secondary. Sends best-effort partial updates to
            # translation; translation will skip stale partials under load.
            text = _run_whisper(audio, beam_size=PARTIAL_BEAM_SIZE)
            if text:
                state["partial_transcript"] = text
                if text != last_partial_enqueued:
                    try:
                        trans_text_queue.put_nowait(("partial", text, onset_time))
                        last_partial_enqueued = text
                    except queue.Full:
                        # Drop partial update if queue is saturated; final output
                        # is always more important than partial display updates.
                        pass

        elif tag == "final":
            # ── Accuracy-focused path: beam=4 ────────────────────────────────
            # Full utterance after silence. This is the ONLY path that feeds
            # translation and TTS. Never reduce beam_size here — wrong Hindi
            # transcription corrupts the entire downstream pipeline.
            t0   = time.time()
            text = _run_whisper(audio, beam_size=BEAM_SIZE)
            latency = int((time.time() - t0) * 1000)

            if not text:
                state["partial_transcript"] = ""
                continue

            state["final_transcript"]   = (
                state["final_transcript"] + " " + text
                if state["final_transcript"] else text
            )
            state["partial_transcript"] = ""
            last_partial_enqueued = ""
            state["asr_latency_ms"]     = latency
            state["total_words_asr"]   += len(text.split())

            try:
                trans_text_queue.put_nowait(("final", text, onset_time))
            except queue.Full:
                # Drop the oldest queued item to make room for the final result.
                try:
                    trans_text_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    trans_text_queue.put_nowait(("final", text, onset_time))
                except queue.Full:
                    pass

def start_asr():
    t = threading.Thread(target=run_asr, daemon=True, name="asr")
    t.start()
    return t
