import queue
import re
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

# Short neutral prompt — just enough to anchor Devanagari script.
# Do NOT use words like "बातचीत" here: Whisper hallucinates prompt words
# on unclear/silent audio and loops them infinitely.
_HINDI_PROMPT = "नमस्ते।"

_DEVA         = re.compile(r'[\u0900-\u097F]')          # Devanagari Unicode block
_LATIN_WORD   = re.compile(r'\b[A-Za-z]{2,}\b')         # standalone Latin words

def _filter_hindi(text: str) -> str:
    """
    Remove hallucinations from Whisper's Hindi output:
    1. Whole-segment rejection: if less than 50% of non-space chars are
       Devanagari the segment is likely a mis-transcription of noise/English.
    2. Word-level rejection: strip individual Latin-script words that leaked
       in (e.g. 'Kennedy', 'obviously' transcribed in Latin instead of Deva).
    """
    if not text:
        return text

    # ── Segment-level gate ────────────────────────────────────────────────────
    # Threshold = 0.25: only reject segments that are >75% non-Devanagari.
    # 0.50 was too aggressive — valid Hindi with numbers, proper nouns, or
    # code-mixed words (e.g. "iPhone लो") was being discarded incorrectly.
    chars = re.sub(r'[\s।,.\-?!\d]', '', text)
    if chars:
        deva_ratio = len(_DEVA.findall(chars)) / len(chars)
        if deva_ratio < 0.25:
            return ""   # >75% non-Devanagari — hallucinated English, discard

    # ── Word-level: strip Latin words ─────────────────────────────────────────
    text = _LATIN_WORD.sub('', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


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
        # Prevent hallucination propagation: each segment is decoded
        # independently instead of conditioning on previous segment text.
        # This is the single biggest fix for "बातचीत बातचीत" loops.
        condition_on_previous_text=False,
    )
    # Consume the generator fully — faster-whisper is lazy.
    raw = " ".join(s.text for s in segments).strip()
    return _filter_hindi(raw)

def run_asr():
    last_partial_enqueued = ""

    while True:
        try:
            tag, audio, onset_time = asr_queue.get(timeout=1.0)
        except (queue.Empty, ValueError):
            continue

        if _model is None or state["asr_status"] == "unavailable":
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
            # ── Accuracy-focused final path (BEAM_SIZE from config) ──────────
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
                # Queue is full of stale partials. Drain partials-only to make
                # room. Finals already in the queue are kept — never dropped.
                for _ in range(trans_text_queue.maxsize):
                    try:
                        head = trans_text_queue.get_nowait()
                        if head[0] == "final":
                            trans_text_queue.put_nowait(head)  # put back
                            break
                    except queue.Empty:
                        break
                # Block up to 2 s — finals must reach the translation thread.
                try:
                    trans_text_queue.put(("final", text, onset_time), timeout=2.0)
                except queue.Full:
                    # Last-resort path: block until queue drains. This prevents
                    # silent loss of finalized utterances.
                    print("⚠️  Translation queue saturated, blocking to preserve final ASR output...")
                    trans_text_queue.put(("final", text, onset_time))

def start_asr():
    t = threading.Thread(target=run_asr, daemon=True, name="asr")
    t.start()
    return t
