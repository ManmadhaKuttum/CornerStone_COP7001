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
trans_text_queue = queue.Queue(maxsize=20)

_model = None

_HINDI_PROMPT = "नमस्ते।"

_DEVA         = re.compile(r'[\u0900-\u097F]')         
_LATIN_WORD   = re.compile(r'\b[A-Za-z]{2,}\b')        

def _filter_hindi(text: str) -> str:
    
    if not text:
        return text

    chars = re.sub(r'[\s।,.\-?!\d]', '', text)
    if chars:
        deva_ratio = len(_DEVA.findall(chars)) / len(chars)
        if deva_ratio < 0.25:
            return ""   
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
        condition_on_previous_text=False,
    )

    raw = " ".join(s.text for s in segments).strip()
    return _filter_hindi(raw)

def run_asr():
    last_partial_enqueued = ""

    while True:
        try:
            session_id, tag, audio, onset_time = asr_queue.get(timeout=1.0)
        except (queue.Empty, ValueError):
            continue

        if _model is None or state["asr_status"] == "unavailable":
            continue
        if session_id != state["session_id"]:
            continue

        if tag == "partial":
            text = _run_whisper(audio, beam_size=PARTIAL_BEAM_SIZE)
            if text:
                state["partial_transcript"] = text
                if text != last_partial_enqueued:
                    try:
                        trans_text_queue.put_nowait((session_id, "partial", text, onset_time))
                        last_partial_enqueued = text
                    except queue.Full:
                        pass

        elif tag == "final":
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
                trans_text_queue.put_nowait((session_id, "final", text, onset_time))
            except queue.Full:
                for _ in range(trans_text_queue.maxsize):
                    try:
                        head = trans_text_queue.get_nowait()
                        if head[1] == "final":
                            trans_text_queue.put_nowait(head)  
                            break
                    except queue.Empty:
                        break
                try:
                    trans_text_queue.put((session_id, "final", text, onset_time), timeout=2.0)
                except queue.Full:
                    print("⚠️  Translation queue saturated, blocking to preserve final ASR output...")
                    trans_text_queue.put((session_id, "final", text, onset_time))

def start_asr():
    t = threading.Thread(target=run_asr, daemon=True, name="asr")
    t.start()
    return t
