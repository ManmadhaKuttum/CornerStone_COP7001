import os
import re
import queue
import threading
import time
import ctranslate2
from transformers import AutoTokenizer

from config import (
    SRC_LANG, TGT_LANG, MAX_LENGTH,
    OFFLINE_MODE, HF_HOME, TRANSLATION_TOKENIZER_ID,
    PROJECT_ROOT, TRANS_BEAMS, CT2_INTRA_THREADS, CT2_INTER_THREADS,
    TRANSLATION_DEVICE, TRANSLATION_COMPUTE_TYPE,
)
from asr import trans_text_queue
from state import state

# Output queue to TTS thread.
# Protocol: (text_str, onset_time_float)
# Keep small to avoid long speaker backlog and stale e2e numbers.
tts_text_queue = queue.Queue(maxsize=3)

_translator = None
_tok        = None
_translator_device = "cpu"

def _load_translation():
    global _translator, _tok, _translator_device
    print("⏳ Loading CTranslate2 NLLB engine...")

    _tok = AutoTokenizer.from_pretrained(
        TRANSLATION_TOKENIZER_ID,
        cache_dir=HF_HOME,
        local_files_only=OFFLINE_MODE,
    )

    ct2_path = os.path.join(PROJECT_ROOT, "pretrained", "nllb-200-distilled-600M-int8")
    if not os.path.isdir(ct2_path):
        raise RuntimeError(
            f"CTranslate2 NLLB model not found at {ct2_path}.\n"
            "Run once:\n"
            "  ct2-transformers-converter --model facebook/nllb-200-distilled-600M "
            "--output_dir pretrained/nllb-200-distilled-600M-int8 --quantization int8"
        )

    translator_kwargs = {
        "device": TRANSLATION_DEVICE,
        "compute_type": TRANSLATION_COMPUTE_TYPE,
    }
    if TRANSLATION_DEVICE == "cpu":
        translator_kwargs["intra_threads"] = CT2_INTRA_THREADS
        translator_kwargs["inter_threads"] = CT2_INTER_THREADS

    try:
        _translator = ctranslate2.Translator(ct2_path, **translator_kwargs)
        _translator_device = TRANSLATION_DEVICE
    except Exception:
        # Fallback for CUDA edge cases:
        # 1) Requested CUDA compute type not available
        # 2) Installed CTranslate2 build is CPU-only
        if TRANSLATION_DEVICE == "cuda":
            print("⚠️  CTranslate2 CUDA unavailable, falling back to CPU translation")
            _translator = ctranslate2.Translator(
                ct2_path,
                device="cpu",
                compute_type="int8",
                intra_threads=CT2_INTRA_THREADS,
                inter_threads=CT2_INTER_THREADS,
            )
            _translator_device = "cpu"
        else:
            raise

    state["trans_status"]  = "ready"
    state["trans_backend"] = f"CTranslate2-NLLB-600M ({_translator_device})"
    print("  ✅ Translation loaded")

def _clean(text: str) -> str:
    """Remove <unk> tokens and collapse extra whitespace."""
    text = text.replace("<unk>", "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _translate(text: str) -> tuple[str, int]:
    t0 = time.time()
    _tok.src_lang = SRC_LANG
    source  = _tok.convert_ids_to_tokens(_tok.encode(text))
    results = _translator.translate_batch(
        [source],
        target_prefix=[[TGT_LANG]],
        max_decoding_length=MAX_LENGTH,
        beam_size=TRANS_BEAMS,
        no_repeat_ngram_size=3,
    )
    target  = results[0].hypotheses[0][1:]
    result  = _clean(_tok.decode(_tok.convert_tokens_to_ids(target)))
    latency = int((time.time() - t0) * 1000)
    return result, latency

def run_translation():
    last_partial_src = ""
    last_partial_tgt = ""

    while True:
        try:
            tag, text, onset_time = trans_text_queue.get(timeout=1.0)
        except (queue.Empty, ValueError):
            continue

        if not text:
            continue

        # If queue already has pending items, skip stale partial work and keep
        # CPU available for final results.
        if tag == "partial" and not trans_text_queue.empty():
            continue

        if tag == "partial" and text == last_partial_src:
            if last_partial_tgt:
                state["partial_translation"] = last_partial_tgt
            continue

        translated, latency = _translate(text)
        if not translated:
            continue

        if tag == "partial":
            last_partial_src = text
            last_partial_tgt = translated
            state["partial_translation"] = translated
            continue

        # RTF = (asr inference time + translation inference time) / audio duration
        audio_dur = state.get("last_audio_duration", 1.0)
        if audio_dur > 0:
            total_proc = (state["asr_latency_ms"] + latency) / 1000.0
            state["rtf"] = round(total_proc / audio_dur, 2)

        state["final_translation"] = (
            state["final_translation"] + " " + translated
            if state["final_translation"] else translated
        )
        state["partial_translation"] = ""
        last_partial_src = ""
        last_partial_tgt = ""
        state["trans_latency_ms"]    = latency
        state["total_words_trans"]  += len(translated.split())

        try:
            tts_text_queue.put_nowait((translated, onset_time))
        except queue.Full:
            # Drop one oldest TTS item so speaker stays near real-time.
            try:
                tts_text_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                tts_text_queue.put_nowait((translated, onset_time))
            except queue.Full:
                pass

def start_translation():
    t = threading.Thread(target=run_translation, daemon=True, name="translation")
    t.start()
    return t
