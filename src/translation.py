import os
import re
import queue
import threading
import time
import ctranslate2
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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
_hf_model   = None
_tok        = None
_translator_device = "cpu"
_backend    = None

def _cached_hf_snapshot(model_id: str) -> str | None:
    repo_dir = os.path.join(HF_HOME, f"models--{model_id.replace('/', '--')}")
    ref_path = os.path.join(repo_dir, "refs", "main")
    try:
        with open(ref_path, "r", encoding="utf-8") as fh:
            revision = fh.read().strip()
        snapshot_dir = os.path.join(repo_dir, "snapshots", revision)
        if os.path.isdir(snapshot_dir):
            return snapshot_dir
    except OSError:
        pass
    return None

def _load_translation():
    global _translator, _hf_model, _tok, _translator_device, _backend
    print("⏳ Loading CTranslate2 NLLB engine...")
    tokenizer_src = _cached_hf_snapshot(TRANSLATION_TOKENIZER_ID) or TRANSLATION_TOKENIZER_ID
    _tok = AutoTokenizer.from_pretrained(
        tokenizer_src,
        cache_dir=HF_HOME,
        local_files_only=OFFLINE_MODE or os.path.isdir(tokenizer_src),
    )

    ct2_path = os.path.join(PROJECT_ROOT, "pretrained", "nllb-200-distilled-600M-int8")
    if os.path.isdir(ct2_path):
        _translator = ctranslate2.Translator(
            ct2_path, device="cpu", compute_type="int8",
            intra_threads=CT2_INTRA_THREADS,
            inter_threads=CT2_INTER_THREADS,
        )
        _hf_model = None
        _backend = "ct2"
        state["trans_backend"] = "CTranslate2-NLLB-600M"
        state["trans_status"] = "ready"
        print("  ✅ Translation loaded (CTranslate2)")
        return

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
    print("  ⚠️  CTranslate2 export missing, using cached Transformers NLLB model")
    _translator = None
    _hf_model = AutoModelForSeq2SeqLM.from_pretrained(
        tokenizer_src,
        cache_dir=HF_HOME,
        local_files_only=OFFLINE_MODE or os.path.isdir(tokenizer_src),
        low_cpu_mem_usage=True,
    ).to("cpu")
    _hf_model.eval()
    _backend = "hf"
    state["trans_status"] = "ready"
    state["trans_backend"] = "Transformers-NLLB-600M"
    print("  ✅ Translation loaded (Transformers)")

def _clean(text: str) -> str:
    """Remove <unk> tokens and collapse extra whitespace."""
    text = text.replace("<unk>", "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _translate(text: str, tag: str) -> tuple[str, int]:
    t0 = time.time()
    _tok.src_lang = SRC_LANG
    if _backend == "ct2":
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
    else:
        inputs = _tok(text, return_tensors="pt")
        forced_bos_token_id = _tok.convert_tokens_to_ids(TGT_LANG)
        max_new_tokens = min(MAX_LENGTH, max(24, int(inputs["input_ids"].shape[1]) + 12))
        if tag == "partial":
            max_new_tokens = min(max_new_tokens, 48)
        with torch.inference_mode():
            generated = _hf_model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=max_new_tokens,
                num_beams=TRANS_BEAMS,
                do_sample=False,
                use_cache=True,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        result = _clean(_tok.batch_decode(generated, skip_special_tokens=True)[0])
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
        if state["trans_status"] == "unavailable" or _tok is None:
            continue
        if _backend == "ct2" and _translator is None:
            continue
        if _backend == "hf" and _hf_model is None:
            continue

        # If queue already has pending items, skip stale partial work and keep
        # CPU available for final results.
        if tag == "partial" and not trans_text_queue.empty():
            continue

        if tag == "partial" and text == last_partial_src:
            if last_partial_tgt:
                state["partial_translation"] = last_partial_tgt
            continue

        translated, latency = _translate(text, tag)
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
