import queue
import os
import threading
import time
import ctranslate2

from config import SRC_LANG, TGT_LANG, MAX_LENGTH, OFFLINE_MODE, HF_HOME, TRANSLATION_TOKENIZER_ID
from transformers import AutoTokenizer
from asr import trans_text_queue
from state import state

# --- THE FIX: We define the queue physically here, and remove the bad import ---
tts_text_queue = queue.Queue(maxsize=20)

_translator = None
_tok = None

def _load_translation():
    global _translator, _tok
    print("⏳ Loading local CTranslate2 NLLB Engine...")
    
    # Load the tokenizer
    model_name = TRANSLATION_TOKENIZER_ID
    _tok = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=HF_HOME,
        local_files_only=OFFLINE_MODE,
    )
    
    # Point directly to the folder you just compiled! (relative to project root)
    project_root = os.path.dirname(os.path.dirname(__file__))
    ct2_model_path = os.path.join(project_root, "pretrained/nllb-200-distilled-600M-int8") 

    if not os.path.isdir(ct2_model_path):
        raise RuntimeError(
            "CTranslate2 NLLB model not found at "
            f"{ct2_model_path}. Run the converter step once."
        )
    
    # Load the C++ engine
    _translator = ctranslate2.Translator(
        ct2_model_path, 
        device="cpu", 
        compute_type="int8"
    )
    state["trans_status"] = "ready"

def _translate(text: str) -> tuple[str, int]:
    t0 = time.time()
    
    _tok.src_lang = SRC_LANG
    source = _tok.convert_ids_to_tokens(_tok.encode(text))
    
    # CTranslate2 native execution (Greedy Search by default)
    results = _translator.translate_batch(
        [source], 
        target_prefix=[[TGT_LANG]],
        max_decoding_length=MAX_LENGTH
    )
    
    target = results[0].hypotheses[0][1:]
    result = _tok.decode(_tok.convert_tokens_to_ids(target))
    
    latency = int((time.time() - t0) * 1000)
    return result, latency

def run_translation():
    while True:
        try:
            item = trans_text_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        if isinstance(item, tuple) and len(item) == 3:
            kind, text, onset_time = item
        elif isinstance(item, tuple) and len(item) == 2:
            kind, text = item
            onset_time = None
        else:
            kind, text, onset_time = "final", item, None

        if not text:
            continue

        translated, latency = _translate(text)
        if not translated:
            continue

        if kind == "partial":
            state["partial_translation"] = translated
            continue

        # RTF Calculation
        total_processing_time_sec = (state["asr_latency_ms"] + latency) / 1000.0
        audio_length = state.get("last_audio_duration", 1.0) 
        
        if audio_length > 0:
            current_rtf = total_processing_time_sec / audio_length
            state["rtf"] = round(current_rtf, 2)

        state["final_translation"] = (
            state["final_translation"] + " " + translated
            if state["final_translation"] else translated
        )
        state["partial_translation"] = ""
        state["trans_latency_ms"] = latency
        state["total_words_trans"] += len(translated.split())

        try:
            tts_text_queue.put_nowait((translated, onset_time))
        except queue.Full:
            pass
def start_translation():
    t = threading.Thread(target=run_translation, daemon=True)
    t.start()
    return t        
