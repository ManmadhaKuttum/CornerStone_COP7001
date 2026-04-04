import queue
import threading
import time
import ctranslate2
from transformers import AutoTokenizer

from config import SRC_LANG, TGT_LANG, MAX_LENGTH
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
    model_name = "facebook/nllb-200-distilled-600M"
    _tok = AutoTokenizer.from_pretrained(model_name)
    
    # Point directly to the folder you just compiled!
    ct2_model_path = "pretrained/nllb-200-distilled-600M-int8" 
    
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
            text = trans_text_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        if not text:
            continue

        translated, latency = _translate(text)
        if not translated:
            continue

        onset = state.get("speech_onset_time")
        e2e = int((time.time() - onset) * 1000) if onset else 0

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
        state["trans_latency_ms"] = latency
        state["e2e_latency_ms"] = e2e
        state["total_words_trans"] += len(translated.split())

        try:
            tts_text_queue.put_nowait(translated)
        except queue.Full:
            pass
def start_translation():
    t = threading.Thread(target=run_translation, daemon=True)
    t.start()
    return t        