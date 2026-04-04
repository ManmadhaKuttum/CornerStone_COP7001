import queue
import threading
import time
import os
import torch
import soundfile as sf
from transformers import VitsModel, AutoTokenizer

from translation import tts_text_queue
from state import state

_tts_model = None
_tts_tokenizer = None

def _load_tts():
    global _tts_model, _tts_tokenizer
    try:
        print("Loading Telugu TTS Model (MMS)...")
        model_id = "facebook/mms-tts-tel"
        _tts_tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Force model strictly to CPU to prevent memory allocator crashes
        _tts_model = VitsModel.from_pretrained(model_id).to("cpu")
        state["tts_status"] = "ready (MMS-TTS)"
    except Exception as e:
        print(f"⚠️ TTS failed to load: {e}")
        state["tts_status"] = "unavailable"
def _synthesize_and_play(text: str) -> int:
    t0 = time.time()
    
    # --- CRASH PREVENTION 1: Clean the text of AI garbage ---
    text = text.replace("<unk>", "").strip()
    
    # If the text is empty after cleaning, abort safely
    if not text:
        return 0

    print(f"\n🔊 SYNTHESIZING TELUGU AUDIO: {text}\n")
    
    try:
        inputs = _tts_tokenizer(text, return_tensors="pt")
        
        # --- CRASH PREVENTION 2: Check for valid tokens ---
        # If the tokenizer couldn't find any valid Telugu letters, abort
        if inputs["input_ids"].shape[1] <= 2: 
            print("⚠️ Skipping audio: No valid Telugu characters found.")
            return 0

        with torch.no_grad():
            output = _tts_model(**inputs).waveform
            
        sample_rate = _tts_model.config.sampling_rate
        audio_data = output.squeeze().cpu().numpy()
        
        temp_file = "temp_output.wav"
        import soundfile as sf
        sf.write(temp_file, audio_data, sample_rate)
        
        import os
        if os.uname().sysname == 'Darwin':
            os.system(f"afplay {temp_file}")
        else:
            os.system(f"aplay {temp_file}")
            
    except Exception as e:
        print(f"Audio playback error: {e}")
        
    latency = int((time.time() - t0) * 1000)
    return latency

def run_tts():
    # Model is already loaded by main.py, so we just loop
    while True:
        try:
            text = tts_text_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        if not text:
            continue

        state["tts_status"] = "speaking"
        latency = _synthesize_and_play(text)
        state["tts_latency_ms"] = latency
        state["tts_status"] = "idle"

def start_tts():
    t = threading.Thread(target=run_tts, daemon=True)
    t.start()
    return t