"""
translation.py — Phase 3
IndicTrans2 (AI4Bharat / IIT Madras) — Hindi → Telugu, fully offline.
Install: pip install transformers sentencepiece torch
Model downloads on first run: ai4bharat/indictrans2-indic-indic-dist-200M (~800MB)
"""

from email.mime import text
import time
import queue
import threading

from torch import device

from state import state
from config import INDICTRANS_SRC, INDICTRANS_TGT, TRANS_DEVICE

asr_to_trans_queue = queue.Queue(maxsize=50)

def _load_model():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

    name = "facebook/nllb-200-distilled-600M"
    print(f"[trans] Loading NLLB-600M on {TRANS_DEVICE}...")
    tok   = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    safe_device = "cpu" if TRANS_DEVICE == "mps" else TRANS_DEVICE
    model = model.to(safe_device)
    model.eval()
    print(f"[trans] NLLB ready on {safe_device}.")
    return tok, model, safe_device


def _translate(text: str, tok, model, device: str) -> str:
    import torch
    inputs = tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    target_lang_id = tok.lang_code_to_id["tel_Telu"]
    # Set source language for NLLB
    tok.src_lang = "hin_Deva"
    inputs = tok(text, return_tensors="pt",
                truncation=True, max_length=512).to(device)
    with torch.no_grad():
        
        out = model.generate(
            **inputs,
            forced_bos_token_id=target_lang_id,
            num_beams=4,
            max_length=512,
        )
    return tok.decode(out[0], skip_special_tokens=True)


def _worker(tok, model, device: str):
    state["trans_status"]  = "idle"
    state["trans_backend"] = f"NLLB-600M ({device})"

    while True:
        try:
            item = asr_to_trans_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        text = item.get("text", "").strip()
        if not text:
            continue

        state["trans_status"] = "processing"
        t0 = time.time()
        try:
            translated = _translate(text, tok, model, device)
            lat = round((time.time() - t0) * 1000, 2)

            state["final_translation"]   = translated
            state["partial_translation"] = ""
            state["trans_latency_ms"]    = lat
            state["total_words_trans"]  += len(translated.split())

            if state["_e2e_start"] > 0:
                state["e2e_latency_ms"] = round(
                    (time.time() - state["_e2e_start"]) * 1000, 2
                )

            print(f"[trans] lat={lat}ms  text={translated}")
        except Exception as e:
            print(f"[trans] Error: {e}")

        state["trans_status"] = "idle"


def start_translation():
    try:
        tok, model, device = _load_model()
        threading.Thread(target=_worker, args=(tok, model, device), daemon=True).start()
    except Exception as e:
        print(f"[trans] Failed to load: {e}")
        state["trans_status"]  = "unavailable"
        state["trans_backend"] = "unavailable"