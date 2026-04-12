import os
import re
import queue
import threading
import time

import numpy as np
import torch
from transformers import VitsModel, AutoTokenizer

from config import TTS_MODEL_ID, OFFLINE_MODE, HF_HOME, TTS_DEVICE
from translation import tts_text_queue
from state import state

_tts_model     = None
_tts_tokenizer = None
_tts_device    = "cpu"
_sd = None

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

def _sounddevice():
    global _sd
    if _sd is None:
        import sounddevice as sd
        _sd = sd
    return _sd

def _load_tts():
    global _tts_model, _tts_tokenizer, _tts_device
    try:
        print(f"⏳ Loading TTS model ({TTS_MODEL_ID})...")
        model_src = _cached_hf_snapshot(TTS_MODEL_ID) or TTS_MODEL_ID
        _tts_tokenizer = AutoTokenizer.from_pretrained(
            model_src, cache_dir=HF_HOME, local_files_only=OFFLINE_MODE or os.path.isdir(model_src))

        target = TTS_DEVICE
        try:
            _tts_model = VitsModel.from_pretrained(
                model_src, cache_dir=HF_HOME, local_files_only=OFFLINE_MODE or os.path.isdir(model_src)).to(target)
            _tts_device = target
        except Exception:
            if target != "cpu":
                print(f"⚠️  TTS device '{target}' unavailable, falling back to CPU")
                _tts_model = VitsModel.from_pretrained(
                    model_src, cache_dir=HF_HOME, local_files_only=OFFLINE_MODE or os.path.isdir(model_src)).to("cpu")
                _tts_device = "cpu"
            else:
                raise
        _tts_model.eval()    # lock BatchNorm / Dropout to eval mode permanently
        state["tts_status"]  = "idle"
        state["tts_backend"] = f"MMS-TTS-Telugu ({_tts_device})"
        print("  ✅ TTS loaded")
        # Pre-warm: first inference is always slow (JIT compilation + memory alloc).
        # Run a silent dummy pass so the first real utterance is not penalised.
        _prewarm_tts()
    except Exception as e:
        state["tts_status"] = "unavailable"
        print(f"⚠️  TTS failed to load: {e}")

def _prewarm_tts():
    """Synthesise (but don't play) a short Telugu word at startup."""
    try:
        dummy = "నమస్కారం"
        inp = _tts_tokenizer(dummy, return_tensors="pt")
        inp = {k: v.to(_tts_device) for k, v in inp.items()}
        with torch.inference_mode():
            _ = _tts_model(**inp).waveform   # warm up model weights only
        print("  ✅ TTS pre-warmed")
    except Exception:
        pass   # non-fatal

def _clean(text: str) -> str:
    text = text.replace("<unk>", "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _synthesize_and_play(text: str) -> tuple[int, int]:
    """Returns (synthesis_ms, playback_ms) separately so we can report them."""
    text = _clean(text)
    if not text:
        return 0, 0

    try:
        inputs = _tts_tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(_tts_device) for k, v in inputs.items()}
        if inputs["input_ids"].shape[1] <= 2:
            return 0, 0

        # ── Synthesis ────────────────────────────────────────────────────────
        t_synth = time.time()
        with torch.inference_mode():   # faster than no_grad — skips autograd machinery
            waveform = _tts_model(**inputs).waveform
        synth_ms = int((time.time() - t_synth) * 1000)

        # ── Playback ─────────────────────────────────────────────────────────
        audio = waveform.squeeze().cpu().numpy().astype(np.float32)
        sr    = _tts_model.config.sampling_rate
        t_play = time.time()
        # Use OutputStream.write() instead of sd.play() — the convenience
        # sd.play() uses a global internal stream that conflicts with the
        # concurrent sd.InputStream running for mic capture on Mac PortAudio.
        sd = _sounddevice()
        with sd.OutputStream(samplerate=sr, channels=1, dtype="float32") as out:
            out.write(audio.reshape(-1, 1))
        play_ms = int((time.time() - t_play) * 1000)

        return synth_ms, play_ms

    except Exception as e:
        print(f"⚠️  TTS playback error: {e}")
        return 0, 0

_MAX_ONSET_AGE_S = 15.0   # discard e2e measurement if utterance queued > 15 s ago

def run_tts():
    while True:
        try:
            text, onset_time = tts_text_queue.get(timeout=1.0)
        except (queue.Empty, ValueError):
            continue

        if not text or state["tts_status"] == "unavailable":
            continue

        # Drop stale items — if an utterance has waited more than 15 s in the
        # queue the E2E number would be meaningless and speaker already missed it.
        if onset_time and (time.time() - onset_time) > _MAX_ONSET_AGE_S:
            continue

        state["tts_status"] = "speaking"
        synth_ms, play_ms = _synthesize_and_play(text)

        # Report synthesis-only latency (excludes audio playback duration which
        # is fixed by the length of the translated sentence — not a system cost).
        state["tts_latency_ms"] = synth_ms
        state["tts_play_ms"]    = play_ms

        # Dashboard E2E should reflect pipeline processing time, not the
        # variable duration of audio playback. Playback length is tracked
        # separately in tts_play_ms.
        state["e2e_latency_ms"] = (
            state.get("asr_latency_ms", 0)
            + state.get("trans_latency_ms", 0)
            + synth_ms
        )

        state["tts_status"] = "idle"

def start_tts():
    t = threading.Thread(target=run_tts, daemon=True, name="tts")
    t.start()
    return t
