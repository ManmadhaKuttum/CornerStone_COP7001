# Real-Time Speech Translation System

## Phase 2 – Hindi Speech Recognition (ASR)

This project aims to build a real-time speech-to-speech translation system for Indian languages.

The current implementation completes **Phase 2**:
Real-time **Hindi speech recognition (ASR)** from microphone input with a live monitoring dashboard.

---

## Implemented (Phase 2)

* Live microphone audio capture (streaming)
* Voice Activity Detection (VAD) for speech segmentation
* Real-time Hindi speech-to-text using Faster-Whisper
* GPU-accelerated inference (CUDA support)
* Producer–consumer pipeline using a bounded queue
* Concurrent processing (audio + segmentation + ASR + dashboard)
* Real-time web dashboard (WebSocket-based)
* Live audio energy meter
* Speech activity indicator (Speaking / Silence)
* Partial and final transcript display
* ASR latency tracking
* Stable continuous execution
* Fully offline operation (no external APIs)

---

## Architecture (Current)

Microphone → Audio Buffer → Segmenter (VAD) → ASR (Whisper) → Web Dashboard

This forms the core pipeline required for real-time speech processing.

---

## Tech Stack

* Python
* SoundDevice
* Faster-Whisper (ASR)
* FastAPI
* WebSockets
* HTML / CSS / JavaScript
* CUDA (GPU acceleration)

---

## Running the System

```bash
python src/main.py
```

Then open in browser:

```
http://localhost:8000
```

---

## Offline-Ready Setup (No Runtime Downloads)

This project now defaults to offline mode. It will not download models at runtime.
Make sure the following assets are present locally before demo day.

1. Silero VAD repo (local path used by `SILERO_VAD_PATH`)

If you already downloaded Silero VAD (or it exists in your Torch Hub cache),
just point to it:
```bash
export SILERO_VAD_PATH="/absolute/path/to/silero-vad"
```

Optional (first-time setup only):
```bash
git clone https://github.com/snakers4/silero-vad pretrained/silero-vad
```

2. Faster-Whisper model directory (set `WHISPER_MODEL_PATH`)

Example:
```bash
export WHISPER_MODEL_PATH="/absolute/path/to/whisper-small"
```

3. Hugging Face cache for tokenizer + TTS (stored under `pretrained/hf`)

Example (run once online to cache locally):
```bash
export HF_HOME="$(pwd)/pretrained/hf"
python - <<'PY'
from transformers import AutoTokenizer, VitsModel
AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
AutoTokenizer.from_pretrained("facebook/mms-tts-tel")
VitsModel.from_pretrained("facebook/mms-tts-tel")
PY
```

4. CTranslate2 NLLB model (already configured)

```bash
ct2-transformers-converter --model facebook/nllb-200-distilled-600M \
    --output_dir pretrained/nllb-200-distilled-600M-int8 \
    --quantization int8
```

If you want to allow downloads again, set `OFFLINE_MODE = False` in `src/config.py`.

---

## How to Use

1. Start the server
2. Open the dashboard in your browser
3. Speak clearly in **Hindi**
4. Observe:

   * Energy levels
   * Speech detection
   * Live transcription output

---

## Example Test Sentences

* मेरा नाम मनमधा है
* मैं हिंदी बोल रहा हूँ
* यह एक परीक्षण है

---

## Current Output

* Real-time Hindi text transcription from speech
* Displayed on the dashboard instantly after speech ends

---

## Next Phase (Planned)

### Phase 3 – Translation

* Hindi text → Telugu text translation
* Integration with Indic translation models
* Real-time translated output on dashboard

---

## Future Scope

* Telugu speech recognition
* Text-to-speech (TTS)
* Full speech-to-speech translation
* Multi-language support

---

## Notes

* Works best with clear speech and minimal background noise
* GPU improves performance significantly (RTX 2050 supported)
* No internet required after model download
