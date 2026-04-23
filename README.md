# Real-Time Speech Translation System

A real-time **Hindi → Telugu speech-to-speech translation** system that runs fully offline after initial model setup.

---

## System Overview

Speak in Hindi — hear the translation in Telugu within seconds. The full pipeline runs locally with no cloud calls.

```
Microphone → VAD Segmenter → ASR (Whisper) → Translation (NLLB-200) → TTS (MMS) → Audio Output
                                                        ↓
                                              Web Dashboard (live)
```

---

## Features

- Live microphone audio capture (streaming, 32 ms frames)
- Voice Activity Detection (Silero VAD) for speech segmentation
- Real-time Hindi speech-to-text using Faster-Whisper (`small` model)
- Hindi → Telugu translation using Facebook NLLB-200-distilled-600M
- Telugu text-to-speech using Facebook MMS-TTS
- GPU-accelerated inference (CUDA / int8_float16)
- Producer–consumer pipeline with bounded queues
- Concurrent processing across audio, segmentation, ASR, translation, and TTS threads
- Real-time web dashboard (WebSocket-based, 10 Hz updates)
- Live audio energy meter and speech activity indicator
- Partial and final transcript display with translation output
- ASR and end-to-end latency tracking
- Session reset (clears all queues and pipeline state)
- Fully offline operation after first-run model download

---

## Tech Stack

| Layer | Library |
|---|---|
| Audio capture | SoundDevice |
| VAD | Silero VAD |
| ASR | Faster-Whisper |
| Translation | CTranslate2 + NLLB-200-distilled-600M |
| TTS | Transformers + MMS-TTS (Telugu) |
| Backend | FastAPI + Uvicorn |
| Dashboard | HTML / CSS / JavaScript (WebSocket) |
| Acceleration | CUDA (NVIDIA) / CPU fallback |

---

## Running the System

**Install dependencies:**

macOS:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-macos.txt
```

Linux with NVIDIA CUDA:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-linux-cuda.txt
```

**Start the server:**
```bash
python src/main.py
```

**Open the dashboard:**
```
http://localhost:8000
```

---

## Offline-Ready Setup

By default the project starts with `OFFLINE_MODE = False` so models download automatically on first run. After all models are cached, set `OFFLINE_MODE = True` in [src/config.py](src/config.py) for strict offline mode.

### Required assets

**1. Silero VAD**

Point to a local clone or let Torch Hub cache it:
```bash
git clone https://github.com/snakers4/silero-vad pretrained/silero-vad
export SILERO_VAD_PATH="$(pwd)/pretrained/silero-vad"
```

**2. Faster-Whisper model**

```bash
export WHISPER_MODEL_PATH="/path/to/whisper-small"
```
If not set, `faster-whisper` downloads and caches `small` automatically.

**3. Hugging Face models (translation + TTS)**

```bash
export HF_HOME="$(pwd)/pretrained/hf"
python - <<'PY'
from transformers import AutoTokenizer, VitsModel
AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
AutoTokenizer.from_pretrained("facebook/mms-tts-tel")
VitsModel.from_pretrained("facebook/mms-tts-tel")
PY
```

**4. NLLB CTranslate2 model**

```bash
ct2-transformers-converter --model facebook/nllb-200-distilled-600M \
    --output_dir pretrained/nllb-200-distilled-600M-int8 \
    --quantization int8
```

---

## How to Use

1. Start the server and open the dashboard in a browser
2. Speak clearly in **Hindi**
3. The dashboard shows:
   - Live audio energy and speech / silence status
   - Hindi transcription (partial and final)
   - Telugu translation
   - Latency metrics
4. Click **Reset Session** to clear state and start a new session

---

## Example Sentences

| Hindi | Telugu |
|---|---|
| मेरा नाम मनमधा है | నా పేరు మన్మధ |
| मैं हिंदी बोल रहा हूँ | నేను హిందీ మాట్లాడుతున్నాను |
| यह एक परीक्षण है | ఇది ఒక పరీక్ష |

---

## Notes

- Works best with clear speech and minimal background noise
- GPU (NVIDIA) significantly reduces end-to-end latency
- No internet connection required after initial model download
- macOS: MPS is not used (CTranslate2 / VITS instability); CPU mode is used instead
