# Real-Time Speech Translation System  
## Phase 1 – Audio Ingress & Live Dashboard

This project aims to build a real-time speech-to-speech translation system for Indian languages.

The current implementation completes **Phase 1 (Day 20 deliverable)**:
Live microphone streaming with a real-time web dashboard for system observability.

---

## Implemented (Phase 1)

- Live microphone audio capture (streaming)
- Producer–consumer pipeline using a bounded queue
- Concurrent processing (audio + monitoring thread)
- Real-time web dashboard (WebSocket-based)
- Live audio energy meter
- Runtime and liveness indicators
- Stable continuous execution
- Fully offline operation

---

## Architecture (Current)

Microphone → Audio Buffer → Monitor Thread → Web Dashboard

This establishes the streaming skeleton required for later integration of ASR, translation, and TTS modules.

---

## Tech Stack

- Python
- SoundDevice
- FastAPI
- WebSockets
- HTML / CSS / JavaScript

---

## Running the System

```bash
python src/main.py