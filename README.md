# Real-Time Speech Conversion System

A low-latency, streaming speech-to-speech translation system that converts spoken input in one language into synthesized speech in a target Indian language using a pipelined AI architecture.

---

## Overview

This project implements a **real-time streaming pipeline** that processes live microphone input and produces translated speech output with minimal delay. Unlike batch-based systems, it processes audio continuously using concurrent modules to ensure interactive performance.

---

## Architecture

The system follows a **Pipelined Streaming Architecture**:

Microphone → Audio Buffer → Streaming ASR → Translation → TTS → Speaker Output

Each module runs independently using producer–consumer queues to reduce end-to-end latency.

---

## Core Components

- **Audio Ingress:** Captures microphone input in small streaming frames.
- **ASR Engine:** Converts speech to text incrementally (partial + final transcripts).
- **Translation Engine:** Translates finalized segments into the target language.
- **TTS Engine:** Synthesizes translated text into speech for immediate playback.

---

## Technology Stack

- Python  
- PyAudio / SoundDevice  
- Whisper / Vosk (ASR)  
- IndicTrans2 / MarianMT (Translation)  
- Coqui TTS (Speech Synthesis)

---


---

## Goals

- End-to-End Latency < 2 seconds  
- Real-Time Factor (RTF) < 1  
- Smooth streaming without buffering delays  
- Modular and extensible design  

---

## Status

- Under Active Development

