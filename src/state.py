import time

state = {
    # ── Audio activity ────────────────────────────────────────────────────
    "energy":              0.0,
    "frames":              0,
    "speech_active":       False,
    "start_time":          time.time(),
    "speech_onset_time":   None,
    "last_audio_duration": 0.0,

    # ── Pipeline liveness ─────────────────────────────────────────────────
    "asr_status":   "loading",
    "trans_status": "loading",
    "tts_status":   "idle",
    "offline_mode": False,

    # ── ASR — partial (live while speaking) + finalized ───────────────────
    "partial_transcript": "",
    "final_transcript":   "",

    # ── Translation — partial display + finalized ─────────────────────────
    "partial_translation": "",
    "final_translation":   "",

    # ── Per-stage latency ─────────────────────────────────────────────────
    "asr_latency_ms":   0,
    "trans_latency_ms": 0,
    "tts_latency_ms":   0,    # synthesis only (model inference, not audio playback)
    "tts_play_ms":      0,    # audio playback duration (fixed by text length)
    "e2e_latency_ms":   0,

    # ── RTF and word counters ─────────────────────────────────────────────
    "rtf":             0.0,
    "total_words_asr":   0,
    "total_words_trans": 0,

    # ── Backend labels ────────────────────────────────────────────────────
    "trans_backend": "CTranslate2-NLLB-600M",
    "tts_backend":   "MMS-TTS-Telugu",
}