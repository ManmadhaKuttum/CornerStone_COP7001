import time

state = {
    # Phase 1
    "energy":               0.0,
    "frames":               0,
    "start_time":           time.time(),
    "alive":                True,

    # Phase 2 — ASR
    "speech_active":        False,
    "partial_transcript":   "",
    "final_transcript":     "",
    "asr_latency_ms":       0.0,
    "asr_status":           "idle",
    "total_words_asr":      0,

    # Phase 3 — Translation
    "partial_translation":  "",
    "final_translation":    "",
    "trans_latency_ms":     0.0,
    "trans_status":         "idle",
    "trans_backend":        "loading",
    "total_words_trans":    0,

    # E2E
    "e2e_latency_ms":       0.0,
    "_e2e_start":           0.0,

    # Phase 4
    "tts_status":           "not_implemented",
}