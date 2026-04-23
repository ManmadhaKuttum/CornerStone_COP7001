import time

state = {
    "session_id":         0,

    "energy":              0.0,
    "frames":              0,
    "speech_active":       False,
    "start_time":          time.time(),
    "speech_onset_time":   None,
    "last_audio_duration": 0.0,
    "mic_status":          "loading",

    "asr_status":   "loading",
    "trans_status": "loading",
    "tts_status":   "idle",
    "offline_mode": False,

    "partial_transcript": "",
    "final_transcript":   "",

    "partial_translation": "",
    "final_translation":   "",

    "asr_latency_ms":   0,
    "trans_latency_ms": 0,
    "tts_latency_ms":   0,    
    "tts_play_ms":      0,    
    "e2e_latency_ms":   0,

    "rtf":             0.0,
    "total_words_asr":   0,
    "total_words_trans": 0,

    "vad_backend":   "silero-vad",
    "trans_backend": "CTranslate2-NLLB-600M",
    "tts_backend":   "MMS-TTS-Telugu",
}

def reset_session_state():
    state["session_id"] += 1
    state["energy"] = 0.0
    state["frames"] = 0
    state["speech_active"] = False
    state["start_time"] = time.time()
    state["speech_onset_time"] = None
    state["last_audio_duration"] = 0.0
    state["partial_transcript"] = ""
    state["final_transcript"] = ""
    state["partial_translation"] = ""
    state["final_translation"] = ""
    state["asr_latency_ms"] = 0
    state["trans_latency_ms"] = 0
    state["tts_latency_ms"] = 0
    state["tts_play_ms"] = 0
    state["e2e_latency_ms"] = 0
    state["rtf"] = 0.0
    state["total_words_asr"] = 0
    state["total_words_trans"] = 0
