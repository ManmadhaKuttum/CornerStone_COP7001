import time

state = {
    "energy": 0.0,
    "frames": 0,
    "start_time": time.time(),
    "alive": True,
    "speech_active": False,
    "partial_transcript": "",
    "final_transcript": "",
    "asr_latency_ms": 0.0,
}