import queue
import threading
import numpy as np

from config import SAMPLE_RATE, FRAME_SAMPLES, CHANNELS, AUDIO_QUEUE_SIZE, MIC_DEVICE
from state import state

audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_SIZE)
_sd = None
_mic_stream = None

def _sounddevice():
    global _sd
    if _sd is None:
        import sounddevice as sd
        _sd = sd
    return _sd

def audio_callback(indata, frames, time_info, status):
    frame = indata[:, 0].astype(np.float32).copy()
    try:
        audio_queue.put_nowait(frame)
    except queue.Full:
        pass

def start_mic():
    global _mic_stream
    sd = _sounddevice()
    state["mic_status"] = "loading"

    devices_to_try = [MIC_DEVICE] if MIC_DEVICE is not None else []
    devices_to_try.append(None)

    last_error = None
    for device in devices_to_try:
        try:
            _mic_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                blocksize=FRAME_SAMPLES,
                dtype="float32",
                callback=audio_callback,
                device=device,
            )
            _mic_stream.start()
            state["mic_status"] = "ready"
            return _mic_stream
        except Exception as exc:
            last_error = exc

    state["mic_status"] = "unavailable"
    raise RuntimeError(f"Unable to start microphone input: {last_error}") from last_error

def start_mic_background():
    def _worker():
        try:
            start_mic()
        except Exception as exc:
            print(f"⚠️  Microphone failed to start: {exc}")

    t = threading.Thread(target=_worker, daemon=True, name="mic")
    t.start()
    return t
