import queue
import numpy as np
import sounddevice as sd
from config import SAMPLE_RATE, FRAME_SAMPLES, CHANNELS, AUDIO_QUEUE_SIZE

audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_SIZE)

def audio_callback(indata, frames, time_info, status):
    frame = indata[:, 0].astype(np.float32).copy()
    try:
        audio_queue.put_nowait(frame)
    except queue.Full:
        pass

def start_mic():
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=FRAME_SAMPLES,
        dtype="float32",
        callback=audio_callback,
    )
    stream.start()
    return stream