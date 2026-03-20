import queue
import sounddevice as sd
from config import SAMPLE_RATE, FRAME_SIZE, QUEUE_MAXSIZE

audio_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)


def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)
    try:
        audio_queue.put_nowait(indata.copy())
    except queue.Full:
        pass


def start_audio_stream():
    return sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=FRAME_SIZE,
        callback=audio_callback,
    )