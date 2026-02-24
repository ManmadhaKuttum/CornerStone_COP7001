import sounddevice as sd
import numpy as np
import queue
import time
import whisper
import tempfile
import soundfile as sf
import torch

SAMPLE_RATE = 16000
FRAME_DURATION = 0.02
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)

ENERGY_THRESHOLD = 0.02
SILENCE_DURATION = 1.0

audio_queue = queue.Queue()

print("Loading Whisper model...")
model = whisper.load_model("base", device="cuda")

def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

def transcribe_audio(audio_frames):
    audio_data = np.concatenate(audio_frames, axis=0)

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, audio_data, SAMPLE_RATE)
        result = model.transcribe(tmp.name)
        print("📝 Transcript:", result["text"])

def start_stream():
    print("🎤 Listening...")
    speech_buffer = []
    silence_start = None
    speaking = False

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=FRAME_SIZE,
        callback=audio_callback
    ):
        while True:
            frame = audio_queue.get()
            energy = np.linalg.norm(frame) / len(frame)

            if energy > ENERGY_THRESHOLD:
                if not speaking:
                    print("🗣 Speech detected")
                    speaking = True
                silence_start = None
                speech_buffer.append(frame)

            else:
                if speaking:
                    if silence_start is None:
                        silence_start = time.time()

                    elif time.time() - silence_start > SILENCE_DURATION:
                        print("🔇 Speech ended")
                        transcribe_audio(speech_buffer)
                        speech_buffer = []
                        speaking = False

if __name__ == "__main__":
    start_stream()