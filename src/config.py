import platform
import subprocess

SAMPLE_RATE = 16000
FRAME_DURATION = 0.02
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)

QUEUE_MAXSIZE = 100

SOURCE_LANG = "hi"
TARGET_LANG = "te"

START_THRESHOLD = 0.02
STOP_THRESHOLD = 0.01
SILENCE_DURATION = 0.8
MAX_SPEECH_DURATION = 8.0
MIN_SPEECH_DURATION = 0.8

ASR_MODEL_SIZE = "small"

def has_nvidia_gpu():
    """Checks if an NVIDIA GPU is available on the system."""
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        return True
    except Exception:
        return False

# Dynamic Hardware Detection
if platform.system() == "Darwin" or not has_nvidia_gpu():
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"
else:
    DEVICE = "cuda"
    COMPUTE_TYPE = "float16"