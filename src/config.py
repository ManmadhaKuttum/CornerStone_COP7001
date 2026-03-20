import platform
import subprocess
import torch

# ── Audio ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 16000
FRAME_DURATION = 0.032   # was 0.02
FRAME_SIZE     = int(SAMPLE_RATE * FRAME_DURATION)  # 512 samples
QUEUE_MAXSIZE  = 200

# ── Languages ─────────────────────────────────────────────────────────────────
SOURCE_LANG      = "hi"          # Hindi  (ASR)
TARGET_LANG      = "te"          # Telugu (Translation)
INDICTRANS_SRC   = "hin_Deva"
INDICTRANS_TGT   = "tel_Telu"

# ── VAD (Silero) ──────────────────────────────────────────────────────────────
VAD_THRESHOLD       = 0.6    # raised — reduces noise triggers
SILENCE_DURATION    = 0.8    # seconds silence before flush
MAX_SPEECH_DURATION = 8.0
MIN_SPEECH_DURATION = 0.5

# ── ASR ───────────────────────────────────────────────────────────────────────
ASR_MODEL_SIZE = "small"
ASR_BEAM_SIZE  = 5

# ── Device: faster-whisper doesn't support MPS, use cpu+int8 on Mac ──────────
def _has_nvidia():
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        return True
    except Exception:
        return False

if torch.cuda.is_available():
    DEVICE       = "cuda"
    COMPUTE_TYPE = "float16"
else:
    DEVICE       = "cpu"
    COMPUTE_TYPE = "int8"

# Translation can use MPS on Mac
if torch.cuda.is_available():
    TRANS_DEVICE = "cuda"
elif platform.system() == "Darwin" and torch.backends.mps.is_available():
    TRANS_DEVICE = "mps"
else:
    TRANS_DEVICE = "cpu"

print(f"[config] device={DEVICE}  compute={COMPUTE_TYPE}  trans_device={TRANS_DEVICE}")