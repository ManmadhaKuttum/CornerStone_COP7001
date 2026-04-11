import os
import multiprocessing as _mp
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Offline mode ──────────────────────────────────────────────────────────────
# Set True only after all models are cached locally.
# When False, models download automatically on first run.
OFFLINE_MODE = False

HF_HOME          = os.getenv("HF_HOME", os.path.join(PROJECT_ROOT, "pretrained", "hf"))
SILERO_VAD_PATH  = os.getenv("SILERO_VAD_PATH", os.path.join(PROJECT_ROOT, "pretrained", "silero-vad"))

# ── ASR model ─────────────────────────────────────────────────────────────────
# If WHISPER_MODEL_PATH points to a real local directory, use it.
# Otherwise use "small" — faster-whisper downloads and caches it automatically.
_whisper_env = os.getenv("WHISPER_MODEL_PATH", "")
# Accuracy-first default:
# "small" has noticeably better Hindi quality than "tiny" in most noisy/live cases.
ASR_MODEL = _whisper_env if (os.path.isdir(_whisper_env)) else "small"

# ── Translation / TTS model IDs ───────────────────────────────────────────────
TRANSLATION_TOKENIZER_ID = "facebook/nllb-200-distilled-600M"
TTS_MODEL_ID             = "facebook/mms-tts-tel"

if OFFLINE_MODE:
    os.environ.setdefault("HF_HUB_OFFLINE",      "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", HF_HOME)

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE       = "cuda"
    COMPUTE_TYPE = "int8_float16"
elif torch.backends.mps.is_available():
    DEVICE       = "cpu"      # CTranslate2 does not support MPS
    COMPUTE_TYPE = "int8"
else:
    DEVICE       = "cpu"
    COMPUTE_TYPE = "int8"

# ── Audio ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 16000
FRAME_DURATION = 0.032          # 32 ms = 512 samples (Silero minimum)
FRAME_SAMPLES  = int(SAMPLE_RATE * FRAME_DURATION)
CHANNELS       = 1

# ── VAD / segmenter ───────────────────────────────────────────────────────────
VAD_THRESHOLD     = 0.55        # keep sensitivity for soft Hindi consonants
SILENCE_DURATION  = 0.50        # safer end-of-utterance cut for better word completion
MAX_UTTERANCE_SEC = 8.0         # more context for final ASR quality
SILENCE_FRAMES    = int(SILENCE_DURATION / FRAME_DURATION)
MAX_FRAMES        = int(MAX_UTTERANCE_SEC / FRAME_DURATION)

# Partial flush: send buffered audio to fast ASR every N seconds while speaking.
# This drives live partial_transcript on dashboard.
PARTIAL_INTERVAL  = 1.0         # seconds between partial ASR inferences

# ── ASR ───────────────────────────────────────────────────────────────────────
ASR_LANGUAGE                = "hi"
BEAM_SIZE                   = 4     # accuracy-first: beam=4 is OpenAI's recommended default for Whisper
PARTIAL_BEAM_SIZE           = 1     # partial path — speed only, never feeds TTS
NO_SPEECH_THRESHOLD         = 0.60
COMPRESSION_RATIO_THRESHOLD = 2.4
LOG_PROB_THRESHOLD          = -1.0

# ── Translation ───────────────────────────────────────────────────────────────
SRC_LANG      = "hin_Deva"
TGT_LANG      = "tel_Telu"
MAX_LENGTH    = 128
TRANS_BEAMS   = 1                           # greedy decoding — ~2× faster
# CTranslate2 thread counts — uses all Apple Silicon efficiency+performance cores
CT2_INTRA_THREADS = min(8, _mp.cpu_count())  # parallelise each op
CT2_INTER_THREADS = 2                        # overlap independent ops

# ── Queue sizes ───────────────────────────────────────────────────────────────
AUDIO_QUEUE_SIZE = 200
TRANS_QUEUE_SIZE = 50

# ── Dashboard ─────────────────────────────────────────────────────────────────
WS_HZ = 10
