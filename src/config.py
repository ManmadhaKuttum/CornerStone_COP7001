import os
import multiprocessing as _mp
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OFFLINE_MODE = False

HF_HOME          = os.getenv("HF_HOME", os.path.join(PROJECT_ROOT, "pretrained", "hf"))
SILERO_VAD_PATH  = os.getenv("SILERO_VAD_PATH", os.path.join(PROJECT_ROOT, "pretrained", "silero-vad"))

_whisper_env = os.getenv("WHISPER_MODEL_PATH", "")
ASR_MODEL = _whisper_env if (os.path.isdir(_whisper_env)) else "small"

TRANSLATION_TOKENIZER_ID = "facebook/nllb-200-distilled-600M"
TTS_MODEL_ID             = "facebook/mms-tts-tel"

if OFFLINE_MODE:
    os.environ.setdefault("HF_HUB_OFFLINE",      "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", HF_HOME)

if torch.cuda.is_available():
    DEVICE       = "cuda"
    COMPUTE_TYPE = "int8_float16"
    TRANSLATION_DEVICE       = "cuda"
    TRANSLATION_COMPUTE_TYPE = "int8_float16"
    TTS_DEVICE               = "cuda"
elif torch.backends.mps.is_available():
    DEVICE       = "cpu"      
    COMPUTE_TYPE = "int8"
    TRANSLATION_DEVICE       = "cpu"
    TRANSLATION_COMPUTE_TYPE = "int8"
    TTS_DEVICE               = "cpu"   
else:
    DEVICE       = "cpu"
    COMPUTE_TYPE = "int8"
    TRANSLATION_DEVICE       = "cpu"
    TRANSLATION_COMPUTE_TYPE = "int8"
    TTS_DEVICE               = "cpu"

SAMPLE_RATE    = 16000
FRAME_DURATION = 0.032          
FRAME_SAMPLES  = int(SAMPLE_RATE * FRAME_DURATION)
CHANNELS       = 1
_mic_device_env = os.getenv("MIC_DEVICE")
if _mic_device_env is None or _mic_device_env == "":
    MIC_DEVICE = None
else:
    try:
        MIC_DEVICE = int(_mic_device_env)
    except ValueError:
        MIC_DEVICE = _mic_device_env

VAD_THRESHOLD     = 0.55        
SILENCE_DURATION  = 0.50       
MAX_UTTERANCE_SEC = 8.0       
SILENCE_FRAMES    = int(SILENCE_DURATION / FRAME_DURATION)
MAX_FRAMES        = int(MAX_UTTERANCE_SEC / FRAME_DURATION)

PARTIAL_INTERVAL  = 1.0         

ASR_LANGUAGE                = "hi"
BEAM_SIZE                   = 4 if torch.cuda.is_available() else 3
PARTIAL_BEAM_SIZE           = 1     
NO_SPEECH_THRESHOLD         = 0.60
COMPRESSION_RATIO_THRESHOLD = 2.4
LOG_PROB_THRESHOLD          = -1.0

SRC_LANG      = "hin_Deva"
TGT_LANG      = "tel_Telu"
MAX_LENGTH    = 128
TRANS_BEAMS   = 2                          

if TRANSLATION_DEVICE == "cpu":
    CT2_INTRA_THREADS = min(8, _mp.cpu_count()) 
    CT2_INTER_THREADS = 2                        
else:
    CT2_INTRA_THREADS = 1
    CT2_INTER_THREADS = 1

AUDIO_QUEUE_SIZE = 200
TRANS_QUEUE_SIZE = 50

WS_HZ = 10
