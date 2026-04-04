import torch

if torch.cuda.is_available():
    DEVICE = "cuda"
    COMPUTE_TYPE = "int8_float16"
elif torch.backends.mps.is_available():
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"
else:
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"

SAMPLE_RATE     = 16000
FRAME_DURATION  = 0.032
FRAME_SAMPLES   = int(SAMPLE_RATE * FRAME_DURATION)
CHANNELS        = 1

VAD_THRESHOLD       = 0.6
SILENCE_DURATION    = 0.55
MAX_UTTERANCE_SEC   = 8.0
SILENCE_FRAMES      = int(SILENCE_DURATION / FRAME_DURATION)
MAX_FRAMES          = int(MAX_UTTERANCE_SEC / FRAME_DURATION)

ASR_MODEL       = "base"
ASR_LANGUAGE    = "hi"
BEAM_SIZE       = 1
NO_SPEECH_THRESHOLD         = 0.6
COMPRESSION_RATIO_THRESHOLD = 2.4
LOG_PROB_THRESHOLD          = -1.0

TRANSLATION_MODEL   = "facebook/nllb-200-distilled-600M"
SRC_LANG            = "hin_Deva"
TGT_LANG            = "tel_Telu"
NUM_BEAMS           = 1
MAX_LENGTH          = 128

TTS_BACKEND         = "indic"
TTS_LANGUAGE        = "te"

AUDIO_QUEUE_SIZE    = 200
TRANS_QUEUE_SIZE    = 50
TTS_QUEUE_SIZE      = 20

WS_HZ               = 10