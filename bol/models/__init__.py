no_exception = 0

try:
    from .wav2vec2._wav2vec2_fairseq import Wav2vec2Fairseq, Wav2VecCtc
except:
    no_exception = 1

from ._load_model import load_model
from ._model import BolModel
from .wav2vec2._wav2vec2_ts import Wav2Vec2TS

__all__ = [
    "Wav2Vec2TS",
    "load_model",
    "BolModel",
]

if no_exception == 0:
    __all__.append("Wav2vec2Fairseq")
    __all__.append("Wav2VecCtc")
