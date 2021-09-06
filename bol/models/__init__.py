#from .wav2vec2._wav2vec2_fairseq import Wav2vec2Fairseq, Wav2VecCtc
from .wav2vec2._wav2vec2_ts import Wav2Vec2TS
from ._load_model import load_model
from ._model import BolModel

__all__=[
#    "Wav2vec2Fairseq",
#    "Wav2VecCtc",
    "Wav2Vec2TS",
    "load_model",
    "BolModel",
]
