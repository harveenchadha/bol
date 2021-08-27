import torch
import torchaudio
from .._model import Model

class Wav2Vec2TS(Model):
    def __init__(self, model_path, use_cuda_if_available):
        super().__init__(model_path, 'False')
        self.load_jit_model()

    def predict(self, file_path,  return_filenames = False):
        waveform , _ = torchaudio.load(file_path[0])
        return self._model(waveform)
