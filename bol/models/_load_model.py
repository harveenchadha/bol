import torch
from ._wav2vec2 import Wav2VecCtc

class Model:
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        print("I am here")
        pass



class Wav2vec2(Model):
    
    def __init__(self, model_path):
        super().__init__()
        self.load_model(model_path)

    def get_model(self):
        return self._model

    def load_model(self, model_path):   
        from ._wav2vec2 import Wav2VecCtc
        self._model = torch.load(model_path)


def load_model(model_path, type='wav2vec2'):
    if type=='wav2vec2':
        model = Wav2vec2(model_path)
        # model.load_model()
        return model
    