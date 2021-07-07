

import torch
from .. inference import load_decoder, get_results_for_single_file
import time
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
        #super().__init__()
        self.model_path = model_path
        self._alternative_decoder = 'viterbi'
        self.load_model(model_path)
        self.load_decoder(model_path)

    def get_model(self):
        return self._model

    def load_model(self, model_path):   
        self._model = torch.load(model_path+'/hindi.pt')
        print('Model Loaded')

    def get_decoder(self):
        return self._decoder

    def get_alternative_decoder(self):
        return self._alternative_decoder

    def load_decoder(self, model_path):
        start = time.time()
        self._decoder = load_decoder(model_path+'/dict.ltr.txt', model_path+'/lexicon.lst', model_path+'/lm.binary', 'kenlm')
        end = time.time()
        print('Decoder Loaded in '+ str(end-start) + ' seconds')
        self._alternative_decoder = load_decoder(model_path+'/dict.ltr.txt', model_path+'/lexicon.lst', model_path+'/lm.binary', 'viterbi')
        end_viterbi = time.time()

        print('Viterbi Loaded in '+ str(end_viterbi-end) + ' seconds')

    def summary(self):
        print(self._model)

    def predict(self, wav_path, viterbi=False):
        if viterbi:
            text = get_results_for_single_file(wav_path, self.model_path+'/dict.ltr.txt', self.get_alternative_decoder(), self.get_model())
        else: 
            text = get_results_for_single_file(wav_path, self.model_path+'/dict.ltr.txt', self.get_decoder(), self.get_model())
        print(text)
        return text


def load_model(model_path, type='wav2vec2'):
    if type=='wav2vec2':
        model = Wav2vec2(model_path)

        return model
    