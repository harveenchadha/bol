

import torch
from .. inference import load_decoder, get_results_for_single_file, get_results_for_batch
import time
import os
from os.path import expanduser
import pickle
from .. data import Wav2VecDataLoader

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

        # if os.path.exists(model_path + '/kenlm_decoder.pkl'):
        #     print("Loading decoder from cache")
        #     with open(model_path + '/kenlm_decoder.pkl', 'rb') as input:
        #         decoder = pickle.load(input)
        #         self._decoder = decoder
        # else:             
        self._decoder = load_decoder(model_path+'/dict.ltr.txt', model_path+'/lexicon.lst', model_path+'/lm.binary', 'kenlm')

            # with open(model_path  + '/kenlm_decoder.pkl', 'wb') as output:
            #     pickle.dump([self._decoder], output, pickle.HIGHEST_PROTOCOL)

        end = time.time()
        print('Decoder Loaded in '+ str(end-start) + ' seconds')
        self._alternative_decoder = load_decoder(model_path+'/dict.ltr.txt', model_path+'/lexicon.lst', model_path+'/lm.binary', 'viterbi')
        end_viterbi = time.time()

        

        print('Viterbi Loaded in '+ str(end_viterbi-end) + ' seconds')

    def summary(self):
        print(self._model)

    def predict(self, wav_path, viterbi=False):
        start = time.time()
        # import glob
        # from tqdm import tqdm
        # files = glob.glob("../vak/hindi_test_dummy/*.wav")

        # for wav_path in tqdm(files):

        ### single file
        # if viterbi:
        #     text = get_results_for_single_file(wav_path, self.model_path+'/dict.ltr.txt', self.get_alternative_decoder(), self.get_model())
        # else: 
        #     text = get_results_for_single_file(wav_path, self.model_path+'/dict.ltr.txt', self.get_decoder(), self.get_model())

        
        w = Wav2VecDataLoader(16, 4 ,'../vak/hindi_test_dummy')
        dataloader = w.get_train_data_loader()
        #print(len(dataloader))

        text = get_results_for_batch(dataloader, self.model_path+'/dict.ltr.txt', self.get_decoder(), self.get_model())
        end = time.time()

        print("Total time to predict " , str(end-start))

        print(text)
        return text


def check_if_required_files_exist(model_path):
    list_files = os.listdir(model_path)
    required_files = ['hindi.pt', 'lexicon.lst', 'lm.binary', 'dict.ltr.txt']

    list_files = list(set(required_files) - set(list_files))
    
    if len(list_files) > 0:
        raise Exception("Not all files are present. Please make sure files: "+",".join(list_files)+" are present")
        return False
    else:
        return True


def check_model_path(model_path, force_reload=False):
    '''
    Checks whether model path is a path or a language.
    if directory check for all the required files required to run the model.
    if language download the model files from net and save in ~/.bol/models/language
    '''

    path_type= ''

    if os.path.isdir(model_path):
        path_type = 'directory'
        check_if_required_files_exist(model_path)
        return model_path
    else:
        languages_supported = ['hi', 'en-IN']
        if model_path in languages_supported:
            path_type = 'model'

        home = expanduser("~")
        base_path = home + '/.bol/models'
        os.makedirs(base_path, exist_ok=True)
        
        full_path = base_path + '/' + model_path
        if os.path.exists(full_path):
            print("Path already exists")
            # check if required files are present
            check_if_required_files_exist(full_path)
            return full_path
        else:
            wget_cmd = 'wget https://storage.googleapis.com/vakyaansh-open-models/hindi/v2/hindi_v2.zip -P '+ full_path
            print("Uncompressing ...")
            unzip_cmd = 'unzip -q ' + full_path+'/hindi_v2.zip -d ' +full_path
            remove_cmd = 'rm ' + full_path+'/hindi_v2.zip'
            os.system(wget_cmd)
            os.system(unzip_cmd)
            os.system(remove_cmd)
            return full_path 


    

def load_model(model_path, type='wav2vec2'):

    path = check_model_path(model_path)



    if type=='wav2vec2':
        model = Wav2vec2(path)

        return model
    