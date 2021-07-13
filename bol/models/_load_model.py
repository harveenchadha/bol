

import torch
import torch.nn as nn
from bol.inference import load_decoder, get_results_for_single_file, get_results_for_batch
import time
import os
from os.path import expanduser
import pickle
from bol.data import Wav2VecDataLoader
from bol.utils.helper_functions import validate_file 
from bol.metrics import calculate_metrics_for_single_file, calculate_metrics_for_batch, wer, cer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist

class Model:
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        print("I am here")
        pass

class Wav2vec2(Model):
    
    def __init__(self, model_path, use_cuda_if_available=True):
        #super().__init__()
        self.model_path = model_path
        self._alternative_decoder = 'viterbi'
        self.use_cuda_if_available = use_cuda_if_available
        self.device = 'cpu'
        self.load_model(model_path, use_cuda_if_available)
        self.load_decoder(model_path)

    def get_model(self):
        return self._model

    
#    def load_model(self, model_path, load_kenlm=True, force_download=False, use_cuda_if_available=True, language_model_params= {}, donwload_without_lm = False )

    def setup(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    def cleanup(self):
        dist.destroy_process_group()




    def run_demo(self,demo_fn, world_size):
        mp.spawn(demo_fn,
                args=(world_size,),
                nprocs=world_size,
                join=True)




    def demo_basic(self,rank, world_size):
        print(f"Running basic DDP example on rank {rank}.")
        self.setup(rank, world_size)

        # create model and move it to GPU with id rank
        self._model = self._model.to(rank)
        self._model = DDP(self._model, device_ids=[rank])
        self.cleanup()



    def load_model(self, model_path, use_cuda_if_available=True):
        if torch.cuda.is_available() and use_cuda_if_available:
            self._model = torch.load(model_path+'/hindi.pt', map_location=torch.device('cuda'))
            self.use_cuda_if_available = True
            print("Model loaded on GPUs")
        else:
            self._model = torch.load(model_path+'/hindi.pt')
            self.use_cuda_if_available = False
            print('Model Loaded on CPU')

        if torch.cuda.device_count() > 1:
            self._model = nn.DataParallel(self._model)

            #self.run_demo(self.demo_basic, 2)

        # else:
        #     self._model = torch.load(model_path+'/hindi.pt')

        #print('Model Loaded')

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
        # print('Viterbi Loaded in '+ str(end_viterbi-end) + ' seconds')

    def summary(self):
        print(self._model)

    def predict(self, file_path, viterbi=False, return_filenames = False):
        type_file_path = check_if_prediction_is_wav_or_directory(file_path)
        text = ''
        if type_file_path == 'file':
            file_path = validate_file(file_path)
            print(file_path)
            if viterbi:
                text = get_results_for_single_file(file_path, self.model_path+'/dict.ltr.txt', self.get_alternative_decoder(), self.get_model())
            else: 
                text = get_results_for_single_file(file_path, self.model_path+'/dict.ltr.txt', self.get_decoder(), self.get_model())

#            text = [(file_path, text)]
        
        elif type_file_path == 'dir':
            dataloader_obj = Wav2VecDataLoader(train_batch_size = 4, num_workers= 4 ,file_data_path = file_path)
            dataloader = dataloader_obj.get_file_data_loader()

            if viterbi:
                text = get_results_for_batch(dataloader, self.model_path+'/dict.ltr.txt', self.get_alternative_decoder(), self.get_model(),  self.use_cuda_if_available)
            else:

                text = get_results_for_batch(dataloader, self.model_path+'/dict.ltr.txt', self.get_decoder(), self.get_model(), self.use_cuda_if_available)
                #self.cleanup()
        #print(text)

        if return_filenames:
            if type_file_path=='file':
                return [(file_path, text)]
            elif type_file_path=='dir':
                return text
        else:
            if type_file_path=='file':
                return text
            elif type_file_path=='dir':
                only_preds = []
                for item in text:
                    # for file_name, local_text in item:
                    only_preds.append(item[1])
                text = only_preds
        return text

    def evaluate(self, ground_truth, predictions,metrics=['wer','cer']):
        metrics = [metric.lower() for metric in metrics]
        dict_metrics = {}
        if 'wer' in metrics:
            calculated_wer = wer(ground_truth, predictions)
            dict_metrics['wer'] = calculated_wer

        if 'cer' in metrics:
            calculated_cer = cer(ground_truth, predictions)
            dict_metrics['cer'] = calculated_cer
        
        return dict_metrics
        #print(calculated_wer)
        #wer, cer = calculate_metrics_for_batch()

    def predict_evaluate(self, file_path, txt_path=None, viterbi=False, metrics=['wer','cer']):
        predicted_text = self.predict(file_path, viterbi, return_filenames=True)

        type_file_path = check_if_prediction_is_wav_or_directory(file_path)

        if not txt_path:
            txt_path = file_path

        if type_file_path == 'file':
            # open txt file
            wer, cer = calculate_metrics_for_single_file(txt_path, predicted_text)
        elif type_file_path == 'dir':
            wer, cer = calculate_metrics_for_batch(txt_path, predicted_text)
            

        return wer, cer

def check_if_prediction_is_wav_or_directory(file_path):
    if os.path.isfile(file_path):
        return 'file'
    elif os.path.isdir(file_path):
        return 'dir'
    else:
        raise Exception("Not a valid file or directory")


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
            # print("Path already exists")
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




