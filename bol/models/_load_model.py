import torch
import torch.nn as nn
from bol.inference import load_decoder, get_results_for_single_file, get_results_for_batch
import time
import os

from bol.data import Wav2VecDataLoader
from bol.utils.helper_functions import validate_file 
from bol.metrics import evaluate_metrics, wer, cer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist

from ._model import Model



class Wav2vec2(Model):
    
    def __init__(self, model_path, dict_path, lm_path, lexicon_path, use_cuda_if_available=True):
        super().__init__(model_path, use_cuda_if_available)
        
        #self.model_path = model_path
        self.dict_path = dict_path 
        self.lm_path = lm_path
        self.lexicon_path = lexicon_path
        self._alternative_decoder = 'viterbi'
        self.use_cuda_if_available = use_cuda_if_available
        self.device = 'cpu'
        self.load_model()
        if lm_path and lexicon_path:
            self.load_decoder()
        else:
            self._decoder = load_decoder(self.dict_path, '', '', 'viterbi')

    def get_model(self):
        return self._model

    
#    def load_model(self, model_path, load_kenlm=True, force_download=False, use_cuda_if_available=True, language_model_params= {}, donwload_without_lm = False )

    # def setup(self, rank, world_size):
    #     os.environ['MASTER_ADDR'] = 'localhost'
    #     os.environ['MASTER_PORT'] = '12355'

    #     # initialize the process group
    #     dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # def cleanup(self):
    #     dist.destroy_process_group()




    # def run_demo(self,demo_fn, world_size):
    #     mp.spawn(demo_fn,
    #             args=(world_size,),
    #             nprocs=world_size,
    #             join=True)




    # def demo_basic(self,rank, world_size):
    #     print(f"Running basic DDP example on rank {rank}.")
    #     self.setup(rank, world_size)

    #     # create model and move it to GPU with id rank
    #     self._model = self._model.to(rank)
    #     self._model = DDP(self._model, device_ids=[rank])
    #     self.cleanup()



    def load_model(self):
        """ Loads ASR model and language model according to the params specified.

        Args:
            model_path ([str]): directory path where model is present.
            load_language_model (bool, optional): whether to load a language model in addition to the ASR model. Defaults to True.
            force_download (bool, optional): if for a particular language code model files are already present, setting this to true will redownload all the files. Defaults to False.
            use_cuda_if_available (bool, optional): whether to use cuda or not if available. Defaults to True.
            language_model_params (dict, optional): long description of the kenlm based language model params. Defaults to {}. Long description.
            donwload_without_lm (bool, optional): whether to download the language model in addition to the ASR model. Defaults to False.


        Description:
            model_path: if a system path is specified to .pt file will pick the model from there, else if the string matches a language code, will download the required model files
                        from the server. This download of files will happen only once. To check all the language codes available run list_models.
            
            load_language_model: 

        """
        
        if torch.cuda.is_available() and self.use_cuda_if_available:
            self.use_cuda_if_available = True
            self._model = torch.load(self.model_path, map_location=torch.device('cuda'))
            print("Model loaded on GPUs")
        else:
            self.use_cuda_if_available=False
            self._model = torch.load(self.model_path)
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

    def load_decoder(self):
        start = time.time()

        # if os.path.exists(model_path + '/kenlm_decoder.pkl'):
        #     print("Loading decoder from cache")
        #     with open(model_path + '/kenlm_decoder.pkl', 'rb') as input:
        #         decoder = pickle.load(input)
        #         self._decoder = decoder
        # else:    
     
        self._decoder = load_decoder(self.dict_path, self.lexicon_path, self.lm_path, 'kenlm')

            # with open(model_path  + '/kenlm_decoder.pkl', 'wb') as output:
            #     pickle.dump([self._decoder], output, pickle.HIGHEST_PROTOCOL)

        end = time.time()
        print('Decoder Loaded in '+ str(end-start) + ' seconds')
        self._alternative_decoder = load_decoder(self.dict_path, self.lexicon_path, self.lm_path, 'viterbi')
        end_viterbi = time.time()
        # print('Viterbi Loaded in '+ str(end_viterbi-end) + ' seconds')

    def summary(self):
        print(self._model)

    def predict(self, file_path, viterbi=False, return_filenames = False):
        type_file_path = check_if_prediction_is_wav_or_directory(file_path)
        text = ''

        if len(file_path) == 1:
            if viterbi:
                text = get_results_for_single_file(file_path, self.dict_path, self.get_alternative_decoder(), self.get_model(), self.use_cuda_if_available)
            else: 
                text = get_results_for_single_file(file_path, self.dict_path, self.get_decoder(), self.get_model(), self.use_cuda_if_available)

            if return_filenames:
                return [{'file':file_path, 'transcription': text}] 
            else:
                return text

        else:


        # if type_file_path == 'file':
        #     file_path = validate_file(file_path[0])
        #     if viterbi:
        #         text = get_results_for_single_file(file_path[0], self.dict_path, self.get_alternative_decoder(), self.get_model(), self.use_cuda_if_available)
        #     else: 
        #         text = get_results_for_single_file(file_path[0], self.dict_path, self.get_decoder(), self.get_model(), self.use_cuda_if_available)


        # else:
            dataloader_obj = Wav2VecDataLoader(train_batch_size = 8, num_workers= 4 ,file_data_path = file_path)
            dataloader = dataloader_obj.get_file_data_loader()

            if viterbi:
                text = get_results_for_batch(dataloader, self.dict_path, self.get_alternative_decoder(), self.get_model(),  self.use_cuda_if_available)
            else:

                text = get_results_for_batch(dataloader, self.dict_path, self.get_decoder(), self.get_model(), self.use_cuda_if_available)

            if return_filenames:
                return text
            else:
                only_preds = [item['transcription'] for item in text]
                return only_preds
        
        # elif type_file_path == 'dir':
        #     dataloader_obj = Wav2VecDataLoader(train_batch_size = 8, num_workers= 4 ,file_data_path = file_path)
        #     dataloader = dataloader_obj.get_file_data_loader()

        #     if viterbi:
        #         text = get_results_for_batch(dataloader, self.dict_path, self.get_alternative_decoder(), self.get_model(),  self.use_cuda_if_available)
        #     else:

        #         text = get_results_for_batch(dataloader, self.dict_path, self.get_decoder(), self.get_model(), self.use_cuda_if_available)

        # if return_filenames:
        #     if type_file_path=='file':
        #         return [{'file':file_path, 'transcription': text}]
        #     elif type_file_path=='dir':
        #         return text
        # else:
        #     if type_file_path=='file':
        #         return text
        #     elif type_file_path=='dir':
        #         only_preds = []
        #         for item in text:
        #             # for file_name, local_text in item:
        #             only_preds.append(item['transcription'])
        #         text = only_preds
        # return text

    def evaluate(self, ground_truth, predictions,metrics=['wer','cer']):
        metrics = [metric.lower() for metric in metrics]
        dict_metrics = {}
        import glob
        ground_truth_ = glob.glob(ground_truth+'/*.txt')

        gt_content_list = [] 
        pr_content_list = []        
        for gt in ground_truth_:
            filename = gt.split('/')[-1]
            with open(gt) as gt_file:
                gt_content = gt_file.read().strip()
                gt_content_list.append(gt_content)

            with open(predictions+'/'+filename) as pr_file:
                pr_content = pr_file.read().strip()
                pr_content_list.append(pr_content)

            # if 'wer' in metrics:
            #     calculated_wer = wer(ground_truth, predictions)
            #     #print(calculated_wer)
            #     dict_metrics['wer'] = calculated_wer

            # if 'cer' in metrics:
            #     calculated_cer = cer(ground_truth, predictions)
            #     dict_metrics['cer'] = calculated_cer
        
        # return dict_metrics

        wer, cer = evaluate_metrics(gt_content_list, pr_content_list, mode='list')
        print("WER: ", wer)
        print("CER: ", cer)


        #print(calculated_wer)
        #wer, cer = calculate_metrics_for_batch()

    def predict_evaluate(self, file_path, txt_path=None, viterbi=False, metrics=['wer','cer']):
        predicted_text = self.predict(file_path, viterbi, return_filenames=True)
        print(predicted_text)

        type_file_path = check_if_prediction_is_wav_or_directory(file_path)

        if not txt_path:
            txt_path = file_path

        if type_file_path == 'file':
            # open txt file
            wer, cer =  evaluate_metrics(txt_path, predicted_text, mode='file')
        elif type_file_path == 'dir':
            wer, cer = evaluate_metrics(txt_path, predicted_text, mode='dir')
        elif type_file_path == 'list':
            wer, cer = evaluate_metrics(txt_path, predicted_text, mode='list')
            

        return wer, cer

def check_if_prediction_is_wav_or_directory(file_path):
    if os.path.isfile(file_path):
        return 'file'
    elif os.path.isdir(file_path):
        return 'dir'
    elif type(file_path) == list:
        return 'list'
    else:
        raise Exception("Not a valid file or directory")




from bol.utils.model_zoo import verify_model_mapping, verify_lm_mapping

from .wav2vec2._wav2vec2_ts import Wav2Vec2TS



def load_model_ts(model_path,
                use_cuda_if_available=True,
                ):
    model = Wav2Vec2TS(model_path, use_cuda_if_available)
    return model



def load_model( model_code= None,
                model_path= None,
                dict_path= None,
                lm_path= None,
                lexicon_path = None,
                backend='wav2vec2', 
                force_download=False,
                use_cuda_if_available=True,
                lm_params={},
                use_lm=True
                ):
    if not model_code and not model_path:
        raise Exception("Either model code or model path is required")

    if model_code:
        ## check string mapping to available bol models
        #force_download=True
        model_path, dict_path = verify_model_mapping(model_code, force_download)

    elif model_path:
        if not dict_path:
            raise Exception("dict path is required with model path")

    if use_lm:
        if model_code:
            ## donwload
            lm_path, lexicon_path = verify_lm_mapping(model_code)
        elif model_path:
            if not lm_path:
                raise Exception("path to lm.binary is required is use_lm is True")
            if not lexicon_path:
                raise Exception("path to lexicon.lst is required is use_lm is True")

            ## lm_path, lexicon_path
        ## check whether to load language model or not.

#    path = check_model_path(model_path)
    if backend=='wav2vec2':
        model = Wav2vec2(model_path, dict_path, lm_path, lexicon_path, use_cuda_if_available)
        return model




