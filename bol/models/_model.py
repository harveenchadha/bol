from bol.metrics.cer import cer_for_evaluate
import torch
import torch.nn as nn
import glob
from bol.utils import load_text_files_in_parallel, load_text_files_in_parallel_from_dir
from bol.metrics import wer_for_evaluate
from bol.utils import get_audio_duration
from bol.inference import call_vad
from collections import OrderedDict

class BolModel:
    def __init__(self, model_path, use_cuda_if_available):
        self.model_path = model_path
        self.use_cuda_if_available = use_cuda_if_available
        # self.load_model()

    def fit(self):
        pass

    def preprocess(self):
        pass

    def predict(self, file_path, with_lm = False, return_filenames = True):
        #get dataloader
        pass

    def predict_from_dir(self, dir_path, ext, return_filenames = True, with_lm=False):
        file_path = glob.glob(dir_path+'/*.' + ext, recursive=True)
        return self.predict(file_path, return_filenames = return_filenames, with_lm = with_lm)


    def preprocess_vad(self, wav_path):
        duration = get_audio_duration(wav_path)
        file_paths = []
        if duration > 15:
            file_paths = call_vad(wav_path)
        else:
            file_paths.append(wav_path)

        return file_paths

    def postprocess_vad(self, filenames_from_vad, preds_from_vad):
        predictions = dict(zip(filenames_from_vad, preds_from_vad))
        pred_dict = OrderedDict({})
        for key, value in predictions.items():
            pred_dict[key.split('/')[-1].split('.')[0]] = value

        predictions = OrderedDict(sorted(pred_dict.items()))
        predictions = pred_dict.values()
        return " ".join(predictions)



    def calculate_metrics(self, metrics, ground_truth, predictions):
        metrics = [metric.lower() for metric in metrics]

        calculated_metrics = {}        
        if 'wer' in metrics:
            wer = wer_for_evaluate(ground_truth, predictions)
            calculated_metrics['wer'] = wer

        if 'cer' in metrics:
            cer = cer_for_evaluate(ground_truth, predictions)
            calculated_metrics['cer'] = cer

        return calculated_metrics

    def evaluate(self, audio_file_paths, text_file_paths, return_preds = False,  metrics = ['wer', 'cer']):
        if len(audio_file_paths) != len(text_file_paths):
            raise Exception("The value of ground truth and preds should be same")

        predictions = self.predict(audio_file_paths, return_filenames = True)
        ground_truth = load_text_files_in_parallel(text_file_paths)

        return self.calculate_metrics(metrics, ground_truth, predictions)
        


    def evaluate_from_dir(self, dir_path, ext, text_dir_path, return_preds=False, metrics = ['wer', 'cer']):
        predictions = self.predict_from_dir(dir_path, ext, return_filenames = True)
        ground_truth = load_text_files_in_parallel_from_dir(text_dir_path)

        return self.calculate_metrics(metrics, ground_truth, predictions)


    def move_to(self, device):
        pass

    def get_model(self):
        return self._model

    def load_jit_model(self):
        self._model = torch.jit.load(self.model_path)
    
    def load_model_torch( self):
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

    def get_decoder(self):
        return self._decoder

    def get_alternative_decoder(self):
        return self._alternative_decoder

