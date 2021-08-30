from bol.metrics.cer import cer_for_evaluate
import torch
import torch.nn as nn
import glob
from bol.utils import load_text_files_in_parallel, load_text_files_in_parallel_from_dir
from bol.metrics import wer_for_evaluate

class Model:
    def __init__(self, model_path, use_cuda_if_available):
        self.model_path = model_path
        self.use_cuda_if_available = use_cuda_if_available
        # self.load_model()

    def fit(self):
        pass

    def preprocess(self):
        pass

    def predict(self, file_path, return_filenames = True):
        #get dataloader
        pass

    def predict_from_dir(self, dir_path, ext,  return_filenames = True):
        file_path = glob.glob(dir_path+'/*.' + ext, recursive=True)
        return self.predict(file_path)



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
    
    def load_model(self):        
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