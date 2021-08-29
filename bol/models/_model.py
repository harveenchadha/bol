import torch
import torch.nn as nn
import glob

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




#    def evaluate(self, ground_truth, predictions, metrics=['wer','cer']):
    # def evaluate(self, audio_file_paths, text_file_paths, return_preds = False, metrics = ['wer', 'cer']):
    #     predictions = self.predict(audio_file_paths, return_filenames = True)

    #     metrics = [metric.lower() for metric in metrics]

    #     dict_metrics = {}
    #     import glob
    #     ground_truth_ = glob.glob(ground_truth+'/*.txt')

    #     gt_content_list = [] 
    #     pr_content_list = []        
    #     for gt in ground_truth_:
    #         filename = gt.split('/')[-1]
    #         with open(gt) as gt_file:
    #             gt_content = gt_file.read().strip()
    #             gt_content_list.append(gt_content)

    #         with open(predictions+'/'+filename) as pr_file:
    #             pr_content = pr_file.read().strip()
    #             pr_content_list.append(pr_content)

    #         # if 'wer' in metrics:
    #         #     calculated_wer = wer(ground_truth, predictions)
    #         #     #print(calculated_wer)
    #         #     dict_metrics['wer'] = calculated_wer

    #         # if 'cer' in metrics:
    #         #     calculated_cer = cer(ground_truth, predictions)
    #         #     dict_metrics['cer'] = calculated_cer
        
    #     # return dict_metrics

    #     wer, cer = evaluate_metrics(gt_content_list, pr_content_list, mode='list')
    #     print("WER: ", wer)
    #     print("CER: ", cer)


    def move_to(self, device):
        pass

    def get_model(self):
        return self._model

    def load_jit_model(self):
        self._model = torch.jit.load(self.model_path)
    
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