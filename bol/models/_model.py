import torch
import torch.nn as nn

class Model:
    def __init__(self, model_path, use_cuda_if_available):
        self.model_path = model_path
        self.use_cuda_if_available = use_cuda_if_available
        # self.load_model()

    def fit(self):
        pass

    def preprocess(self):
        pass

    def predict(self, file_path):
        pass

    def evaluate(self):
        pass

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