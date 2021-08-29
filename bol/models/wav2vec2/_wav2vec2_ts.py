
from .._model import Model
from bol.data import Wav2Vec2TsDataLoader
from tqdm import tqdm
from joblib import Parallel, delayed
import glob
import torchaudio
from joblib import parallel_backend
import gc



class Wav2Vec2TS(Model):
    def __init__(self, model_path, use_cuda_if_available):
        super().__init__(model_path, 'False')
        self.load_jit_model()


    # def predict(self, file_path,  return_filenames = False):
    #     #waveform , _ = torchaudio.load(file_path[0])
    #     #print(waveform.shape)
    #     dataloader_obj = Wav2Vec2TsDataLoader(batch_size = 8, num_workers= 4 ,file_data_path = file_path)
    #     dataloader = dataloader_obj.get_file_data_loader()
        
    #     preds = []
    #     filenames = []
    #     for batch in tqdm(dataloader):
    #         #print(batch[0].shape)
    #         preds.append(self._model(batch[0][0]))
    #         filenames.append(batch[1])

    #     # Parallel(n_jobs = -1)(delayed())

    #     return dict(zip(preds, filenames))



    def predict(self, file_path,  return_filenames = True):
        preds = []
        filenames = []

        # ## works in dataloader but output is not correct ##
        # dataloader_obj = Wav2Vec2TsDataLoader(batch_size = 8, num_workers= 4 ,file_data_path = file_path)
        # dataloader = dataloader_obj.get_file_data_loader()

        # new_preds = []
        # new_filenames = []

        # for batch in tqdm(dataloader):
        #     audios = batch[0]
        #     filename = batch[1]

        #     preds.extend(self._model(audios))
        #     filenames.extend(filename)
        # ## end dataloader ##
        

        
        for file in tqdm(file_path):
            wav, _ = torchaudio.load(file)
            pred = self._model(wav)
            preds.append(pred)
            filenames.append(file)
        
        

        # def load_file(local_file):
        #     wav, _ = torchaudio.load(local_file)
        #     output = {'filename': local_file, 'tensor': wav}
        #     return output 

        # lst_preds = []
        # lst_preds.extend(Parallel(n_jobs=-1)( delayed(load_file)(local_file) for local_file in tqdm(file_path) ))
        

        # for item in tqdm(lst_preds):
        #     pred = self._model(item['tensor'])
        #     preds.append(pred)
        #     filenames.append(item['filename'])

        
        predictions = dict(zip(filenames, preds))
        

        final_preds = [{'file':key, 'transcription':value} for key, value in predictions.items()]

        return final_preds

        # def predict_in_parallel(file, model):
        #     wav, _ = torchaudio.load(file)
        #     pred = model(wav)
        #     return file, pred


        # ls =[]
        # #with parallel_backend('threading'):
        # ls.extend(Parallel(n_jobs = 2, prefer="threads", verbose=100)(delayed(predict_in_parallel)(file, self._model) for file in tqdm(file_path)))

        # gc.collect()
        # return ls

    # def predict_p(self, file_path):
    #     split_file_paths = [file_path[i:i+2] for i in range(0, len(file_path), 2)]

    #     Parallel(n_jobs=2)(delayed(self.predict)(file_p) for file_p in split_file_paths)
            

       
    # def evaluate(self, file_path, gt_path):
    #     predictions = self.predict(file_path)

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


