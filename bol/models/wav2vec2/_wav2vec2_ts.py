from tqdm import tqdm
from bol.utils.helper_functions import  read_wav_file, convert_to_tensor
from .._model import BolModel
from bol.data import Wav2Vec2TsDataLoader
from bol.utils.resampler import resample_using_sox


class Wav2Vec2TS(BolModel):
    def __init__(self, model_path, use_cuda_if_available):
        super().__init__(model_path, "False")
        self.load_jit_model()

    def predict_for_files(self, file_path, verbose, dataloader=False, convert=False):
        preds = []
        filenames = []

        if verbose:
            disable = False
        else:
            disable = True


        if dataloader:
            dataloader_obj = Wav2Vec2TsDataLoader(batch_size = 1, num_workers = 1 ,file_data_path = file_path, convert=convert)
            dataloader = dataloader_obj.get_file_data_loader()

            for batch in tqdm(dataloader, disable=disable):
                wav = batch[0].squeeze(1)
                file = batch[1]
                pred = self._model(wav)
                preds.append(pred)
                filenames.extend(file)

        else:

            for file in tqdm(file_path, disable=disable):
                wav, sample_rate = read_wav_file(file, 'sf')
                if sample_rate != 16000 and convert:
                    wav = resample_using_sox(wav, input_type='array', output_type='array', sample_rate_in=sample_rate)
                
                wav = convert_to_tensor(wav)
                pred = self._model(wav)
                preds.append(pred)
                filenames.append(file)

        return preds, filenames

    def predict(
        self,
        file_path,
        with_lm=False,
        return_filenames=True,
        apply_vad=False,
        verbose=0,
        convert=False
    ):
        if type(file_path) == str:
            file_path = [file_path]

        preds = []
        filenames = []


        if apply_vad:
            for file in file_path:
                files_split_from_vad = self.preprocess_vad(file)
                preds_local, filenames_local = self.predict_for_files(
                    files_split_from_vad, verbose=verbose, convert=False
                )
                predictions = self.postprocess_vad(filenames_local, preds_local)
                preds.append(predictions)
                filenames.append(file)

        else:
            preds, filenames = self.predict_for_files(file_path, verbose=verbose, convert=convert)

        predictions = dict(zip(filenames, preds))

        if return_filenames:
            final_preds = [
                {"file": key, "transcription": value}
                for key, value in predictions.items()
            ]
        else:
            final_preds = preds

        return final_preds