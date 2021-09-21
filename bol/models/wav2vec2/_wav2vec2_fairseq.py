import time

from fairseq import utils
from fairseq.models import BaseFairseqModel
from fairseq.models.wav2vec.wav2vec2_asr import (Wav2Vec2CtcConfig,
                                                 Wav2VecEncoder)

from bol.data import Wav2Vec2FDataLoader
from bol.inference import (get_results_for_batch, get_results_for_single_file,
                           load_decoder)

from .._model import BolModel


class Wav2VecCtc(BaseFairseqModel):
    def __init__(self, cfg: Wav2Vec2CtcConfig, w2v_encoder: BaseFairseqModel):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def build_model(cls, cfg: Wav2Vec2CtcConfig, target_dictionary):  ##change here
        """Build a new model instance."""
        w2v_encoder = Wav2VecEncoder(cfg, target_dictionary)
        return cls(cfg, w2v_encoder)

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_logits(self, net_output):
        logits = net_output["encoder_out"]
        padding = net_output["padding_mask"]
        if padding is not None and padding.any():
            padding = padding.T
            logits[padding][..., 0] = 0
            logits[padding][..., 1:] = float("-inf")

        return logits

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x


class Wav2vec2Fairseq(BolModel):
    def __init__(
        self, model_path, dict_path, lm_path, lexicon_path, use_cuda_if_available=True
    ):
        super().__init__(model_path, use_cuda_if_available)

        self.dict_path = dict_path
        self.lm_path = lm_path
        self.lexicon_path = lexicon_path
        self._alternative_decoder = "viterbi"
        self.use_cuda_if_available = use_cuda_if_available
        self.device = "cpu"
        self.load_model_torch()

        if lm_path and lexicon_path:
            self.load_decoder()
        else:
            self._decoder = load_decoder(self.dict_path, "", "", "viterbi")
            self._alternative_decoder = self._decoder

    def load_decoder(self):
        start = time.time()

        # if os.path.exists(model_path + '/kenlm_decoder.pkl'):
        #     print("Loading decoder from cache")
        #     with open(model_path + '/kenlm_decoder.pkl', 'rb') as input:
        #         decoder = pickle.load(input)
        #         self._decoder = decoder
        # else:

        self._decoder = load_decoder(
            self.dict_path, self.lexicon_path, self.lm_path, "kenlm"
        )

        # with open(model_path  + '/kenlm_decoder.pkl', 'wb') as output:
        #     pickle.dump([self._decoder], output, pickle.HIGHEST_PROTOCOL)

        end = time.time()
        print("Decoder Loaded in " + str(end - start) + " seconds")
        self._alternative_decoder = load_decoder(
            self.dict_path, self.lexicon_path, self.lm_path, "viterbi"
        )
        time.time()
        # print('Viterbi Loaded in '+ str(end_viterbi-end) + ' seconds')

    def predict_in_batch(self, file_paths, can_use_lm, verbose):
        # Hardcoding #
        dataloader_obj = Wav2Vec2FDataLoader(
            train_batch_size=1, num_workers=2, file_data_path=file_paths
        )
        dataloader = dataloader_obj.get_file_data_loader()

        if not can_use_lm:
            filenames, predictions = get_results_for_batch(
                dataloader,
                self.dict_path,
                self.get_alternative_decoder(),
                self.get_model(),
                self.use_cuda_if_available,
                verbose=verbose,
            )
        else:
            filenames, predictions = get_results_for_batch(
                dataloader,
                self.dict_path,
                self.get_decoder(),
                self.get_model(),
                self.use_cuda_if_available,
                verbose=verbose,
            )

        return filenames, predictions

    def predict(
        self,
        file_path,
        with_lm=False,
        return_filenames=True,
        apply_vad=False,
        verbose=0,
        convert=False
    ):
        text = ""

        can_use_lm = with_lm and self.lm_path

        if type(file_path) == str:
            file_path = [file_path]

        if len(file_path) == 1:
            if not can_use_lm and not apply_vad:
                text = get_results_for_single_file(
                    file_path[0],
                    self.dict_path,
                    self.get_alternative_decoder(),
                    self.get_model(),
                    self.use_cuda_if_available,
                )
            else:
                ## experimental

                if apply_vad:
                    file_paths = []
                    for local_file in file_path:
                        file_paths.extend(self.preprocess_vad(local_file))
                    # print(file_paths)

                    filenames, predictions = self.predict_in_batch(
                        file_paths, can_use_lm, verbose
                    )

                    preds = self.postprocess_vad(filenames, predictions)
                    text = preds

                else:
                    ##experimental
                    text = get_results_for_single_file(
                        file_path[0],
                        self.dict_path,
                        self.get_decoder(),
                        self.get_model(),
                        self.use_cuda_if_available,
                    )

            if return_filenames:
                final_preds = [{"file": file_path[0], "transcription": text}]
            else:
                final_preds = text
        else:
            filenames, predictions = self.predict_in_batch(
                file_path, can_use_lm, verbose
            )
            preds = zip(filenames, predictions)
            if return_filenames:
                final_preds = [
                    {"file": key, "transcription": value} for key, value in preds
                ]
            else:
                final_preds = predictions

        return final_preds
