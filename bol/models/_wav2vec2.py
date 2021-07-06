from fairseq.models import BaseFairseqModel
from fairseq.data import Dictionary
from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecEncoder, Wav2Vec2CtcConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq import utils
import torch

class Wav2VecCtc(BaseFairseqModel):
    def __init__(self, cfg: Wav2Vec2CtcConfig, w2v_encoder: BaseFairseqModel):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: Wav2Vec2CtcConfig, target_dictionary): ##change here
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
            logits[padding][...,0] = 0
            logits[padding][...,1:] = float('-inf')

        return logits

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x



# def load_model_wav2vec(model_path):
#     model =  torch.load(model_path)#,map_location=torch.device("cuda"))
#     print("Model Loaded")
#     return model


