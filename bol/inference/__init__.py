no_exception = 0

try:
    from .decoders._wav2vec2_decoder import load_decoder
    from .wav2vec2_fairseq._wav2vec2_infer_single import get_results_for_single_file
    from .wav2vec2_fairseq._wav2vec2_infer_batch import get_results_for_batch
except:
    no_exception = 1

from ._vad_for_long_audios import call_vad

__all__ = [
    # "load_decoder",
    # "get_results_for_single_file",
    # "get_results_for_batch",
    "call_vad",
]

if no_exception == 0:
    __all__.append("load_decoder")
    __all__.append("get_results_for_single_file")
    __all__.append("get_results_for_batch")
