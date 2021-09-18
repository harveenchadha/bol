no_exception = 0

try:
    pass
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
