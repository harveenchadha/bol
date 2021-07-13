from .cer import calculate_cer
from .wer import calculate_wer
from .calculate_metrics import calculate_metrics_for_single_file, calculate_metrics_for_batch
from .wer import wer
from .cer import cer

__all__=[
    'calculate_metrics_for_single_file',
    'calculate_metrics_for_batch',
    'wer',
    'cer'
]