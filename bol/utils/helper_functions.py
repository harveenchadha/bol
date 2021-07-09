
from tempfile import NamedTemporaryFile
import subprocess
import sox
import os

def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x

    return _apply(sample)

def move_to_cuda(sample, device=None):
    device = device or torch.cuda.current_device()

    def _move_to_cuda(tensor):
        # non_blocking is ignored if tensor is not pinned, so we can always set
        # to True (see github.com/PyTorchLightning/pytorch-lightning/issues/620)
        return tensor.to(device=device, non_blocking=True)

    return apply_to_sample(_move_to_cuda, sample)

from pydub import AudioSegment

def convert_audio_to_wav16(file_path):
    file_name = file_path.split('/')[-1][:-4]
    file_name_16 = '/tmp/'+file_name+'_16.wav'
    if os.path.isfile(file_name_16):
        print("File already present not converting")
        return file_name_16 
    #subprocess.call(["sox {} -r {} -b 16 -c 1 {}".format(file_path, str(16000), file_name_16)], shell=True)

    raw_audio = AudioSegment.from_file(file_path)
    raw_audio.export(file_name_16, format="wav", parameters=["-ac", "1", "-ar", "16000"])
    return file_name_16


def check_if_file_is_ok(file_path):
    #duration > 0
    #sample_rate = 16000
    #channel 1

    if file_path[-3:] != 'wav':
        print("Not a wav file")
        return False

    dict_sox = sox.file_info.info(file_path)
    if dict_sox['channels'] > 1 or dict_sox['sample_rate']!=16000 or dict_sox['duration'] == 0:
        print(dict_sox)
        return False
    else:
        return True


def validate_file(file_path):
    if not check_if_file_is_ok(file_path):
        print("Converting file")
        file_path_16 = convert_audio_to_wav16(file_path)
        return file_path_16
    else:
        return file_path

def get_directory_report():
    pass