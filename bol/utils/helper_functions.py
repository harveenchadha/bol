import subprocess
import sys

import soundfile as sf
import torch
import scipy.signal as sps
from scipy.io import wavfile
try:
    import torchaudio
except:
    pass
import subprocess
from .resampler import resample_using_ffmpeg



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


# def convert_audio_to_wav16(file_path):
#     file_name = file_path.split('/')[-1][:-4]
#     file_name_16 = '/tmp/'+file_name+'_16.wav'
#     if os.path.isfile(file_name_16):
#         print("File already present not converting")
#         return file_name_16
#     #subprocess.call(["sox {} -r {} -b 16 -c 1 {}".format(file_path, str(16000), file_name_16)], shell=True)

#     raw_audio = AudioSegment.from_file(file_path)
#     raw_audio.export(file_name_16, format="wav", parameters=["-ac", "1", "-ar", "16000"])
#     return file_name_16


# def check_if_file_is_ok(file_path):
#     #duration > 0
#     #sample_rate = 16000
#     #channel 1

#     if file_path[-3:] != 'wav':
#         print("Not a wav file")
#         return False

#     dict_sox = sox.file_info.info(file_path)
#     if dict_sox['channels'] > 1 or dict_sox['sample_rate']!=16000 or dict_sox['duration'] == 0:
#         print(dict_sox)
#         return False
#     else:
#         return True


# def validate_file(file_path):
#     if not check_if_file_is_ok(file_path):
#         print("Converting file")
#         file_path_16 = convert_audio_to_wav16(file_path)
#         return file_path_16
#     else:
#         return file_path


def get_directory_report():
    pass


def read_txt_file(txt_path):
    with open(txt_path, mode="r", encoding="utf-8") as file:
        text = file.read()
        return text


def get_audio_duration(path):
    return sf.info(path).duration

def get_sample_rate(path):
    return sf.info(path).samplerate

def read_wav_file(path, type):
    if type == 'sf':
        return sf.read(path)
    if type == 'ta':
        return torchaudio.load(path)

def write_audio_file(path, wav, sample_rate, type):
    if type=='ta':
        torchaudio.save(path, wav, sample_rate)

def convert_audio(wav_file, sample_rate, downsample_rate, type):
    if type=='ta':
        return torchaudio.transforms.Resample(sample_rate, downsample_rate, resampling_method='kaiser_window')(wav_file)
    if type=='sox':
        subprocess.call(["sox {} -r {} -b 16 -c 1 {}".format(wav_file, str(16000), '/tmp/' + wav_file.split('/')[-1])], shell=True)

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def convert_audio_using_scipy(signal, sample_rate):
    new_sample_rate = 16000
    # signal = signal.mean(-1)                                                                                                                           
    number_of_samples = round(len(signal) * float(new_sample_rate) / sample_rate)                                                                                   
    resampled_signal = sps.resample(signal, number_of_samples)
    return resampled_signal

def convert_audio_to_16k(wav_file):
    # signal, sample_rate  = sf.read(wav_file)
    sample_rate, signal = wavfile.read(wav_file)                                                                                                                        
    #signal = signal.mean(-1)
    new_sample_rate = 16000                                                                                                                              
    number_of_samples = round(len(signal) * float(new_sample_rate) / sample_rate)                                                                                   
    resampled_signal = sps.resample(signal, number_of_samples)
    return resampled_signal

def convert_to_tensor(arr):
    return torch.from_numpy(arr).float().unsqueeze(0)

def convert_mp3_to_wav(input_file, output_file=None, resample=False):
    if input_file[-3:] !='mp3':
        raise ValueError("File is not mp3")

    if not output_file:
        output_file = '/tmp/' + input_file.split('/')[-1][:-3] + 'wav'

    
    if resample:
        output_file = resample_using_ffmpeg(input_file=input_file, output_file=output_file)
    else:
        subprocess.call(['ffmpeg', '-y', '-i', input_file, output_file], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    return output_file
