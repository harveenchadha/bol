import soundfile as sf
import torch
import torch.nn.functional as F
from fairseq.data import Dictionary
from bol.utils.helper_functions import move_to_cuda, get_audio_duration
# # from ._vad_for_long_audios import call_vad
# from bol.data import Wav2VecDataLoader
# from ._wav2vec2_infer_batch import get_results_for_batch
# import os

def get_feature(filepath):
    def postprocess(feats, sample_rate):
        if feats.dim == 2:
            feats = feats.mean(-1)

        assert feats.dim() == 1, feats.dim()

        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
        return feats

    wav, sample_rate = sf.read(filepath)
    feats = torch.from_numpy(wav).float()
    feats = postprocess(feats, sample_rate)
    return feats

def post_process(sentence: str, symbol: str):
    if symbol == "sentencepiece":
        sentence = sentence.replace(" ", "").replace("\u2581", " ").strip()
    elif symbol == 'wordpiece':
        sentence = sentence.replace(" ", "").replace("_", " ").strip()
    elif symbol == 'letter':
        sentence = sentence.replace(" ", "").replace("|", " ").strip()
    elif symbol == "_EOW":
        sentence = sentence.replace(" ", "").replace("_EOW", " ").strip()
    elif symbol is not None and symbol != 'none':
        sentence = (sentence + " ").replace(symbol, "").rstrip()
    return sentence



def get_results_for_single_file(wav_path,dict_path,generator,model,use_cuda=False, half=None):
    sample = dict()
    net_input = dict()
    target_dict = Dictionary.load(dict_path)

    # ## Experimental

    # duration = get_audio_duration(wav_path)
    # file_paths = []
    # if duration > 15:
    #     file_paths = call_vad(wav_path)

    #     filtered_file_paths = []
    #     # for file in file_paths:
    #     #     if get_audio_duration(file) > 15:
    #     #         print("Skipping ", file)
    #     #         os.system('rm '+file)
    #     #     else:
    #     #         filtered_file_paths.append(file)

    #     file_path = "/".join(file_paths[0].split('/')[:-1])
    #     dataloader_obj = Wav2VecDataLoader(train_batch_size = 4, num_workers= 4 ,file_data_path = file_path)
    #     dataloader = dataloader_obj.get_file_data_loader()


    #     text = get_results_for_batch(dataloader, dict_path, generator, model, use_cuda)
    #     complete_text = []
    #     local_dict = {}

    #     for filename, local_text in text:
    #         local_file=int(filename.split('/')[-1].split('.')[0].split('-')[1])
    #         local_dict[local_file] = local_text

    #     sorted_dict = dict(sorted(local_dict.items()))
    #     print(sorted_dict)
    #     for key, value in sorted_dict.items():
    #         complete_text.append(value)
    #     print(complete_text)

    #     print(" ".join(complete_text))
    #     return complete_text

    # ## Experimental


    feature = get_feature(wav_path)
 
    model.eval()
           
    if half:
        net_input["source"] = feature.unsqueeze(0).half()
    else:
        net_input["source"] = feature.unsqueeze(0)

    padding_mask = torch.BoolTensor(net_input["source"].size(1)).fill_(False).unsqueeze(0)

    net_input["padding_mask"] = padding_mask
    sample["net_input"] = net_input
    sample = move_to_cuda(sample) if use_cuda else sample

    with torch.no_grad():
        hypo = generator.generate(model, sample, prefix_tokens=None)
    hyp_pieces = target_dict.string(hypo[0][0]["tokens"].int().cpu())
    text=post_process(hyp_pieces, 'letter')

    return text
