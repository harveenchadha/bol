import soundfile as sf
import torch
import torch.nn.functional as F
from fairseq.data import Dictionary
from tqdm import tqdm
from joblib import Parallel, delayed
import time
from joblib.externals.loky import set_loky_pickler
from joblib import parallel_backend

def post_process_sentence(sentence: str, symbol: str):
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

def postprocess_features(feats, sample_rate):
    if feats.dim() == 2:
        feats = feats.mean(-1)

    assert feats.dim() == 1, feats.dim()

    with torch.no_grad():
        feats = F.layer_norm(feats, feats.shape)
    return feats

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


def process_batch_element(element, model, generator, target_dict, use_cuda=False, input_half=False):
    sample = dict()
    net_input = dict()
    feature = element
    padding_mask = torch.BoolTensor(feature.size(1)).fill_(False).unsqueeze(0)

    if use_cuda:
        net_input["source"] = feature.cuda()
        net_input["padding_mask"] = padding_mask.cuda()
    else:
        net_input["source"] = feature
        net_input["padding_mask"] = padding_mask

    if input_half:
        net_input["source"] = net_input["source"].half()

    sample["net_input"] = net_input
    sample = move_to_cuda(sample) if use_cuda else sample
    with torch.no_grad():
        hypo = generator.generate(model, sample, prefix_tokens=None)

    hyp_pieces = [target_dict.string(item[0]["tokens"].int().cpu()) for item in hypo]
    prediction = [post_process_sentence(item, 'letter') for item in hyp_pieces]
    
    return prediction


def process_batch(batch,model,generator, target_dict, use_cuda, half):
    prediction = process_batch_element(batch[0], model=model, generator=generator, target_dict=target_dict, use_cuda=use_cuda, input_half=half)
    filenames = batch[1]
    return prediction, filenames

def get_results_for_batch(data_loader,dict_path,generator,model,use_cuda=False,w2v_path=None, half=None):
    use_cuda=True
    predictions = []
    filenames = []

    model.eval()
    target_dict = Dictionary.load(dict_path)

    for batch in tqdm(data_loader):
        prediction, filename = process_batch(batch, model=model, generator=generator, target_dict=target_dict, use_cuda=use_cuda, half=half)
        predictions.append(prediction)
        filenames.append(filename)
    
    local_dict = []
    for pred, file in zip(predictions, filenames):
        for local_pred, local_file in zip(pred, file):
            local_dict.append((local_file, local_pred))

    return local_dict
