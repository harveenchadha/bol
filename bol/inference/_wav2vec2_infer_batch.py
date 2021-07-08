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

def process_batch_element(element, model, generator, target_dict, use_cuda=False, input_half=False):
    start = time.time()
    sample = dict()
    net_input = dict()
    # print(element)
    # print(element.shape)
    #feature = postprocess_features(element[0][0][0], element[1]).unsqueeze(0)
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

    # data_load_time = time.time() - start_time

    sample["net_input"] = net_input

    with torch.no_grad():
        hypo = generator.generate(model, sample, prefix_tokens=None)
    end = time.time()

    print("Time generator:", end-start)
    #print(hypo)
    hyp_pieces = [target_dict.string(item[0]["tokens"].int().cpu()) for item in hypo]
    #print(hyp_pieces)
    #print(len(hyp_pieces))
    prediction = [post_process_sentence(item, 'letter') for item in hyp_pieces]
    
    return prediction

# import ray
# ray.init()

#@ray.remote
def process_batch(batch,model,generator, target_dict, use_cuda, half):
    prediction = process_batch_element(batch, model=model, generator=generator, target_dict=target_dict, use_cuda=use_cuda, input_half=half)
    return prediction

def get_results_for_batch(data_loader,dict_path,generator,model,use_cuda=False,w2v_path=None, half=None):
    predictions = []
    ground_truths = []
    model.eval()
    print(data_loader)
    # batch = next(iter(data_loader))
    # print("batc len,", len(batch))

    target_dict = Dictionary.load(dict_path)

   

    # set_loky_pickler('pickle')
    # with parallel_backend('multiprocessing'):
    #     predictions = Parallel(n_jobs=-1)(delayed(process_batch)(batch,model,generator, target_dict, use_cuda, half) for batch in tqdm(data_loader)) ##doesn't works

    predicions = [process_batch(batch,model,generator, target_dict, use_cuda, half) for batch in tqdm(data_loader)]
    #ray.get(predicions)

    # for batch in tqdm(data_loader):
    #     prediction = process_batch_element(batch, model=model, generator=generator, target_dict=target_dict, use_cuda=use_cuda, input_half=half)
    #     predictions.append(prediction)
    #     #print(batch)
    #     #ground_truths.append(batch[2][0])
    return predictions

# class inference_pipeline:

#     def __init__(self,
#                  target_dict,
#                  use_cuda,
#                  input_half):

#         self.generator = W2lViterbiDecoder(target_dict)
#         self.target_dict = target_dict
#         self.use_cuda = use_cuda
#         self.input_half = input_half

#     def run_inference_pipeline(self, model, data_loader):
#         predictions = []
#         ground_truths = []
#         model.eval()

#         for i, batch in enumerate(data_loader):
#             prediction = process_batch_element(batch, model=model, generator=self.generator, target_dict=self.target_dict, use_cuda=self.use_cuda, input_half=self.input_half)
#             predictions.append(prediction)
#             ground_truths.append(batch[2][0])

#         wer_score = wer(ground_truths, predictions)
#         return {"inference_result": wer_score}

