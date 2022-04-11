import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM
import glob
import soundfile as sf
from tqdm import tqdm

def load_model(model_id):
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)

    return processor, model


def load_model_with_lm(model_id):
    processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    model.to('cuda')
    return processor, model

def evaluate(wav_file):
    audio_input, sample_rate = sf.read(wav_file)
    inputs = processor(audio_input, sampling_rate=16_000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to('cuda')).logits

    pred_ids = torch.argmax(logits, dim=-1)
    trans = processor.decode(pred_ids, skip_special_tokens=True)
    return trans

def evaluate_with_lm(wav_file):
    audio_input, sample_rate = sf.read(wav_file)
    inputs = processor(audio_input, sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(**inputs.to('cuda')).logits
    int_result = processor.batch_decode(logits.cpu().numpy())

    trans =  int_result.text

    del int_result
    torch.cuda.empty_cache()

    return trans


def evaluate_dir(local_dir, target_dest, with_lm = True):
    local_files = glob.glob(local_dir + '/*wav')
    for lfile in tqdm(local_files):
        if with_lm:
            trans = evaluate_with_lm(lfile)
        else:
            trans = evaluate(lfile)

        filename = lfile.split('/')[-1].replace('.wav', '.txt')
        with open(target_dest + '/' + filename, mode='w+', encoding='utf-8') as ofile:
            ofile.write(trans[0])



model_id = 'Harveenchadha/hindi_model_with_lm_vakyansh'
with_lm = True
local_dir = '/home/harveen/indic_wav2vec/IndicWav2Vec/w2v_inference/scripts/taarini_without_numbers'
target_dest = '/home/harveen/indic_wav2vec/hindi_tarini_evaluation_hf/predicted/'

if with_lm : 
    processor, model = load_model_with_lm(model_id)
else:
    processor, model = load_model(model_id)    

evaluate_dir(local_dir, target_dest, with_lm)
