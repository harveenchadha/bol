from .wav2vec2._wav2vec2_fairseq import Wav2vec2Fairseq
from .wav2vec2._wav2vec2_ts import Wav2Vec2TS
from bol.utils.model_zoo import get_model_from_params, get_model_from_unique_code, get_model_from_local, setup_model_on_local,setup_language_model_on_local

def load_model( unique_code = None, 
                    lang = None ,
                    backend = None,
                    algo = None,
                    force_download = False,
                    use_cuda_if_available = True,
                    use_lm = True,
                    **kwargs
                   ):

    model_obj = {}
    if unique_code:
        #get_model_from_unique_code()
        model_obj = get_model_from_unique_code(unique_code)
    elif lang and backend and algo:
        unique_code , model_obj = get_model_from_params(lang, backend, algo)
        

    if model_obj:
        model_paths = setup_model_on_local(unique_code, model_obj, force_download)
        print("Zoo: ", model_paths)

        if use_lm and len(model_obj['urls']['lm_url']) > 0:
            lm_paths = setup_language_model_on_local(unique_code, model_obj)
            print("Zoo: ", lm_paths)
    else:
        model_paths = get_model_from_local(backend, algo, use_lm, args = kwargs)
        print("Local:" , model_paths)


    if model_obj['algo'] == 'wav2vec2':
        if model_obj['backend'] == 'torchscript':
            model = Wav2Vec2TS(model_paths[0], use_cuda_if_available)
            return model

        if model_obj['backend'] == 'fairseq':
            
            for item in model_paths:
                if item.endswith('pt'):
                    model_path = item
                elif item.endswith('ltr.txt'):
                    dict_path = item

            lm_path = None
            lexicon_path = None
            if use_lm and len(model_obj['urls']['lm_url']) > 0:
                for item in lm_paths:
                    if item.endswith('binary'):
                        lm_path = item
                    elif item.endswith('lst'):
                        lexicon_path = item
                

            model = Wav2vec2Fairseq(model_path, dict_path, lm_path, lexicon_path, use_cuda_if_available)
            return model