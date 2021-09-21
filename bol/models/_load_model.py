

try:
    from .wav2vec2._wav2vec2_fairseq import Wav2vec2Fairseq
except:
     pass
from bol.utils.model_zoo import (get_model_from_local, get_model_from_params,
                                 get_model_from_unique_code,
                                 setup_language_model_on_local,
                                 setup_model_on_local)

from .wav2vec2._wav2vec2_ts import Wav2Vec2TS


def load_model(
    unique_code=None,
    lang=None,
    backend=None,
    algo=None,
    force_download=False,
    use_cuda_if_available=True,
    use_lm=True,
    **kwargs
):

    model_obj = {}
    if unique_code:
        # get_model_from_unique_code()
        model_obj = get_model_from_unique_code(unique_code)
    elif lang and backend and algo:
        unique_code, model_obj = get_model_from_params(lang, backend, algo)

    if model_obj:
        model_paths = setup_model_on_local(unique_code, model_obj, force_download)
        print("Zoo: ", model_paths)

        if model_obj["backend"] == "torchscript":
            use_lm = False
        if use_lm and len(model_obj["urls"]["lm_url"]) > 0:
            lm_paths = setup_language_model_on_local(unique_code, model_obj)
            print("Zoo: ", lm_paths)
    else:
        model_paths, lm_paths = get_model_from_local(backend, algo, use_lm, args=kwargs)
        model_obj = {}
        model_obj["algo"] = algo
        model_obj["backend"] = backend
        model_obj["urls"] = {"model_url": model_paths, "lm_url": lm_paths}
        print("Local:", model_paths)

    if not model_paths:
        raise Exception(
            "The model is not present in bol model zoo and you have not specified a local model to load as well."
        )

    if model_obj["algo"] == "wav2vec2":
        if model_obj["backend"] == "torchscript":
            model = Wav2Vec2TS(model_paths[0], use_cuda_if_available)
            return model

        if model_obj["backend"] == "fairseq":

            for item in model_paths:
                if item.endswith("pt"):
                    model_path = item
                elif item.endswith("ltr.txt"):
                    dict_path = item

            lm_path = None
            lexicon_path = None
            if use_lm and len(model_obj["urls"]["lm_url"]) > 0:
                for item in lm_paths:
                    if item.endswith("binary"):
                        lm_path = item
                    elif item.endswith("lst"):
                        lexicon_path = item

            model = Wav2vec2Fairseq(
                model_path, dict_path, lm_path, lexicon_path, use_cuda_if_available
            )
            return model
