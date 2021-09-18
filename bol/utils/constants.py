file = [
    "unique_code,lang_code,backend,algo,description,provider,url_model,url_lm,contributed_by,citation",
    "hi-ts,hi-IN,torchscript,wav2vec2,Hindi,vakyansh_ekstep,[https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main/vakyansh-wav2vec2-hindi-him-4200_quant.pt_tar.xz],,Harveen,",
    "bn-ts,bn-IN,torchscript,wav2vec2,Bengali,vakyansh_ekstep,[https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main/vakyansh-wav2vec2-bengali-bnm-200_quant.pt_tar.xz],,Harveen,",
    "gn-ts,gn-IN,torchscript,wav2vec2,Gujarati,vakyansh_ekstep,[https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main/vakyansh-wav2vec2-gujarati-gnm-100_quant.pt_tar.xz],,Harveen,",
    "en-in-ts,en-IN,torchscript,wav2vec2,Indian English,vakyansh_ekstep,[https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main/vakyansh-wav2vec2-indian-english-enm-700_quant.pt_tar.xz],,Harveen,",
    "kn-ts,kn-IN,torchscript,wav2vec2,Kannada,vakyansh_ekstep,[https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main/vakyansh-wav2vec2-kannada-knm-560_quant.pt_tar.xz],,Harveen,",
    "ne-ts,ne-IN,torchscript,wav2vec2,Nepali,vakyansh_ekstep,[https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main/vakyansh-wav2vec2-nepali-nem-130_quant.pt_tar.xz],,Harveen,",
    "ta-ts,ta-IN,torchscript,wav2vec2,Tamil,vakyansh_ekstep,[https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main/vakyansh-wav2vec2-tamil-tam-250_quant.pt_tar.xz],,Harveen,",
    "te-ts,te-IN,torchscript,wav2vec2,Telugu,vakyansh_ekstep,[https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main/vakyansh-wav2vec2-telugu-tem-100_quant.pt_tar.xz],,Harveen,",
    "hi-vakyansh,hi-IN,fairseq,wav2vec2,Hindi,vakyansh_ekstep,[https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main/hi-IN_model.tar.xz],[https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main/hi-IN_lm.tar.xz],Harveen,",
    "bn-vakyansh,bn-IN,fairseq,wav2vec2,Bengali,vakyansh_ekstep,[https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main/bn-IN_model.tar.xz],[https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main/bn-IN_lm.tar.xz],Harveen,",
    "en-vakyansh,en-IN,fairseq,wav2vec2,English,vakyansh_ekstep,[https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main/en-IN_model.tar.xz],[https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main/en-IN_lm.tar.xz],Harveen,",
    "gu-vakyansh,gu-IN,fairseq,wav2vec2,Gujarati,vakyansh_ekstep,[https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main/gu-IN_model.tar.xz],[https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main/gu-IN_lm.tar.xz],Harveen,",
    "kn-vakyansh,kn-IN,fairseq,wav2vec2,Kannada,vakyansh_ekstep,[https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main/kn-IN_model.tar.xz],[https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main/kn-IN_lm.tar.xz],Harveen,",
    "ne-vakyansh,ne-IN,fairseq,wav2vec2,Nepali,vakyansh_ekstep,[https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main/ne-IN_model.tar.xz],[https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main/ne-IN_lm.tar.xz],Harveen,",
    "ta-vakyansh,ta-IN,fairseq,wav2vec2,Tamil,vakyansh_ekstep,[https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main/ta-IN_model.tar.xz],[https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main/ta-IN_lm.tar.xz],Harveen,",
    "te-vakyansh,te-IN,fairseq,wav2vec2,Telugu,vakyansh_ekstep,[https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main/te-IN_model.tar.xz],[https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main/te-IN_lm.tar.xz],Harveen,",
]

# with open('./modelzoo.csv') as file:
models = {}
attrs = []
for indx, line in enumerate(file):
    if indx == 0:
        attrs = line.split(",")
    else:
        line_model = line.split(",")
        # line_model = [literal_eval(line) for line in line_model if line[0] == '[']
        models[line_model[0]] = dict(zip(attrs[1:], line_model[1:]))
        models[line_model[0]]["urls"] = {
            "model_url": models[line_model[0]]["url_model"][1:-1].split(","),
            "lm_url": models[line_model[0]]["url_lm"][1:-1].split(","),
        }

MODEL_MAPPING = models

# MODEL_MAPPING={
#     'hi-quant' : { 'lang_code' : 'hi-IN',
#                 'backend' : 'torchscript',
#                 'algo' : 'wav2vec2',
#                 'description': 'Hindi Quantized Model',
#                 'provider' : 'ekstep',
#                 'urls' : {
#                     'model_url' : ['https://storage.googleapis.com/vakyaansh-open-models/quant_mobile/hindi/wav2vec2.pt'],
#                     'lm_url' : []
#                 },
#                 'contributed_by' : '',
#                 'citation' : ''
#               },

#     'hi-IN' : { 'lang_code':'hi-IN',
#                 'backend' : 'fairseq',
#                 'algo':'wav2vec2',
#                 'description':'Hindi bol model',
#                 'provider':'ekstep',
#                 'urls':{
#                     'model_url':['https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main//hindi/hi-IN_model.tar.xz'],
#                     'lm_url':['https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main//hindi/hi-IN_lm.tar.xz']
#                 }
#               },

#     'bn-IN' : { 'lang_code':'bn-IN',
#                 'backend' : 'fairseq',
#                 'algo':'wav2vec2',
#                 'description':'Bengali bol model',
#                 'provider':'ekstep',
#                 'urls':{
#                     'model_url':['https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main//bengali/bn-IN_model.tar.xz'],
#                     'lm_url':['https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main//bengali/bn-IN_lm.tar.xz']
#                 }
#               },


#     'en-IN' : { 'lang_code':'en-IN',
#                 'backend' : 'fairseq',
#                 'algo':'wav2vec2',
#                 'description':'Indian English bol model',
#                 'provider':'ekstep',
#                 'urls':{
#                     'model_url':['https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main//english/en-IN_model.tar.xz'],
#                     'lm_url':['https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main//english/en-IN_lm.tar.xz']
#                 }
#               },


#     'gu-IN' : { 'lang_code':'gu-IN',
#                 'backend' : 'fairseq',
#                 'algo':'wav2vec2',
#                 'description':'Gujarati bol model',
#                 'provider':'ekstep',
#                 'urls':{
#                     'model_url':['https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main//gujarati/gu-IN_model.tar.xz'],
#                     'lm_url':['https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main//gujarati/gu-IN_lm.tar.xz']
#                 }
#               },


#     'kn-IN' : { 'lang_code':'kn-IN',
#                 'backend' : 'fairseq',
#                 'algo':'wav2vec2',
#                 'description':'Kannada bol model',
#                 'provider':'ekstep',
#                 'urls':{
#                     'model_url':['https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main//kannada/kn-IN_model.tar.xz'],
#                     'lm_url':['https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main//kannada/kn-IN_lm.tar.xz']
#                 }
#               },


#     'ne-IN' : { 'lang_code':'ne-IN',
#                 'backend' : 'fairseq',
#                 'algo':'wav2vec2',
#                 'description':'Nepali bol model',
#                 'provider':'ekstep',
#                 'urls':{
#                     'model_url':['https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main//nepali/ne-IN_model.tar.xz'],
#                     'lm_url':['https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main//nepali/ne-IN_lm.tar.xz']
#                 }
#               },


#     'ta-IN' : { 'lang_code':'ta-IN',
#                 'backend' : 'fairseq',
#                 'algo':'wav2vec2',
#                 'description':'Tamil bol model',
#                 'provider':'ekstep',
#                 'urls':{
#                     'model_url':['https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main//tamil/ta-IN_model.tar.xz'],
#                     'lm_url':['https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main//tamil/ta-IN_lm.tar.xz']
#                 }
#               },

#     'te-IN' : { 'lang_code':'te-IN',
#                 'backend' : 'fairseq',
#                 'algo':'wav2vec2',
#                 'description':'Telugu bol model',
#                 'provider':'ekstep',
#                 'urls':{
#                     'model_url':['https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main//telugu/te-IN_model.tar.xz'],
#                     'lm_url':['https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main//telugu/te-IN_lm.tar.xz']
#                 }
#               },

#     'mr-IN' : { 'lang_code':'mr-IN',
#                 'backend' : 'fairseq',
#                 'algo':'wav2vec2',
#                 'description':'Marathi bol model',
#                 'provider':'ekstep',
#                 'urls':{
#                     'model_url':['https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main//marathi/v3-05-08-2021_custom_model_for_single_file_inference.zip'],
#                     'lm_url':['https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main//marathi/LM.zip']
#                 }
#               },

#     'od-IN' : { 'lang_code':'od-IN',
#                 'backend' : 'fairseq',
#                 'algo':'wav2vec2',
#                 'description':'Odiya bol model',
#                 'provider':'ekstep',
#                 'urls':{
#                     'model_url':['https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main//odia/v2-26-07-2021_custom_model_for_single_file_inference.zip'],
#                     'lm_url':['https://huggingface.co/datasets/Harveenchadha/bol-models/resolve/main//odia/LM.zip']
#                 }
#               },


#     # 'hi-IN-old' : { 'lang_code':'hi-IN-old',
#     #             'description':'Hindi bol model',
#     #             'provider':'ekstep',
#     #             'urls':{
#     #                 'model_url':['https://storage.googleapis.com/vakyaansh-open-models/hindi/v1/model_v1.zip'],
#     #                 'lm_url':['https://storage.googleapis.com/vakyaansh-open-models/hindi/v1/lm_v1.zip']
#     #             }
#     #           },
# }

MODEL_PATH = "/.bol/models"

MODEL_FILES_REQ = {
    "wav2vec2_torchscript": ["*.pt"],
    "wav2vec2_fairseq": ["*.pt", "*.ltr.txt"],
    "wav2vec2_fairseq_lm": ["*.binary", "*.lst"],
}
