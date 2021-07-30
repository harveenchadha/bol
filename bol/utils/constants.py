

MODEL_MAPPING={
    'hi-IN' : { 'lang_code':'hi-IN',
                'backend' : 'fairseq',
                'algo':'wav2vec2',
                'description':'Hindi bol model',
                'provider':'ekstep',
                'urls':{
                    'model_url':['https://storage.googleapis.com/vakyaansh-open-models/compressed_v2/hindi/hi-IN_model.tar.xz'],
                    'lm_url':['https://storage.googleapis.com/vakyaansh-open-models/compressed_v2/hindi/hi-IN_lm.tar.xz']
                }
              },

    'bn-IN' : { 'lang_code':'bn-IN',
                'backend' : 'fairseq',
                'algo':'wav2vec2',
                'description':'Bengali bol model',
                'provider':'ekstep',
                'urls':{
                    'model_url':['https://storage.googleapis.com/vakyaansh-open-models/compressed_v2/bengali/bn-IN_model.tar.xz'],
                    'lm_url':['https://storage.googleapis.com/vakyaansh-open-models/compressed_v2/bengali/bn-IN_lm.tar.xz']
                }
              },


    'en-IN' : { 'lang_code':'en-IN',
                'backend' : 'fairseq',
                'algo':'wav2vec2',
                'description':'Indian English bol model',
                'provider':'ekstep',
                'urls':{
                    'model_url':['https://storage.googleapis.com/vakyaansh-open-models/compressed_v2/english/en-IN_model.tar.xz'],
                    'lm_url':['https://storage.googleapis.com/vakyaansh-open-models/compressed_v2/english/en-IN_lm.tar.xz']
                }
              },


    'gu-IN' : { 'lang_code':'gu-IN',
                'backend' : 'fairseq',
                'algo':'wav2vec2',
                'description':'Gujarati bol model',
                'provider':'ekstep',
                'urls':{
                    'model_url':['https://storage.googleapis.com/vakyaansh-open-models/compressed_v2/gujarati/gu-IN_model.tar.xz'],
                    'lm_url':['https://storage.googleapis.com/vakyaansh-open-models/compressed_v2/gujarati/gu-IN_lm.tar.xz']
                }
              },


    'kn-IN' : { 'lang_code':'kn-IN',
                'backend' : 'fairseq',
                'algo':'wav2vec2',
                'description':'Kannada bol model',
                'provider':'ekstep',
                'urls':{
                    'model_url':['https://storage.googleapis.com/vakyaansh-open-models/compressed_v2/kannada/kn-IN_model.tar.xz'],
                    'lm_url':['https://storage.googleapis.com/vakyaansh-open-models/compressed_v2/kannada/kn-IN_lm.tar.xz']
                }
              },


    'ne-IN' : { 'lang_code':'ne-IN',
                'backend' : 'fairseq',
                'algo':'wav2vec2',
                'description':'Nepali bol model',
                'provider':'ekstep',
                'urls':{
                    'model_url':['https://storage.googleapis.com/vakyaansh-open-models/compressed_v2/nepali/ne-IN_model.tar.xz'],
                    'lm_url':['https://storage.googleapis.com/vakyaansh-open-models/compressed_v2/nepali/ne-IN_lm.tar.xz']
                }
              },


    'ta-IN' : { 'lang_code':'ta-IN',
                'backend' : 'fairseq',
                'algo':'wav2vec2',
                'description':'Tamil bol model',
                'provider':'ekstep',
                'urls':{
                    'model_url':['https://storage.googleapis.com/vakyaansh-open-models/compressed_v2/tamil/ta-IN_model.tar.xz'],
                    'lm_url':['https://storage.googleapis.com/vakyaansh-open-models/compressed_v2/tamil/ta-IN_lm.tar.xz']
                }
              },

    'te-IN' : { 'lang_code':'te-IN',
                'backend' : 'fairseq',
                'algo':'wav2vec2',
                'description':'Telugu bol model',
                'provider':'ekstep',
                'urls':{
                    'model_url':['https://storage.googleapis.com/vakyaansh-open-models/compressed_v2/telugu/te-IN_model.tar.xz'],
                    'lm_url':['https://storage.googleapis.com/vakyaansh-open-models/compressed_v2/telugu/te-IN_lm.tar.xz']
                }
              },


    # 'hi-IN-old' : { 'lang_code':'hi-IN-old',
    #             'description':'Hindi bol model',
    #             'provider':'ekstep',
    #             'urls':{
    #                 'model_url':['https://storage.googleapis.com/vakyaansh-open-models/hindi/v1/model_v1.zip'],
    #                 'lm_url':['https://storage.googleapis.com/vakyaansh-open-models/hindi/v1/lm_v1.zip']
    #             }
    #           }, 
}

MODEL_PATH='/.bol/models'