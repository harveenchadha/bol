

MODEL_MAPPING={
    'hi-IN' : { 'lang_code':'hi-IN',
                'description':'Hindi bol model',
                'provider':'ekstep',
                'urls':{
                    'model_url':['https://storage.googleapis.com/vakyaansh-open-models/hindi/v2/hindi_v2.zip'],
                    'lm_url':['https://storage.googleapis.com/vakyaansh-open-models/hindi/v2/hindi_v2.zip']
                }
              },
    'hi-IN-old' : { 'lang_code':'hi-IN-old',
                'description':'Hindi bol model',
                'provider':'ekstep',
                'urls':{
                    'model_url':['https://storage.googleapis.com/vakyaansh-open-models/hindi/v1/model_v1.zip'],
                    'lm_url':['https://storage.googleapis.com/vakyaansh-open-models/hindi/v1/lm_v1.zip']
                }
              }, 
}

MODEL_PATH='/.bol/models'