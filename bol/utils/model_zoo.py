

# def check_if_str_is_lang():
#     pass

# def check_if_path_is_pt_or_lang_code():
#     pass
#     check_if_str_is_lang()

# def get_model_acc_to_lang():
#     pass

# def get_model_path(model_path):
#     type_model = check_if_path_is_pt_or_lang_code()
#     if type_model == 'dir':
#         pass
#     elif type_model == 'lang'

import bol.utils.constants as consts
import glob
import os
from os.path import expanduser




 

# def _get_names_of_model_files(path):
#     model_path = glob.glob(path + '/*.pt')
#     dict_path = glob.glob(path + '/*.ltr.txt')

#     if not model_path or not dict_path:
#         return None, None

#     return model_path[0], dict_path[0]

# # def _check_if_required_files_exist(model_path):    
# #     model_path, dict_path = _get_names_of_model_files(model_path)

# #     if not model_path or not dict_path:
# #         return False
# #     else:
# #         return True










    
# def verify_model_mapping(model_code, force_download):
#     if model_code in consts.MODEL_MAPPING.keys():
        
#         # check if model exists, if it does return the path, if it doesn't donwload

#         if force_download:
#             full_path = _get_model_path_from_model_code(model_code)
#             _delete_model_files(full_path)

#         full_path = _check_if_model_exists(model_code, consts.MODEL_MAPPING[model_code]['urls'])
#         model_path, dict_path = _get_names_of_model_files(full_path)
#         return model_path, dict_path
#     else:
#         print('False')




# def _get_names_of_model_files_for_lm(path):
#     lm_path = glob.glob(path + '/*.binary')
#     lexicon_path = glob.glob(path + '/*.lst')

#     if not lm_path or not lexicon_path:
#         return None, None

#     return lm_path[0], lexicon_path[0]

# def _check_if_required_files_exist_for_lm(model_path):    
#     lm_path, lexicon_path = _get_names_of_model_files_for_lm(model_path)
#     if not lm_path or not lexicon_path:
#         return False
#     else:
#         return True


# def _check_if_lm_exists(model_code, urls):
#     full_path = _get_model_path_from_model_code(model_code)
#     if os.path.exists(full_path):
#         # print("Path already exists")
#         check_files_exist = _check_if_required_files_exist_for_lm(full_path)
#         if not check_files_exist:
#             print("LM Folder exists but files not present donwloading again")
#             _download(full_path, urls, 'lm_url')
#         return full_path
#     else:
#         _download(full_path, urls, 'lm_url')
#         return full_path


# def verify_lm_mapping(model_code):
#     if model_code in consts.MODEL_MAPPING.keys():
#         # print('True')
#         # check if model exists, if it does return the path, if it doesn't donwload

#         full_path = _check_if_lm_exists(model_code, consts.MODEL_MAPPING[model_code]['urls'])
#         #get names of all the files.
#         #print(full_path)
#         lm_path, lexicon_path = _get_names_of_model_files_for_lm(full_path)
#         # print('Here123')
#         # print(lm_path, lexicon_path)
#         return lm_path, lexicon_path
#     else:
#         print('False')



########################################################################################################################################################

def _get_model_path_from_model_code(model_code):
    home = expanduser("~")
    base_path = home + consts.MODEL_PATH
    full_local_path = base_path + '/' + model_code
    return full_local_path

def _delete_model_files(local_path):
    cmd = 'rm -rf '+ local_path
    os.system(cmd)


def _get_file_name_from_url(url):
    return url.split('/')[-1]

def _get_model_files(local_path, req_files):
    files = []
    for local_file in req_files:
        files.extend( glob.glob(local_path + '/' + local_file) )

    return files
    

def _check_if_required_files_exist(local_path , req_files):    
    for local_file in req_files:
        files = glob.glob(local_path + '/' + local_file)
        if files:
            continue
        else:
            return False
    return True

def _download(local_path, urls):
    os.makedirs(local_path, exist_ok = True)
    
    for url in urls:
        wget_cmd = 'wget '+ url + ' -P '+ local_path
        filename = _get_file_name_from_url(url)
        
        unzip_cmd = ''
        if filename[-3:] == 'zip':
            unzip_cmd = 'unzip ' + local_path+'/' + filename + ' -d ' +local_path
            
        if filename[-3:] == '.xz':
            unzip_cmd = 'tar -xvf ' + local_path+'/' + filename + ' -C ' +local_path # + ' --strip-components=3'
        

        remove_cmd = 'rm ' + local_path+'/' + filename
        os.system(wget_cmd)
        os.system(unzip_cmd)
        os.system(remove_cmd)


def _check_if_model_exists(local_path, urls, req_files):
    if os.path.exists(local_path):
        print("Path already exists")
        check_files_exist = _check_if_required_files_exist(local_path , req_files)

        if not check_files_exist:
            print("Folder exists but files not present donwloading again")
            _download(local_path, urls)
    else:
        _download(local_path, urls)






def setup_model_on_local(unique_code, model_obj, force_download):
    if unique_code in consts.MODEL_MAPPING.keys():
        
        # check if model exists, if it does return the path, if it doesn't donwload

        local_path = _get_model_path_from_model_code(unique_code)
        if force_download:
            _delete_model_files(local_path)


        req_files = consts.MODEL_FILES_REQ[model_obj['algo'] + '_' + model_obj['backend']]
        _check_if_model_exists(local_path, model_obj['urls']['model_url'], req_files)
        
        model_file_local_paths = _get_model_files(local_path, req_files)
        return model_file_local_paths
    else:
        raise Exception("The unique code specified doesn't exists in the Bol model zoo.")


def setup_language_model_on_local(unique_code, model_obj):
    if unique_code in consts.MODEL_MAPPING.keys():
        local_lmpath = _get_model_path_from_model_code(unique_code)

        req_files = consts.MODEL_FILES_REQ[model_obj['algo'] + '_' + model_obj['backend'] + '_lm']
        _check_if_model_exists(local_lmpath, model_obj['urls']['lm_url'], req_files)

        lm_local_paths = _get_model_files(local_lmpath, req_files)
        return lm_local_paths
    else:
        raise Exception("The unique code specified doesn't exists in the Bol model zoo.")




def get_model_from_unique_code(unique_code):
    try:
        model_obj = consts.MODEL_MAPPING[unique_code]
        return model_obj
    except:
        return []


def get_model_from_params(lang, backend, algo):
    for key, value in consts.MODEL_MAPPING.items():
        if value['lang_code'] == lang:
            if value['backend']== backend and value['algo']== algo:
                return key, value
                
    
    ## Model might be loaded from local

    if algo and backend:
        return []
    else:
        raise Exception("The algo and backend for a model needs to be specified if model is loaded from local.")


def get_model_from_local(backend, algo, use_lm, args):
    if backend == 'torchscript' and algo == 'wav2vec2':
        if 'model_path' in args:
            return [args['model_path']]
        else:
            raise Exception("The model is not present in bol model zoo and you have not specified a local model to load as well.")

    if backend == 'fairseq' and algo == 'wav2vec2':
        model_path = args.get('model_path', None)
        dict_path = args.get('dict_path', None)
        
        if not model_path or not dict_path:
            raise Exception("The model and dict needs to be present for fairseq inference to work.")

        if use_lm:
            lm_path = args.get('lm_path', None)
            lexicon_path = args.get('lexicon_path', None)

            if not lm_path or not lexicon_path:
                raise Exception("For the language model to work both the binary file and lexicon file should be present.")

            return [model_path, dict_path] , [lm_path, lexicon_path]

        return [model_path, dict_path]
