

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



def _get_file_name_from_url(url):
    return url.split('/')[-1]

def _get_model_path_from_model_code(model_code):
    home = expanduser("~")
    base_path = home + consts.MODEL_PATH
    full_path = base_path + '/' + model_code
    return full_path

def _download(full_path, urls, download_type):
    os.makedirs(full_path, exist_ok = True)
    for url in urls[download_type]:
        wget_cmd = 'wget '+ url + ' -P '+ full_path
        filename = _get_file_name_from_url(url)
        if filename[-3:] == 'zip':
            unzip_cmd = 'unzip ' + full_path+'/' + filename + ' -d ' +full_path
            remove_cmd = 'rm ' + full_path+'/' + filename
            os.system(wget_cmd)
            os.system(unzip_cmd)
            os.system(remove_cmd)
        if filename[-3:] == '.xz':
            unzip_cmd = 'tar -xvf ' + full_path+'/' + filename + ' -C ' +full_path + ' --strip-components=3'
            remove_cmd = 'rm ' + full_path+'/' + filename
            os.system(wget_cmd)
            os.system(unzip_cmd)
            os.system(remove_cmd)
 

def _get_names_of_model_files(path):
    model_path = glob.glob(path + '/*.pt')
    dict_path = glob.glob(path + '/*.ltr.txt')

    if not model_path or not dict_path:
        return None, None

    return model_path[0], dict_path[0]

def _check_if_required_files_exist(model_path):    
    model_path, dict_path = _get_names_of_model_files(model_path)

    if not model_path or not dict_path:
        return False
    else:
        return True


def _check_if_model_exists(model_code, urls):
    full_path = _get_model_path_from_model_code(model_code)
    if os.path.exists(full_path):
        print("Path already exists")
        check_files_exist = _check_if_required_files_exist(full_path)

        if not check_files_exist:
            print("Folder exists but files not present donwloading again")
            _download(full_path, urls, 'model_url')
        return full_path
    else:
        _download(full_path, urls, 'model_url')
        return full_path

def _delete_model_files(full_path):
    cmd = 'rm -rf '+ full_path
    os.system(cmd)
    #os.rmdir(full_path)

    
def verify_model_mapping(model_code, force_download):
    if model_code in consts.MODEL_MAPPING.keys():
        
        # check if model exists, if it does return the path, if it doesn't donwload

        if force_download:
            full_path = _get_model_path_from_model_code(model_code)
            _delete_model_files(full_path)

        full_path = _check_if_model_exists(model_code, consts.MODEL_MAPPING[model_code]['urls'])
        model_path, dict_path = _get_names_of_model_files(full_path)
        return model_path, dict_path
    else:
        print('False')




def _get_names_of_model_files_for_lm(path):
    lm_path = glob.glob(path + '/*.binary')
    lexicon_path = glob.glob(path + '/*.lst')

    if not lm_path or not lexicon_path:
        return None, None

    return lm_path[0], lexicon_path[0]

def _check_if_required_files_exist_for_lm(model_path):    
    lm_path, lexicon_path = _get_names_of_model_files_for_lm(model_path)
    if not lm_path or not lexicon_path:
        return False
    else:
        return True


def _check_if_lm_exists(model_code, urls):
    full_path = _get_model_path_from_model_code(model_code)
    if os.path.exists(full_path):
        # print("Path already exists")
        check_files_exist = _check_if_required_files_exist_for_lm(full_path)
        if not check_files_exist:
            print("LM Folder exists but files not present donwloading again")
            _download(full_path, urls, 'lm_url')
        return full_path
    else:
        _download(full_path, urls, 'lm_url')
        return full_path


def verify_lm_mapping(model_code):
    if model_code in consts.MODEL_MAPPING.keys():
        # print('True')
        # check if model exists, if it does return the path, if it doesn't donwload

        full_path = _check_if_lm_exists(model_code, consts.MODEL_MAPPING[model_code]['urls'])
        #get names of all the files.
        #print(full_path)
        lm_path, lexicon_path = _get_names_of_model_files_for_lm(full_path)
        # print('Here123')
        # print(lm_path, lexicon_path)
        return lm_path, lexicon_path
    else:
        print('False')

