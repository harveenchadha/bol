import glob

from joblib import Parallel, delayed
from tqdm import tqdm


def read_text_file(local_file):
    with open(local_file, encoding='utf-8', mode='r') as lfile:
        text = lfile.read()
    output = {'text_file_name': local_file, 'text_file_content': text.strip()}
    return output

def load_text_files_in_parallel(file_paths):
    text_files = [] 
    text_files.extend( Parallel(n_jobs=-1)(delayed(read_text_file)(local_file) for local_file in tqdm(file_paths)) )
    return text_files


def load_text_files_in_parallel_from_dir(dir_path):
    files = glob.glob(dir_path+'/*.txt', recursive=True)
    return load_text_files_in_parallel(files)
