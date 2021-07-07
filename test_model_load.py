
from bol.models import *

if __name__ == "__main__":
    model = load_model('../files')
    text = model.predict('../files/2_chunk-160.wav', viterbi=True)
    with open('file_test.txt', mode='w+', encoding='utf-8') as file:
        file.write(text)

    # model.summary()
    # model.predict()

    # file_path
    # use_cuda
    # half