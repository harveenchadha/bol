
from bol.models import load_model, Wav2VecCtc

if __name__ == "__main__":
    #model = load_model('../files')



    model = load_model('hi')
    
    ## test single file
    text = model.predict('../files/common_voice.mp3')
    with open('file_test.txt', mode='w+', encoding='utf-8') as file:
        file.writelines(text)

    # ## test batch
    # text = model.predict('../vak/hindi_test_dummy')
    # local_text = []
    # for item in text:
    #     for local_item in item:
    #         local_text.append(local_item)

    # with open('file_test.txt', mode='w+', encoding='utf-8') as file:
    #     file.writelines("\n".join(local_text))



    # model.summary()
    # model.predict()

    # file_path
    # use_cuda
    # half