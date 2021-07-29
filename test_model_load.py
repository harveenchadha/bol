
from bol.models import load_model, Wav2VecCtc
import glob

if __name__ == "__main__":
    #model = load_model('../files')



    model = load_model('en-IN')
    
    ## test single file
    # text = model.predict('../files/2_chunk-160.wav')
    # wer, cer = model.evaluate('../files/2_chunk-160.wav', '../files/2_chunk-160.txt')
    # print("WER: ", wer)
    # print("CER: ", cer)
    # with open('file_test.txt', mode='w+', encoding='utf-8') as file:
    #     file.writelines(text)



    ## test batch
    text = model.predict('../dev',  return_filenames=True)
    local_text = []
    for item in text:
        for local_item in item:
            local_text.append(local_item)

    # print(local_text)

    # for j in range(len(local_text)):
    #     if j%3==0:
    #         pred = item[j+1]
    #         time = item[j+2]
    #         print(pred)
    #         print(time)
    #         print(item)


    # for i in range(1,len(time)+1):
    #     print(i)
    #     print(pred)
    #     print(time)
    #     if i+2 == len(time):
    #         break
    #     print(time[i], time[i+1], pred[i-1])

    with open('file_test.txt', mode='w+', encoding='utf-8') as file:
        file.writelines(str(local_text))

    # ground_truth_files = glob.glob('../dev/*.txt')[0:10]
    # ground_truth = []
    # for file in ground_truth_files:
    #     with open(file, encoding='utf-8' ) as local_file:
    #         ground_truth.append(local_file.read())

    # with open('file_gt.txt', mode='w+', encoding='utf-8') as file:
    #     file.writelines("\n".join(ground_truth))
        
    # met = model.evaluate(ground_truth=ground_truth, predictions = text)
    # print(met)

    # wer, cer = model.predict_evaluate('../dev')
    # print(wer, cer)
    # local_text = []
    # for item in text:
    #     for local_item in item:
    #         local_text.append(local_item)

    # with open('file_test.txt', mode='w+', encoding='utf-8') as file:
    #     file.writelines("\n".join(text))


    # wer, cer = model.evaluate('../dev', '../dev')
    # print("WER: ", wer)
    # print("CER: ", cer)


    # model.summary()
    # model.predict()

    # file_path
    # use_cuda
    # half