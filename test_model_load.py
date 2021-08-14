
from bol.models import load_model, Wav2VecCtc
import glob

if __name__ == "__main__":
    #model = load_model('../files')



    model = load_model('od-IN',use_lm=False)
    text = model.predict('/home/harveen/bol/subtask1_blindtest/raw/Odia/audio/', return_filenames=True)
    #text = model.predict('/home/harveen/bol/dev/eval', return_filenames=True)
    # print(text) 

    
    with open('output_odia_without_lm.txt', mode='w+', encoding='utf-8') as file:
        for item in text:
            print(item['file'] + " " + item['transcription'], file=file)

    gt_dict = {}
    with open('/home/harveen/bol/subtask1_blindtest/raw/Odia/transcription.txt', mode='r', encoding='utf-8') as file:   
        ground_truth = file.readlines()
        for line in ground_truth:
            file, trans = line.split(' ', 1)
            gt_dict[file] =  trans.strip()

    pr_dict = {}
    with open('output_odia_without_lm.txt', mode='r',  encoding='utf-8') as file:
        predictions = file.readlines()
        for line in predictions:
            arr = line.split(' ',1)           
            local_file = arr[0]
            local_file = local_file.split('/')[-1].split('.')[0]
            pr_dict[local_file] = arr[1].strip()



    gt = []
    pr = []
    for key, value in gt_dict.items():
        gt.append(value)
        pr.append(pr_dict[key])

    metrics = model.evaluate(gt, pr)
    print("WER: ", metrics['wer'])
    print("CER: ", metrics['cer'])

    ## test single file
    # text = model.predict('../files/2_chunk-160.wav')
    # wer, cer = model.evaluate('../files/2_chunk-160.wav', '../files/2_chunk-160.txt')
    # print("WER: ", wer)
    # print("CER: ", cer)
    # with open('file_test.txt', mode='w+', encoding='utf-8') as file:
    #     file.writelines(text)



    ## test batch
    # text = model.predict('../dev',  return_filenames=True)


    # local_text = []
    # for item in text:
    #     for local_item in item:
    #         local_text.append(local_item)

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

    # with open('file_test.txt', mode='w+', encoding='utf-8') as file:
    #     file.writelines("\n".join(text))

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