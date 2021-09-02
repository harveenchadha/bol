from bol.models import load_model , Wav2VecCtc


model = load_model('hi-IN', use_lm=False)

#model = load_model_test(lang='hi-IN', backend='fairseq', algo='wav2vec2')

#model = load_model_test(lang='hi-IN', backend='fairseq', algo='wav2vec2', use_lm=False)


#model = load_model_test(unique_code='hi-IN')

#model = load_model_test(algo='wav2vec2', backend='torchscript')

# model = load_model_test(algo='wav2vec2', backend='torchscript', model_path = '/home/harveen/bol/test_model/wav2vec2.pt') # local model load


# model = load_model_test('hi-quant')

# from bol.utils import convert_audio_to_wav16

# filename = convert_audio_to_wav16('/home/harveen/bol/dev/long/PM.wav')
# preds = model.predict([filename])

# print(preds)

# model = load_model_test('hi-IN')
# #prediction = model.predict(['/home/harveen/bol/dev/eval/ahd_28_long_1335_hin-002500-005500-1-1.wav'], with_lm=True, apply_vad=True)
prediction = model.predict_from_dir(dir_path='/home/harveen/bol/dev/eval', ext='wav')

print(prediction)


# from bol.utils import convert_audio_to_wav16

# filename = convert_audio_to_wav16('/home/harveen/bol/dev/long/PM.wav')
# preds = model.predict([filename])


# eval = model.evaluate(['/home/harveen/bol/dev/eval/ahd_28_long_1335_hin-002500-005500-1-1.wav'], ['/home/harveen/bol/dev/eval/ahd_28_long_1335_hin-002500-005500-1-1.txt'])

#eval = model.evaluate_from_dir(dir_path='/home/harveen/bol/dev/eval', ext='wav', text_dir_path='/home/harveen/bol/dev/eval')
#print(eval)