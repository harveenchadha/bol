

from bol.models import load_model_ts

model = load_model_ts('/home/harveen/bol/quantized_model/wav2vec2.pt')

#prediction = model.predict(['/home/harveen/bol/dev/eval/ahd_28_long_1335_hin-002500-005500-1-1.wav'])

# prediction = model.predict(test_files)


prediction = model.predict_from_dir(dir_path='/home/harveen/bol/dev/eval', ext='wav')

print(prediction)