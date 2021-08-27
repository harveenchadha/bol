from bol.models import load_model_ts, Wav2Vec2TS

model = load_model_ts('/home/harveen/bol/quantized_model/wav2vec2.pt')

prediction = model.predict(['/home/harveen/bol/dev/eval/ahd_28_long_1335_hin-002500-005500-1-1.wav'])
print(prediction)