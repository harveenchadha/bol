# bol
Speech Recognition Library for Indic Languages

## Installation (Currently in Alpha)

```
pip install --upgrade bol-library==0.141
```

## Do Speech Recognition in 3 lines of code

Usage:

```
from bol.models import load_model

model = load_model('hi-ts')

text = model.predict(['/home/harveen/bol/dev/long/virat.wav'])   

```

Output:
```
मेरा नाम विराट कोहली है 
```



## Current Models that are available

| Unique Code | Language Code | Backend | Algo | Language | provider | LM Support | CPU | GPU |
|------------|------------|--------|-------|-------------|----------|-------------|----|----|
| hi-ts | hi-IN | torchscript | wav2vec2 | Hindi | vakyansh_ekstep |  ✖️ | ✔️ | ✖️ |
| bn-ts | bn-IN | torchscript | wav2vec2 | Bengali | vakyansh_ekstep | ✖️ | ✔️ | ✖️ |
| gn-ts | gn-IN | torchscript | wav2vec2 | Gujarati | vakyansh_ekstep |✖️ | ✔️ | ✖️ |
| en-in-ts | en-IN | torchscript | wav2vec2 | Indian English | vakyansh_ekstep |✖️ | ✔️ | ✖️ |
| kn-ts | kn-IN | torchscript | wav2vec2 | Kannada | vakyansh_ekstep |✖️ | ✔️ | ✖️ |
| ne-ts | ne-IN | torchscript | wav2vec2 | Nepali | vakyansh_ekstep |✖️ | ✔️ | ✖️ |
| ta-ts | ta-IN | torchscript | wav2vec2 | Tamil | vakyansh_ekstep |✖️ | ✔️ | ✖️ |
| te-ts | te-IN | torchscript | wav2vec2 | Telugu | vakyansh_ekstep |✖️ | ✔️ | ✖️ |
| hi-vakyansh | hi-IN | fairseq | wav2vec2 | Hindi | vakyansh_ekstep | ✔️ | ✔️ | ✔️ |
| bn-vakyansh | bn-IN | fairseq | wav2vec2 | Bengali | vakyansh_ekstep |✔️ | ✔️ | ✔️ |
| en-vakyansh | en-IN | fairseq | wav2vec2 | English | vakyansh_ekstep |✔️ | ✔️ | ✔️ |
| gu-vakyansh | gu-IN | fairseq | wav2vec2 | Gujarati | vakyansh_ekstep |✔️ | ✔️ | ✔️ |
| kn-vakyansh | kn-IN | fairseq | wav2vec2 | Kannada | vakyansh_ekstep |✔️ | ✔️ | ✔️ |
| ne-vakyansh | ne-IN | fairseq | wav2vec2 | Nepali | vakyansh_ekstep |✔️ | ✔️ | ✔️ |
| ta-vakyansh | ta-IN | fairseq | wav2vec2 | Tamil | vakyansh_ekstep |✔️ | ✔️ | ✔️ |
| te-vakyansh | te-IN | fairseq | wav2vec2 | Telugu | vakyansh_ekstep |✔️ | ✔️ | ✔️ |




## Project Vision

1. Have state of the art speech recognition models for all the indic languages with or without language model

2. Provide High Performance Inference Pipelines for various backbones.

3. Provide Inference Pipeline for Speech Translation, Speech to Text Translation, Multilingual ASR
