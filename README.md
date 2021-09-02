# bol
Open Source Speech To Text Engines

## Do Speech Recognition in 3 lines of code

Usage:

```
from bol.models import load_model

model = load_model('hi-quant')

text = model.predict(['/home/harveen/bol/dev/long/virat.wav'])   

```

Output:
```
मेरा नाम विराट कोहली है 
```
