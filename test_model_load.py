
from bol.models import *

if __name__ == "__main__":
    model = load_model('../files/hindi.pt')
    model.summary()
    model.predict()
