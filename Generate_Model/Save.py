from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Conv1D, GRU
from tensorflow.keras.models import Model


class Save():
    def __init__(self,model,path='./Models/model_cov1d.h5'):
        self.model = model
        self.path = path
        self.save()

    def save(self):
        self.model.save(self.path)

