import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Conv1D, GRU
from tensorflow.keras.models import Model

class Train():
    def __init__(self,lstm_model,cov1d_model,X_train_indices, X_test_indices, Y_oh_train, Y_oh_test,epoch = 20, batch_size = 64):
        self.history_lstm
        self.history_cov1d

        self.lstm_model = lstm_model
        self.cov1d_model = cov1d_model

        self.X_train_indices = X_train_indices
        self.X_test_indices = X_test_indices
        self.Y_oh_train = Y_oh_train
        self.Y_oh_test = Y_oh_test
        self.epoch = epoch
        self.batch_size = batch_size


        self.train(self.lstm_model)
        #lstm modeli eğit

        self.train(self.cov1d_model)
        #cov1d modeli eğit

        self.evaluate_model(self.lstm_model)
        #lstm sonuçları göster

        self.evaluate_model(self.cov1d_model)
        #cov1d sonuçları göster

        self.plot_accuracy_and_loss(self.history_lstm)
        #lstm grafikleri göster

        self.plot_accuracy_and_loss(self.history_cov1d)
        #cov1d grafikleri göster

    def getmodel_cov1d(self):
        return self.model_cov1d
    def getmodel_lstm(self):
        return self.model_lstm

    def train(self,model):
        self.history_lstm = model.fit(x=self.X_train_indices, y=self.Y_oh_train, batch_size=self.batch_size,
                                      epochs=self.epoch, verbose=1, validation_data=(self.X_test_indices, self.Y_oh_test))

    def evaluate_model(self,model):
        model.evaluate(x=self.X_test_indices,y=self.Y_oh_test)

    def plot_accuracy_and_loss(self,history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title('Training and Test Accuraty')
        plt.legend()

        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and Test Loss')
        plt.legend()

        plt.show()

