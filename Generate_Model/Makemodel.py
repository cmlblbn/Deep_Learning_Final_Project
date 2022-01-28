from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Conv1D, GRU
from tensorflow.keras.models import Model
from Embedding import pretrained_embedding_layer

class Makemodel():
    def __init__(self,input_shape,word_to_vec_map,word_to_index,unit1_count = 50,unit2_count = 100, unit3_count = 50):
        self.model_lstm
        self.model_cov1d

        self.input_shape = input_shape
        self.word_to_vec_map = word_to_vec_map
        self.word_to_index = word_to_index
        self.unit1_count = unit1_count
        self.unit2_count = unit2_count
        self.unit3_count = unit3_count

        self.model_lstm = self.ltsm_model((self.maxlen,), word_to_vec_map, word_to_index)
        self.show_summary(self.model_lstm)
        self.compile_model(self.model_lstm)
        #model oluştur, özetine bak ve compile et


        self.model_cov1d = self.cov1d_model((self.maxlen,), word_to_vec_map, word_to_index)
        self.show_summary(self.model_cov1d)
        self.compile_model(self.model_cov1d)
        #model oluştur, özetine bak ve compile et


    def getmodel_cov1d(self):
        return self.model_cov1d
    def getmodel_lstm(self):
        return self.model_lstm


    def make_cov1d_model(self):

        sentence_indices = Input(shape=self.input_shape, dtype='int32')
        embedding_layer = pretrained_embedding_layer(self.word_to_vec_map, self.word_to_index)
        embeddings = embedding_layer(sentence_indices)

        X = Conv1D(self.unit1_count, 1, activation='relu')(embeddings)
        X = Dropout(0.2)(X)
        X = Conv1D(self.unit2_count, 1, activation='relu')(X)
        X = GRU(self.unit3_count, dropout=0.2, recurrent_dropout=0.5)(X)
        X = Dropout(0.2)(X)
        X = Dense(3, activation='softmax')(X)

        model_cov1d = Model(inputs=[sentence_indices], outputs=X)

        return model_cov1d


    def make_lstm_model(self):
        sentence_indices = Input(shape=self.input_shape, dtype='int32')
        embedding_layer = pretrained_embedding_layer(self.word_to_vec_map, self.word_to_index)
        embeddings = embedding_layer(sentence_indices)

        X = LSTM(self.unit1_count, return_sequences=True)(embeddings)
        X = Dropout(0.2)(X)
        X = LSTM(self.unit2_count, return_sequences=True, dropout=0.2, recurrent_dropout=0.5)(X)
        X = LSTM(self.unit3_count, return_sequences=False)(X)
        X = Dropout(0.2)(X)
        X = Dense(3, activation='softmax')(X)

        model_lstm = Model(inputs=[sentence_indices], outputs=X)

        return model_lstm

    def show_summary(self,model):
        model.summary()

    def compile_model(self,model):
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

