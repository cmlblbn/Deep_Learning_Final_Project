import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from keras.utils import np_utils




class Embedding:
    def __init__(self,df,maxlen=50):

        raw_docs_train = df["OriginalTweet"].values
        sentiment_train = df['sentiment'].values
        #verileri x ve y olarak ayırdık

        self.maxlen = maxlen

        X_train, X_test, Y_train, Y_test = train_test_split(raw_docs_train, sentiment_train,
                                                            stratify=sentiment_train,
                                                            random_state=42,
                                                            test_size=0.20, shuffle=True)

        #split ayırma işlemleriyle eğitim ve validasyon verisi hazırlandı

        num_labels = len(np.unique(sentiment_train))
        self.Y_oh_train = np_utils.to_categorical(Y_train, num_labels)
        self.Y_oh_test = np_utils.to_categorical(Y_test, num_labels)

        self.word_to_index, self.word_to_vec_map = self.read_glove_vecs('./glove_vector/glove.6B.50d.txt')
        self.embedding_layer = self.pretrained_embedding_layer(self.word_to_vec_map,self.word_to_index)
        #pretrained glove vektörleri embedding olarak kullanıyoruz

        self.X_train_indices = self.sentences_to_indices(X_train, self.word_to_index, maxlen)
        self.X_test_indices = self.sentences_to_indices(X_test, self.word_to_index, maxlen)



    def getmaxlen(self):
        return self.maxlen

    def get_word_to_index(self):
        return self.word_to_index

    def get_word_to_vec_map(self):
        return self.word_to_vec_map


    def get_embedding_layer(self):
        return self.embedding_layer


    def get_X_train_indices(self):
        return self.X_train_indices
    def get_X_test_indices(self):
        return self.X_test_indices

    def get_Y_oh_train(self):
        return self.Y_oh_train
    def get_Y_oh_test(self):
        return self.Y_oh_test

    def read_glove_vecs(self,glove_file):
        with open(glove_file, encoding="utf8") as f:
            words = set()
            word_to_vec_map = {}
            for line in f:
                line = line.strip().split()
                curr_word = line[0]
                words.add(curr_word)
                word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

            i = 1
            words_to_index = {}
            for w in sorted(words):
                words_to_index[w] = i
                i = i + 1
        return words_to_index, word_to_vec_map

    def sentences_to_indices(self,X, word_to_index, max_len):

        m = X.shape[0]
        X_indices = np.zeros((m, max_len))

        for i in range(m):

            sentence_words = [word.lower().replace('\t', '') for word in X[i].split(' ') if
                              word.replace('\t', '') != '']
            j = 0
            for w in sentence_words:
                try:
                    X_indices[i, j] = word_to_index[w]
                except:
                    0
                j = j + 1

        return X_indices

    def pretrained_embedding_layer(self,word_to_vec_map, word_to_index):

        vocab_len = len(word_to_index) + 1
        emb_dim = word_to_vec_map["cucumber"].shape[0]

        emb_matrix = np.zeros((vocab_len, emb_dim))

        for word, index in word_to_index.items():
            emb_matrix[index, :] = word_to_vec_map[word]

        embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)

        embedding_layer.build((None,))
        embedding_layer.set_weights([emb_matrix])

        return embedding_layer


def pretrained_embedding_layer(word_to_vec_map, word_to_index):

    vocab_len = len(word_to_index) + 1
    emb_dim = word_to_vec_map["cucumber"].shape[0]

    emb_matrix = np.zeros((vocab_len, emb_dim))

    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)

    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer