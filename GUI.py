from tkinter import *
import tensorflow as tf
import numpy as np

class Editor(Frame):

    def __init__(self,parent):
        Frame.__init__(self, parent)
        self.root=parent
        self.initUI()

    def initUI(self):
        self.model_cov1d = tf.keras.models.load_model('./Generate_Model/models/model_cov1d.h5') #macOs için path kontrolü yapılmalı!
        self.model_lstm = tf.keras.models.load_model('./Generate_Model/models/model_lstm.h5') #macOs için path kontrolü yapılmalı!
        self.word_to_index, self.word_to_vec_map = self.read_glove_vecs('./glove_vector/glove.6B.50d.txt')
        self.classes = {0: 'Olumsuz', 1: 'Tarafsız', 2: 'Olumlu'}
        self.grid()
        frame = Frame(self, bg="Beige", width="500", height="250", pady="25", padx="10")
        frame.grid()
        self.tivit_adi = Label(frame, text="Tivit giriniz(ingilizce): ", fg="Red", bg="Beige")
        self.tivit_adi.config(font=("Courier", 10, "bold italic"))
        self.tivit_adi.grid(row = 0,column = 0,columnspan = 2)

        self.tivit = StringVar()
        self.tivitGir = Entry(frame, textvariable=self.tivit, fg="Red", bg="Beige", width=80)
        self.tivitGir.grid(row = 1, column = 0,columnspan = 2)


        self.bosLabel = Label(frame,fg="Red", bg="Beige")
        self.bosLabel.grid(row = 2,column = 0)



        self.items = Label(frame, text="Duygu analizi için gerekli algoritmayı seç:", fg="Red", bg="Beige")
        self.items.grid(row = 3,column = 0, columnspan = 1,sticky = W)
        self.bosLabel2 = Label(frame, fg="Blue", bg="Beige")
        self.bosLabel2.grid(row=4, column=0)
        self.radio = IntVar()
        self.radio1 = Radiobutton(frame, text="LSTM    ", value=0, variable=self.radio, fg="Red", bg="Beige")#0
        self.radio2 = Radiobutton(frame, text="Conv1d", value=1, variable=self.radio, fg="Red", bg="Beige")#1
        self.radio1.grid(row=5, column=0, sticky=N + S + E + W, padx="30")
        self.radio2.grid(row=6, column=0, sticky=S, padx="30")

        self.data = StringVar()
        self.dataGir = Entry(frame, textvariable=self.data, fg="Red", bg="Beige", width=80)
        self.dataGir.grid(row = 1, column = 0,columnspan = 2)


        self.bosLabel = Label(frame,fg="Red", bg="Beige")
        self.bosLabel.grid(row = 2,column = 0)


     
       
        self.button = Button(frame, text="ANALIZ ET", fg="Red", bg="Gold", command=self.getPrediction, width="10")
        self.button.grid(row=5, column=1, sticky=E + S)

        self.sonuc = Label(frame, text="Duygu durumunuz:", fg="Red", bg="Beige")
        self.sonuc.grid(row=8, column=0, columnspan=1, sticky=W)

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

    def getPrediction(self):
        self.M_data= self.data.get()
        decision = self.radio.get()
        if self.M_data:
            if decision == 0:
                maxlen = 50
                array_data = np.array([self.M_data])
                word2vec = self.sentences_to_indices(array_data, self.word_to_index, maxlen)
                self.prediction = str(self.classes[np.argmax(self.model_lstm.predict(word2vec))])
                print(array_data[0] + ' --- prediction is: ' + self.prediction)
                self.sonuc["text"] = "Duygu durumunuz : " + self.prediction
            else:
                maxlen = 50
                array_data = np.array([self.M_data])
                word2vec = self.sentences_to_indices(array_data, self.word_to_index, maxlen)
                self.prediction = str(self.classes[np.argmax(self.model_cov1d.predict(word2vec))])
                print(array_data[0] + ' --- prediction is: ' + self.prediction)
                self.sonuc["text"] = "Duygu durumunuz : " + self.prediction

        else:
            self.sonuc["text"]= "Lütfen yorum kısmını doldurunuz!"







def main():
    root= Tk()
    root.title("NLP_Prototype")
    root.geometry("500x250")
    root.resizable(FALSE,FALSE)
    App = Editor(root)
    root.mainloop()



if __name__ == '__main__':
    main()