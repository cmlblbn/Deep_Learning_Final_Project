from Preproses import Preproses
from Embedding import Embedding
from Makemodel import Makemodel
from Train import Train
from Save import Save
import pandas as pd


def main():
    train = pd.read_csv('./input/Corona_NLP_train.csv', encoding='latin1')
    test = pd.read_csv('./input/Corona_NLP_test.csv', encoding='latin1')
    df = train.append(test, ignore_index=True)


    preproses = Preproses(df)   #preproses işlemleri yapılıyor
    df = preproses.getDf() #dataframein son halini classdan geri aldık

    glove_embedding = Embedding(df)

    word_to_index = glove_embedding.get_word_to_index()
    word_to_vec_map = glove_embedding.get_word_to_vec_map() #Embedding katmanı için gerekli word2vec ve pozisyon bilgisi
    maxlen = glove_embedding.getmaxlen()

    X_train_indices = glove_embedding.get_X_train_indices() #tokenlara ayırıyoruz
    X_test_indices = glove_embedding.get_X_test_indices() #tokenlara ayırıyoruz

    Y_oh_train = glove_embedding.Y_oh_train() #labellama işlemi
    Y_oh_test = glove_embedding.Y_oh_test() #labellama işlemi

    modelsMaker = Makemodel(maxlen,word_to_vec_map,word_to_index) #lstm ve conv1d modelleri bu class üzerinde oluşturuyoruz

    model_lstm = modelsMaker.getmodel_lstm()
    model_cov1d = modelsMaker.getmodel_cov1d()

    train = Train(model_lstm,model_cov1d,X_train_indices,X_test_indices,Y_oh_train,Y_oh_test) #modellerimiz burada eğitiliyor.

    model_lstm = train.getmodel_lstm()
    model_cov1d = train.getmodel_cov1d() #eğitilen modelleri geri alıyoruz

    save_lstm = Save(model_lstm)
    save_cov1d = Save(model_cov1d) #eğitilen modelleri models klasörüne kaydediyoruz

    #artık gui üzerinden çalışabiliriz.

if __name__ == '__main__':
    main()





