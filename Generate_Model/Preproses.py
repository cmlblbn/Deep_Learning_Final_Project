import pandas as pd
import numpy as np
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
import matplotlib.pyplot as plt



class Preproses():
    def __init__(self,df):
        self.df = self.df_construct(df)
        self.df = self.preprocess(self.df)

    def getDf(self):
        return self.df
    def setDf(self,df):
        self.df = df
    
    def df_construct(self,df):
        df = df.drop(['UserName','ScreenName','Location','TweetAt'],axis = 1)
        print(df.Sentiment.value_counts())
        df = self.df_sentiment_encode(df)
        self.plot_mostCountWords(df)
        return df

    def df_sentiment_encode(self,df):
        df.loc[:,'sentiment'] = df.Sentiment.map({'Negative':0,'Neutral':1,'Positive':2,'Extremely Positive':2,'Extremely Negative':0})
        df = df.drop(['Sentiment'], axis=1)
        return df

    def plot_mostCountWords(self,df):
        positiveWordCount = nltk.FreqDist(
            word for text in df[df["sentiment"] == 2]["OriginalTweet"] for word in text.lower().split())
        negativeWordCount = nltk.FreqDist(
            word for text in df[df["sentiment"] == 0]["OriginalTweet"] for word in text.lower().split())
        neutralWordCount = nltk.FreqDist(
            word for text in df[df["sentiment"] == 1]["OriginalTweet"] for word in text.lower().split())

        plt.subplots(figsize=(8, 6))
        plt.title("pozitif tivitlerde en çok kullanılan 40 kelime")
        positiveWordCount.plot(40)
        plt.show()

        plt.subplots(figsize=(8, 6))
        plt.title("Negatif tivitlerde en çok kullanılan 40 kelime")
        negativeWordCount.plot(40)
        plt.show()

        plt.subplots(figsize=(8, 6))
        plt.title("Tarafsız tivitlerde en çok kullanılan 40 kelime")
        neutralWordCount.plot(40)
        plt.show()

    def preprocess(self,df):
        lemma = WordNetLemmatizer()
        swords = stopwords.words("english")
        for i in df.index:
            text = df['OriginalTweet'][i]

            # http taglerini temizle
            text = re.sub(r'http\S+', '', text)

            # Sadece kelimeleri ifade eden verileri içeride bırak
            text = re.sub("[^a-zA-Z0-9]", " ", text)

            # parçalama ve işleme
            text = nltk.word_tokenize(text.lower())
            text = [lemma.lemmatize(word) for word in text]

            # Stopwordsleri sil
            text = [word for word in text if word not in swords]

            # kelimelerden cümle haline getir
            text = " ".join(text)

            df['OriginalTweet'][i] = text

        return df

