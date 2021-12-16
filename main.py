import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def main():
    df = pd.read_csv("emails.csv")

    X = df["text"].str.lower().values
    y = df["spam"].values

    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=0)

    ham_counts = sum(y_train)
    spam_counts = len(y_train) - ham_counts

    X_ham = []
    X_spam = []

    for i in range(len(X_train)):
        if y_train[i] == 1:
            X_spam.append(X_train[i])
        else:
            X_ham.append(X_train[i])

    X_ham = remove_punctuations(X_ham)
    X_spam = remove_punctuations(X_spam)

    vectorizer_for_ham = CountVectorizer(ngram_range=(1,1), min_df=0.001, stop_words=ENGLISH_STOP_WORDS)
    vectorizer_for_spam = CountVectorizer(ngram_range=(1,1), min_df=0.001, stop_words=ENGLISH_STOP_WORDS)

    ham_words = vectorizer_for_ham.fit_transform(X_ham)
    spam_words = vectorizer_for_spam.fit_transform(X_spam)

    # arrays contains frequency of each word for each sentence (each column is word, each row is sentence)
    # it is transposed because reaching to row is easier than reaching to column
    ham_array = ham_words.transpose().toarray()
    spam_array = spam_words.transpose().toarray()

    # these dictionaries contains words and their frequencies seperately (ham and spam). 
    ham_dict = create_dict(vectorizer_for_ham.get_feature_names(), ham_array)
    spam_dict = create_dict(vectorizer_for_spam.get_feature_names(), spam_array)


def create_dict(features, freq_array):
    dict = {}
    for i in range(len(features)):
        word = features[i]
        freq = sum(freq_array[i])
        dict[word] = freq
    return dict


def remove_punctuations(data):
    for i in range(len(data)):
        data[i] = re.sub(r'[^\w\s]','', data[i])
        data[i] = re.sub("\s\s+", " ", data[i])
    return data


if __name__ == "__main__":
    main()
