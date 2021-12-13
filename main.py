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
    
    #X_ham = np.empty([1, ham_counts]).flatten()
    #X_spam = np.empty([1, spam_counts]).flatten()


    X_ham = []
    X_spam = []


    for i in range(len(X_train)):
        if y_train[i] == 1:
            X_spam.append(X_train[i])
        else:
            X_ham.append(X_train[i])

    #print(f"X_ham: {X_ham}")
    #print(f"X_spam: {X_spam}")

    X_ham = remove_punctuations(X_ham)
    X_spam = remove_punctuations(X_spam)

    vectorizer = CountVectorizer(ngram_range=(1,1), min_df=0.001, stop_words=ENGLISH_STOP_WORDS)
    print(X_ham[0])
    

def remove_punctuations(data):
    for i in range(len(data)):
        data[i] = re.sub(r'[^\w\s]','', data[i])
        data[i] = re.sub("\s\s+", " ", data[i])
    return data


if __name__ == "__main__":
    main()
