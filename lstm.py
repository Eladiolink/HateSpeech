import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from Grids_Search_Models.LSTM import lstm
from Utils.BackTranslation import dataframeWithBackTranslation
from Utils.Ros import ROS,ROS2
from Utils.Rus import RUS

vectorizer = TfidfVectorizer()

def execute_preProcessed():
    train = pd.read_csv("./Dataset/PreProcessed/train.csv")
    test = pd.read_csv("./Dataset/PreProcessed/test.csv")

    # simple(train,test,"preProcessed")
    rus(train,test,"preProcessed")
    # backTranslation(train,test,"preProcessed")

def execute_Lematization():
    train = pd.read_csv("./Dataset/Lematization/train.csv")
    test = pd.read_csv("./Dataset/Lematization/test.csv")

    simple(train, test, "lematization")
    ros(train, test, "lematization")
    backTranslation(train, test, "lematization")

def execute_nonStopWords():
    train = pd.read_csv("./Dataset/NonStopWords/train.csv")
    test = pd.read_csv("./Dataset/NonStopWords/test.csv")

    simple(train, test, "nonStopWords")
    ros(train, test, "nonStopWords")
    backTranslation(train, test, "nonStopWords")

def execute_Stemming():
    train = pd.read_csv("./Dataset/Stemming/train.csv")
    test = pd.read_csv("./Dataset/Stemming/test.csv")

    simple(train, test, "stemming")
    ros(train, test, "stemming")
    backTranslation(train, test, "stemming")

def ros(train,test,ty):
    type = "ros_"+ty
    X_train, y_train, X_test, y_test = ROS2(train, test)

    lstm(X_train,y_train,X_test,y_test,type)

def rus(train,test,ty):
    type = "rus_"+ty
    X_train, y_train, X_test, y_test = RUS(train, test)

    lstm(X_train,y_train,X_test,y_test,type)
    

def backTranslation(train,test,ty):
    type = "backTranslation_"+ty
    X_train, y_train, X_test, y_test = dataframeWithBackTranslation(train, test)

    X_train = X_train.fillna('')
    X_test = X_test.fillna('')

    lstm(X_train,y_train,X_test,y_test,type)
    

def simple(train,test,ty):
    type = "simple_" + ty
    X_train = train['text'].fillna('')
    X_test = test['text'].fillna('')

    y_train = train['label'].fillna('')
    y_test = test['label'].fillna('')

    lstm(X_train, y_train, X_test, y_test, type)


def preprocess_text(text):
    if isinstance(text, str):
        return text.lower()
    else:
        return str(text).lower() if text is not None else ''

import multiprocessing


execute_preProcessed()
execute_Stemming()
execute_Lematization()
execute_nonStopWords()

print("FIM DA EXECUÇÃO :)")

