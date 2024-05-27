import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from Grids_Search_Models.SVC import grid_SVC
from Grids_Search_Models.Naive_Bayes import grid_naive_bayes
from Grids_Search_Models.Logistic_Regression import grid_logistic_regression
from Grids_Search_Models.XgBoost import grid_xgBoost
from Grids_Search_Models.AdaBoost import grid_adaBoost
import os
from Utils.PreProcessing import preProcessing
from Utils.BackTranslation import dataframeWithBackTranslation
from Utils.Ros import ROS
from Utils.Rus import RUS
from Utils.PreProcessing import Stemming
from Utils.PreProcessing import use_stopwords
from Utils.PreProcessing import useLematization

vectorizer = TfidfVectorizer()

def execute_preProcessed():
    train = pd.read_csv("./Dataset/PreProcessed/train.csv")
    test = pd.read_csv("./Dataset/PreProcessed/test.csv")

    simple(train,test,"preProcessed")
    ros(train,test,"preProcessed")
    rus(train,test,"preProcessed")
    backTranslation(train,test,"preProcessed")

def execute_Lematization():
    train = pd.read_csv("./Dataset/Lematization/train.csv")
    test = pd.read_csv("./Dataset/Lematization/test.csv")

    simple(train, test, "lematization")
    ros(train, test, "lematization")
    rus(train, test, "lematization")
    backTranslation(train, test, "lematization")

def execute_nonStopWords():
    train = pd.read_csv("./Dataset/NonStopWords/train.csv")
    test = pd.read_csv("./Dataset/NonStopWords/test.csv")

    simple(train, test, "nonStopWords")
    ros(train, test, "nonStopWords")
    rus(train, test, "nonStopWords")
    backTranslation(train, test, "nonStopWords")

def execute_Stemming():
    train = pd.read_csv("./Dataset/Stemming/train.csv")
    test = pd.read_csv("./Dataset/Stemming/test.csv")

    simple(train, test, "stemming")
    ros(train, test, "stemming")
    rus(train, test, "stemming")
    backTranslation(train, test, "stemming")

def ros(train,test,ty):
    type = "ros_"+ty
    X_train, y_train, X_test, y_test = ROS(train, test)
    grid_SVC(X_train,y_train,X_test,y_test,type)
    grid_naive_bayes(X_train,y_train,X_test,y_test,type)
    
    grid_xgBoost(X_train,y_train,X_test,y_test,type)(X_train,y_train,X_test,y_test,type)
    grid_logistic_regression(X_train, y_train, X_test, y_test, type)

def rus(train,test,ty):
    type = "rus_"+ty
    X_train, y_train, X_test, y_test = ROS(train, test)
    grid_SVC(X_train,y_train,X_test,y_test,type)
    grid_naive_bayes(X_train,y_train,X_test,y_test,type)
    
    grid_xgBoost(X_train,y_train,X_test,y_test,type)(X_train,y_train,X_test,y_test,type)
    grid_logistic_regression(X_train, y_train, X_test, y_test, type)

def backTranslation(train,test,ty):
    type = "backTranslation_"+ty
    X_train, y_train, X_test, y_test = dataframeWithBackTranslation(train, test)

    X_train = vectorizer.fit_transform(X_train.fillna(''))
    X_test = vectorizer.transform(X_test.fillna(''))

    grid_SVC(X_train,y_train,X_test,y_test,type)
    grid_naive_bayes(X_train,y_train,X_test,y_test,type)
    
    grid_xgBoost(X_train,y_train,X_test,y_test,type)(X_train,y_train,X_test,y_test,type)
    grid_logistic_regression(X_train, y_train, X_test, y_test, type)

def simple(train,test,ty):
    type = "simple_" + ty
    X_train = vectorizer.fit_transform(train['text'].fillna(''))
    X_test = vectorizer.transform(test['text'].fillna(''))

    y_train = train['label']
    y_test = test['label']

    grid_SVC(X_train, y_train, X_test, y_test, type)
    grid_naive_bayes(X_train, y_train, X_test, y_test, type)
    
    grid_xgBoost(X_train,y_train,X_test,y_test,type)(X_train, y_train, X_test, y_test, type)
    grid_logistic_regression(X_train, y_train, X_test, y_test, type)

execute_preProcessed()
execute_Stemming()
execute_Lematization()
execute_nonStopWords()

print("FIM DA EXECUÇÃO :)")
