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

def execute_preProcessed(task):
    train = pd.read_csv("./Dataset/PreProcessed/train.csv")
    test = pd.read_csv("./Dataset/PreProcessed/test.csv")

    p1 = multiprocessing.Process(target=simple,
                            args=(
                                pd.read_csv("./Dataset/PreProcessed/train.csv"),
                                pd.read_csv("./Dataset/PreProcessed/test.csv"),
                                "preProcessed"))
    p2 = multiprocessing.Process(target=ros,
                            args=(
                                pd.read_csv("./Dataset/PreProcessed/train.csv"),
                                pd.read_csv("./Dataset/PreProcessed/test.csv"),
                                "preProcessed"))
    
    p3 = multiprocessing.Process(target=rus,
                            args=(
                                pd.read_csv("./Dataset/PreProcessed/train.csv"),
                                pd.read_csv("./Dataset/PreProcessed/test.csv"),
                                "preProcessed"))
    
    p4 = multiprocessing.Process(target=backTranslation,
                            args=(
                                pd.read_csv("./Dataset/PreProcessed/train.csv"),
                                pd.read_csv("./Dataset/PreProcessed/test.csv"),
                                "preProcessed"))
    task.append(p1)
    task.append(p2)
    task.append(p3)
    task.append(p4)
    return task

def execute_Lematization(task):
    train = pd.read_csv("./Dataset/Lematization/train.csv")
    test = pd.read_csv("./Dataset/Lematization/test.csv")

    p1 = multiprocessing.Process(target=simple,
                            args=(
                                pd.read_csv("./Dataset/Lematization/train.csv"),
                                pd.read_csv("./Dataset/Lematization/test.csv"),
                                "lematization"))
    p2 = multiprocessing.Process(target=ros,
                            args=(
                                pd.read_csv("./Dataset/Lematization/train.csv"),
                                pd.read_csv("./Dataset/Lematization/test.csv"),
                                "lematization"))
    
    p3 = multiprocessing.Process(target=rus,
                            args=(
                                pd.read_csv("./Dataset/Lematization/train.csv"),
                                pd.read_csv("./Dataset/Lematization/test.csv"),
                                "lematization"))
    
    p4 = multiprocessing.Process(target=backTranslation,
                            args=(
                                pd.read_csv("./Dataset/Lematization/train.csv"),
                                pd.read_csv("./Dataset/Lematization/test.csv"),
                                "lematization"))
    task.append(p1)
    task.append(p2)
    task.append(p3)
    task.append(p4)
    return task
def execute_nonStopWords(task):
    train = pd.read_csv("./Dataset/NonStopWords/train.csv")
    test = pd.read_csv("./Dataset/NonStopWords/test.csv")

    p1 = multiprocessing.Process(target=simple,
                            args=(
                                pd.read_csv("./Dataset/NonStopWords/train.csv"),
                                pd.read_csv("./Dataset/NonStopWords/test.csv"),
                                "nonStopWords"))
    p2 = multiprocessing.Process(target=ros,
                            args=(
                                pd.read_csv("./Dataset/NonStopWords/train.csv"),
                                pd.read_csv("./Dataset/NonStopWords/test.csv"),
                                "nonStopWords"))
    
    p3 = multiprocessing.Process(target=rus,
                            args=(
                                pd.read_csv("./Dataset/NonStopWords/train.csv"),
                                pd.read_csv("./Dataset/NonStopWords/test.csv"),
                                "nonStopWords"))
    
    p4 = multiprocessing.Process(target=backTranslation,
                            args=(
                                pd.read_csv("./Dataset/NonStopWords/train.csv"),
                                pd.read_csv("./Dataset/NonStopWords/test.csv"),
                                "nonStopWords"))
    task.append(p1)
    task.append(p2)
    task.append(p3)
    task.append(p4)
    return task

def execute_Stemming(task):
    train = pd.read_csv("./Dataset/Stemming/train.csv")
    test = pd.read_csv("./Dataset/Stemming/test.csv")

    simple(train, test, "stemming")
    p1 = multiprocessing.Process(target=simple,
                            args=(
                                pd.read_csv("./Dataset/Stemming/train.csv"),
                                pd.read_csv("./Dataset/Stemming/test.csv"),
                                "stemming"))
    p2 = multiprocessing.Process(target=ros,
                            args=(
                                pd.read_csv("./Dataset/Stemming/train.csv"),
                                pd.read_csv("./Dataset/Stemming/test.csv"),
                                "stemming"))
    
    p3 = multiprocessing.Process(target=rus,
                            args=(
                                pd.read_csv("./Dataset/Stemming/train.csv"),
                                pd.read_csv("./Dataset/Stemming/test.csv"),
                                "stemming"))
    
    p4 = multiprocessing.Process(target=backTranslation,
                            args=(
                                pd.read_csv("./Dataset/Stemming/train.csv"),
                                pd.read_csv("./Dataset/Stemming/test.csv"),
                                "stemming"))
    task.append(p1)
    task.append(p2)
    task.append(p3)
    task.append(p4)
    return task

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

import multiprocessing

processos = []

task = []
task = execute_preProcessed(task)
task = execute_Lematization(task)
task = execute_nonStopWords(task)
task = execute_Stemming(task)

processos.extend(task)

# Execução
for p in processos:
    p.start()

for p in processos:
    p.join()

    print("FIM DA EXECUÇÃO :)")
