from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler
import pandas as pd

def ROS(train,test):
    # ROS
    ros = RandomOverSampler(random_state=42)
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train["text"].fillna(''))
    X_test = vectorizer.transform(test["text"].fillna(''))
    X_train , y_train = ros.fit_resample(X_train, train["label"])

    return X_train, y_train,X_test, test["label"]