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


def ROS2(train, test):
    # ROS
    ros = RandomOverSampler(random_state=42)
    vectorizer = TfidfVectorizer()

    # Storing original text
    train_text = train["text"].fillna('')
    test_text = test["text"].fillna('')

    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)

    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, train["label"])

    # Getting indices of the resampled data
    _, indices = ros.fit_resample(pd.DataFrame(train.index), train["label"])

    # Mapping resampled indices to original text
    train_text_resampled = train_text.iloc[indices.values.flatten()]

    return train_text_resampled, y_train_resampled, test_text, test["label"]