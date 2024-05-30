from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd


def RUS(train, test):
    # RUS
    undersample = RandomUnderSampler(sampling_strategy='majority')
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train["text"].fillna(''))
    X_over, y_over = undersample.fit_resample(X_train, train['label'])

    return X_over, y_over, test["text"], test["label"]
def R2S2(train, test):
    # ROS
    ros = RandomUnderSampler(random_state=42)
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