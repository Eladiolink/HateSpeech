from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
def RUS(train,test):
        # RUS
        ros = RandomUnderSampler(sampling_strategy='majority')
        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(train["text"].fillna(''))
        X_train , y_train = ros.fit_resample(X_train, train["label"])

        df = pd.DataFrame({
            'text': X_train,
            'label': y_train
        })

        df.to_csv('./Dataset/DataAugmentations/RUS/train.csv', index=False)
        test.to_csv('./Dataset/DataAugmentations/RUS/test.csv', index=False)
        return X_train, train, test["text"], test["label"]

        return train["text"], train["label"], test["text"], test["label"]
