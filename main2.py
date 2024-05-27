import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from Grids_Search_Models.Logistic_Regression import grid_logistic_regression
from Utils.PreProcessing import preProcessing
from Utils.BackTranslation import dataframeWithBackTranslation
from Utils.Ros import ROS
from Utils.Rus import RUS
from Utils.PreProcessing import Stemming
from Utils.PreProcessing import use_stopwords
from Utils.PreProcessing import useLematization

df = pd.read_csv("./Dataset/rotulados.csv")

X_train, X_test, y_train, y_test = train_test_split(df['text'],df['label'],test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# grid_linear_regression(X_train, y_train, X_test, y_test, type)
grid_logistic_regression(X_train, y_train, X_test, y_test, "TEST")


