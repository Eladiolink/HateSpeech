import pandas
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report
from Utils.save_results import save_results2
from sklearn.model_selection import train_test_split

def lstm(X_train, y_train, X_test, y_test, type):
    print("Rodando LSTM +", type)
    path = "./Model/LSTM/best_model_" + str(type) + ".txt"
    # Configurações
    max_words = 5000
    max_len = 100
    embedding_dim = 100

    # Tokenização
    tokenizer = Tokenizer(num_words=max_words)

    print(X_train)
    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)

    X_train = pad_sequences(X_train_sequences, maxlen=max_len)
    X_test = pad_sequences(X_test_sequences, maxlen=max_len)

    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Avaliar o modelo no conjunto de teste
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred)

    print(f'Acurácia no conjunto de teste: {accuracy:.2f}')
    print(report)

    # Salvar resultados - substitua save_results2 pelo método que você usa para salvar os resultados
    save_results2(path, report)

# df = pandas.read_csv("../Dataset/minimal.csv")
# df = df.dropna()
# X = df['text']
# y = df['label']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#
# lstm(X_train, y_train, X_test, y_test, "Lemmatization")
