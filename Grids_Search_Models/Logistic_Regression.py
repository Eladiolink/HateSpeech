from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score,  accuracy_score ,recall_score, average_precision_score, f1_score
import numpy as np
from Utils.f1_score import f1_scorer

def grid_logistic_regression(X_train,y_train,X_test,y_test,type):
    print("Rodando Logitic +",type)
    path = "./Model/Logistic_Regression/best_model_"+str(type)+".txt"
    model = LogisticRegression()

    # Definir os parâmetros para o GridSearch
    param_grid = {
        'fit_intercept': [True,False],
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'max_iter': [20000],
        'class_weight':[None,'balanced',{0: 1, 1: 4}],
        'random_state':[42],
    }

    # Configurar o GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring=f1_scorer)
    print("Rodando o grid...")
    grid_search.fit(X_train, y_train)

    # Melhor combinação de parâmetros encontrada
    print("Melhores parâmetros encontrados: ", grid_search.best_params_)

    # Melhor resultado (acurácia) encontrado
    print("Melhor acurácia: ", grid_search.best_score_)

    # Modelo treinado com os melhores parâmetros
    best_model = grid_search.best_estimator_

    # Avaliar o modelo no conjunto de teste
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test,y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print("r2 score no conjunto de teste: ", r2)
    print(f"Mean Squared Error: {mse}")

    accuracy = accuracy_score(y_true=y_test,y_pred=y_pred)
    print("Accuracy score: ",accuracy)

    # calculando f1 score
    print("F1 averaged score :",f1_score(y_true=y_test, y_pred=y_pred, average = 'weighted'))
    
    # calculando a precision score
    print("Precision score :",average_precision_score(y_true=y_test, y_score=y_pred, average = 'weighted'))

    with open(path, 'w') as f:
        f.write(f"melhores parametros: {grid_search.best_params_}\n")
        f.write(f"melhor score: {grid_search.best_score_}\n")
        f.write(f"melhor estimador: {grid_search.best_estimator_}\n")
        f.write(f"score do melhor modelo: {best_model.score(X_test, y_test)}\n")
        f.write(f"clasificador r2-score: {r2}\n")
