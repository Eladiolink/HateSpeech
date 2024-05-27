from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def grid_linear_regression(X_train,y_train,X_test,y_test,type):
    print("Rodando Regressao Linear +",type)
    path = "./Model/Linear_Regression/best_model_"+str(type)+".txt"
    model = LinearRegression()

    # Definir os parâmetros para o GridSearch
    param_grid = {
        'fit_intercept': [True,False],
        'n_jobs': [1,2,4,-1]
    }

    # Configurar o GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2')
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

 
    # with open(path, 'w') as f:
    #     f.write(f"melhores parametros: {grid_search.best_params_}\n")
    #     f.write(f"melhor score: {grid_search.best_score_}\n")
    #     f.write(f"melhor estimador: {grid_search.best_estimator_}\n")
    #     f.write(f"score do melhor modelo: {best_model.score(X_test, y_test)}\n")
    #     f.write(f"clasificador r2-score: {r2}\n")