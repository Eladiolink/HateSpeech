from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from Utils.save_results import save_results
from Utils.f1_score import f1_scorer

def grid_naive_bayes(X_train,y_train,X_test,y_test,type):
    print("Rodando Naive Bayes +",type)
    path = "./Model/Naive_Bayes/best_model_"+str(type)+".txt"
    model = MultinomialNB()

    # Definir os parâmetros para o GridSearch
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        'fit_prior': [True,False],
        'class_prior': [[0.5714,0.4286]],
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
    accuracy = best_model.score(X_test, y_test)
    print("Acurácia no conjunto de teste: ", accuracy)
    report = classification_report(y_test,best_model.predict(X_test))
    print(report)
    save_results(path,grid_search, best_model,report,X_test,y_test)