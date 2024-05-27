def save_results(path,grid_search,best_model,report,X_test,y_test):
    # salvar resultando em .txt
    with open(path, 'w') as f:
        f.write(f"melhores parametros: {grid_search.best_params_}\n")
        f.write(f"melhor score: {grid_search.best_score_}\n")
        f.write(f"melhor estimador: {grid_search.best_estimator_}\n")
        f.write(f"score do melhor modelo: {best_model.score(X_test, y_test)}\n")
        f.write(f"clasificador report: {report}\n")