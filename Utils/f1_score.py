from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer


def f1_score_minority(y_true, y_pred):
    f1_scores = f1_score(y_true, y_pred, average=None)
    return f1_scores[1]

# Criar o scorer personalizado
f1_scorer = make_scorer(f1_score_minority)