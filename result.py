import argparse

parer = argparse.ArgumentParser(description="COISAS")

# parer.add_argument('--type', type=str,required=True,help='modelo')

# args = parer.parse_args()

from Utils.Results import gets_models_result, print_results

svm = gets_models_result("Model5/SVM","SVM")
nb = gets_models_result("Model5/Naive_Bayes","Naive Bayes")
xg = gets_models_result("Model5/XGboost","XGBoost")
ada = gets_models_result("Model5/AdaBoost","AdaBoost")


svm_p = []
nb_p = []
ada_p = []
xg_p = []
for i in svm:
    svm_p.append(i['1']['precision'])

for i in nb:
    nb_p.append(i['1']['precision'])

for i in ada:
    ada_p.append(i['1']['precision'])

for i in xg:
    xg_p.append(i['1']['precision'])

svm_r = []
nb_r = []
ada_r = []
xg_r = []
for i in svm:
    svm_r.append(i['1']['recall'])

for i in nb:
    nb_r.append(i['1']['recall'])

for i in ada:
    ada_r.append(i['1']['recall'])

for i in xg:
    xg_r.append(i['1']['recall'])


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# Dados de exemplo para diferentes grupos
# df = {
#     'Grupo': ['SVM'] * 16 + ['Naive Bayes'] * 16 + ['AdaBoost'] * 16 +  ['XGBoost'] * 16,
#     'Valores': svm_p + nb_p +ada_p+xg_p
# }

df = {
    'Modelos': ['SVM P'] * 16 + ['Naive Bayes P'] * 16 + ['AdaBoost P'] * 16 +  ['XGBoost P'] * 16 + ['SVM R'] * 16 + ['Naive Bayes R'] * 16 + ['AdaBoost R'] * 16 +  ['XGBoost R'] * 16,
    'Valores': svm_r + nb_r +ada_r+xg_r+svm_p + nb_p +ada_p+xg_p
}

# df = {
#     'Grupo': ['SVM'] * 16 + ['Naive Bayes'] * 16 + ['AdaBoost'] * 16 +  ['XGBoost'] * 16,
#     'Valores': svm_p + nb_p +ada_p+xg_p
# }

# Convertendo os dados para um DataFrame
df = pd.DataFrame(df)

sns.set_context("notebook", font_scale=1.5)
# Criando o boxplot
sns.boxplot(x='Modelos', y='Valores', data=df,palette="coolwarm")

# Adicionando título e rótulos
plt.title('Precisão e Recall na classe 1')
plt.xlabel('Modelos',fontsize=14)
plt.ylabel('Valores',fontsize=14)

# Exibindo o gráfico
plt.show()
