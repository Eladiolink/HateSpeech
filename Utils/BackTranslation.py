import pandas as pd
from googletrans import Translator

# Dataframe in Pandas
def backTranslation(dataframe,target,percent):
    print("Traduzindo...")
    label_selecionada = 1
    tamanho = int(dataframe["label"].value_counts()[label_selecionada] * percent)
    # Seleciona linhas aleatórias com a label selecionada
    print(tamanho)
    linhas_aleatorias = dataframe.loc[dataframe['label'] == label_selecionada].sample(n=tamanho)

    for i, infos in linhas_aleatorias.iterrows():
        data = infos["text"]

        if data == None:
            continue

        try:
            data_taducted = traduzir_frase(data)
            data_taducted_back = traduzir_frase(data_taducted,'ch','pt')
            linhas_aleatorias.at[i, 'text'] = data_taducted_back
        except Exception:
            print("Erro: ",data)


    return pd.concat([dataframe,linhas_aleatorias])

def traduzir_frase(frase, idioma_origem='auto', idioma_destino='ch'):
    translator = Translator()
    traducao = translator.translate(frase, src=idioma_origem, dest=idioma_destino)
    return traducao.text

def dataframeWithBackTranslation(train,test):
        train = backTranslation(train, "text", 0.4)
        train = backTranslation(train, "text", 0.15)

        train.to_csv('./Dataset/DataAugmentations/BackTranslation/train.csv', index=False)
        test.to_csv('./Dataset/DataAugmentations/BackTranslation/test.csv', index=False)

        return train["text"], train["label"], test["text"], test["label"]