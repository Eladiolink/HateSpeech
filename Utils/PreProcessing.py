import spacy
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import emoji
import re

# Carregar o modelo em português do spacy
nlp = spacy.load("pt_core_news_sm")

def Stemming(dados,campo):
    stemmer = SnowballStemmer("portuguese")
    for i, infos in dados.iterrows():
        dado = str(infos[campo])

        palavras = word_tokenize(dado, language='portuguese')
        stemmed_palavras = [stemmer.stem(palavra) for palavra in palavras]
        frase_stemmed = ' '.join(stemmed_palavras)

        dados.at[i,campo] = frase_stemmed
    return dados

def preProcessing(dados,campo):
    for i, infos in dados.iterrows():
        dado = infos[campo]

        dado = remover_acentos(dado)

        parcial = re.sub(r'[@#]\w+\s?', '', dado)  # Processamento para remover acentuação
        dado = re.sub(r'\s+', ' ', parcial)
        dado = dado.lower()  # Deixar todo em lower case
        dados.at[i,campo] = dado

    for i, infos in dados.iterrows():
        dic = infos[campo]
        if dic == "" or bool(
                re.match(r'\s*$', dic)):  # remover comentario vazio ou só de espaços
            dados.drop(i)

    for i, infos in dados.iterrows():
        dic = infos[campo]
        if dic == "" or bool(
                re.match(r'\s*$', dic)):  # remover comentario vazio ou só de espaços
            dados.drop(i)

    # remove links
    for i, infos in dados.iterrows():
        dic = infos[campo]
        url_regex = r'(https?://\S+|www\.\S+)'
        dados.at[i,campo] = re.sub(url_regex, '', dic)

    # Remove Emojis
    for i, infos in dados.iterrows():
        dic = infos[campo]

        dados.at[i,campo] = emoji_to_text(dic)

    for i, infos in dados.iterrows():
        dic = infos[campo]

        dados.at[i,campo] = remove_specific_punctuation(dic)

    print("Tamanho da Base de Dados: ", len(dados),' in ',campo)
    return dados

def emoji_to_text(text):
    text_with_descriptions = emoji.demojize(text,language='pt')
    return text_with_descriptions.replace(":", "").replace("_", " ")

def remove_specific_punctuation(text):
    pattern = r'[?!()|\\/;,<>.:}\]{\[]'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text
def remover_acentos(texto):
    # Dicionário de mapeamento de caracteres especiais para sem acentuação
    mapeamento = {
        'á': 'a', 'à': 'a', 'ã': 'a', 'â': 'a',
        'é': 'e', 'è': 'e', 'ê': 'e',
        'í': 'i', 'ì': 'i', 'î': 'i',
        'ó': 'o', 'ò': 'o', 'õ': 'o', 'ô': 'o',
        'ú': 'u', 'ù': 'u', 'û': 'u',
        'ç': 'c',
        'Á': 'A', 'À': 'A', 'Ã': 'A', 'Â': 'A',
        'É': 'E', 'È': 'E', 'Ê': 'E',
        'Í': 'I', 'Ì': 'I', 'Î': 'I',
        'Ó': 'O', 'Ò': 'O', 'Õ': 'O', 'Ô': 'O',
        'Ú': 'U', 'Ù': 'U', 'Û': 'U',
        'Ç': 'C'
    }

    # Substituir caracteres especiais pelo mapeamento
    for caracter_especial, sem_acentuacao in mapeamento.items():
        texto = texto.replace(caracter_especial, sem_acentuacao)

    return texto


# Função para aplicar o lematization do spacy na base de dados
def useLematization(dados,campo):

    for i, infos in dados.iterrows():
        comentario = str(infos[campo])
        doc = nlp(comentario)

        frase_lematizada = " ".join([token.lemma_ for token in doc])
        dados.at[i,campo] = frase_lematizada

    return dados

# Stopwords

# Baixe o conjunto de dados de stopwords
nltk.download('stopwords')
nltk.download('punkt')

def use_stopwords(dados,campo):
    for i, infos in dados.iterrows():
        dic = str(infos[campo])
        dados.at[i,campo] = remove_stopwords(dic)

    return dados

# Função para remover stopwords
def remove_stopwords(text, language='portuguese'):
    stop_words = set(stopwords.words(language))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

