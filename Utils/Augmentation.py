import spacy
from spacy.tokens import Doc
from spacy.symbols import NOUN, VERB, ADJ, ADV
from nltk.corpus import wordnet as wn
import random

# Carregar o modelo de idioma em português
nlp = spacy.load("pt_core_news_sm")

# Definir função para adicionar a extensão do WordNet ao Doc
def add_wordnet_to_doc(doc):
    doc_wn = []
    for token in doc:
        synsets = wn.synsets(token.text, lang='por')
        doc_wn.append(synsets)
    doc.set_extension('wordnet', getter=lambda: doc_wn)
    return doc

# Adicionar a extensão ao modelo de idioma
Doc.set_extension('wordnet', getter=add_wordnet_to_doc)

# Função para substituir palavras por sinônimos
def synonym_replacement(text, n=1):
    doc = nlp(text)
    new_text = text
    for _ in range(n):
        for token in doc:
            if token.pos in (NOUN, VERB, ADJ, ADV):  # Considerar apenas palavras com categorias gramaticais específicas
                syns = [lemma.lemma_ for synset in token._.wordnet for lemma in synset.lemmas()]
                if syns:
                    synonym = random.choice(syns)
                    new_text = new_text.replace(token.text, synonym, 1)
    return new_text

# Exemplo de uso
original_text = "O gato está dormindo no tapete."
augmented_text = synonym_replacement(original_text)
print("Original:", original_text)
print("Aumentada:", augmented_text)
