import nltk
from nltk.tokenize import word_tokenize

# Przykładowe zapytanie
text = "My laptop is overheating after the update."

# Tokenizacja tekstu
tokens = word_tokenize(text)
print(tokens)


# Tagowanie POS
tagged_tokens = nltk.pos_tag(tokens)
print(tagged_tokens)


import spacy

# Załaduj model NER
nlp = spacy.load("en_core_web_sm")

# Przetwórz tekst
doc = nlp(text)

# Wydobywanie encji
for ent in doc.ents:
    print(ent.text, ent.label_)


from transformers import pipeline

# Inicjalizacja analizy sentymentu
sentiment_analyzer = pipeline('sentiment-analysis')

# Analiza sentymentu zapytania
result = sentiment_analyzer(text)
print(result)
