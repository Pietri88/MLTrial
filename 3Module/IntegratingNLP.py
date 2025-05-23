import spacy
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

# Sample text
text = "Serena Williams won Wimbledon in 2016, solidifying her status as one of the greatest tennis players in history."

# Tokenize the text
tokens = word_tokenize(text)
print(tokens)

nltk.download('averaged_perceptron_tagger_eng')

# Tagowanie POS
tagged_tokens = nltk.pos_tag(tokens)
print(tagged_tokens)


# Załaduj wytrenowany model
nlp = spacy.load("en_core_web_sm")

# Przetwórz tekst
doc = nlp(text)

# Wyodrębnij encje
for ent in doc.ents:
    print(ent.text, ent.label_)


from transformers import pipeline

# Inicjalizacja modelu analizy sentymentu
sentiment_analyzer = pipeline('sentiment-analysis')

# Analiza sentymentu
result = sentiment_analyzer(text)
print(result)
