import nltk
from nltk.tokenize import word_tokenize

# Przykładowe zapytanie użytkownika
query = "My laptop is overheating after the update."

# Tokenizacja zapytania
tokens = word_tokenize(query)
print(tokens)

# Tagowanie POS tokenów
tagged_tokens = nltk.pos_tag(tokens)
print(tagged_tokens)


import spacy

# Załaduj wstępnie wytrenowany model do NER
nlp = spacy.load("en_core_web_sm")

# Zastosuj NER do zapytania
doc = nlp(query)

# Wyodrębnij i wypisz encje
for ent in doc.ents:
    print(ent.text, ent.label_)


from transformers import pipeline

# Inicjalizacja pipeline analizy sentymentu
sentiment_analyzer = pipeline('sentiment-analysis')

# Analiza sentymentu zapytania
result = sentiment_analyzer(query)
print(result)


# Przykładowa baza wiedzy
knowledge_base = {
    "overheating": "Check your cooling system, clean the fans, and ensure proper ventilation.",
    "slow performance": "Close unnecessary applications, restart your system, and check for malware."
}

# Funkcja pobierająca rozwiązanie
def get_solution(issue):
    return knowledge_base.get(issue, "No solution found for this issue.")

def troubleshoot(query):
    if "overheating" in query.lower():
        return get_solution("overheating")
    elif "slow" in query.lower():
        return get_solution("slow performance")
    else:
        return "Can you provide more details about the issue?"

# Przykład użycia
response = troubleshoot(query)
print(response)
