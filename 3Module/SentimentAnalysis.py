from transformers import pipeline

# Inicjalizacja analizatora sentymentu
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Sample texts for sentiment analysis
texts = [
    "I love this product! It's amazing.",
    "The service was terrible and I'm very disappointed.",
    "It's okay, not great but not bad either."
]

# Analyze the sentiment of each text
for text in texts:
    result = sentiment_analyzer(text)
    print(f"Text: {text}")
    print(f"Sentiment: {result[0]['label']}")
    print(f"Confidence: {result[0]['score']:.2f}")
    print()  # Empty line for readability


# Accept user input for custom sentiment analysis
custom_text = input("Enter a sentence for sentiment analysis: ")

# Analyze the sentiment
result = sentiment_analyzer(custom_text)

print(f"\nSentiment: {result[0]['label']}")
print(f"Confidence: {result[0]['score']:.2f}")


# Allow the model to process a longer paragraph of text
long_text = """
The product is good overall, but there are some issues with battery life. 
I wish it lasted longer. However, the design is sleek, and I’m happy with the performance so far.
"""
result = sentiment_analyzer(long_text)
for res in result:
    print(f"Sentiment: {res['label']}, Confidence: {res['score']:.2f}")