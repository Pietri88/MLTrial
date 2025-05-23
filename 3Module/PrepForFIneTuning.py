# Install modules
# A '!' in a Jupyter Notebook runs the line in the system's shell, and not in the Python interpreter

# Import necessary libraries
import pandas as pd
import random
import re
from transformers import BertTokenizer
# Import necessary modules
import random # Random module for generating random numbers and selections
import nltk
nltk.download('wordnet')

from nltk.corpus import wordnet # NLTK's WordNet corpus for finding synonyms
# Load dataset 
import torch # Import PyTorch library
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split # Import function to split dataset
# you can download this dataset from https://huggingface.co/datasets/stepp1/tweet_emotion_intensity/tree/main
data = pd.read_csv('data/tweet_emotion_intensity.csv')

# Preview the data
print(data.head())

def clean_text(text):
    text = text.lower() # Konwersja całego tekstu na małe litery dla jednolitości
    text = re.sub(r'http\S+', '', text) # Usunięcie URL-i z tekstu
    text = re.sub(r'<.*?>', '', text) # Usunięcie wszelkich tagów HTML z tekstu
    text = re.sub(r'[^\w\s]', '', text) # Usunięcie znaków interpunkcyjnych, pozostawiając jedynie słowa i spacje
    return text # Zwróć oczyszczony tekst

# Assume `data` is a pandas DataFrame with a column named 'text'
# Apply the cleaning function to each row of the 'text' column
data['cleaned_text'] = data['tweet'].apply(clean_text)

# Print the first 5 rows of the cleaned text to verify the cleaning process
print(data['cleaned_text'].head())

print(data.isnull().sum())

data = data.dropna(subset=['cleaned_text'])

data['cleaned_text'] = data['cleaned_text'].fillna('unknown')


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tokens = tokenizer(
    data['cleaned_text'].tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt'
)

print(tokens['input_ids'][:5])

def synonym_replacement(word):
# Get all synsets (sets of synonyms) for the given word from WordNet
    synonyms = wordnet.synsets(word)

# If the word has synonyms, randomly choose one synonym, otherwise return the original word
    if synonyms:
# Select a random synonym and get the first lemma (word form) of that synonym
        return random.choice(synonyms).lemmas()[0].name()

# If no synonyms are found, return the original word
    return word

# Define a function to augment text by replacing words with synonyms randomly
def augment_text(text):
# Split the input text into individual words
    words = text.split() # Split the input text into individual words

# Replace each word with a synonym with a probability of 20% (random.random() > 0.8)
    augmented_words = [
    synonym_replacement(word) if random.random() > 0.8 else word 
# If random condition met, replace
for word in words] # Iterate over each word in the original text

# Join the augmented words back into a single string and return it
    return ' '.join(augmented_words)

# Apply the text augmentation function to the 'cleaned_text' column in a DataFrame
# Create a new column 'augmented_text' containing the augmented version of 'cleaned_text'
data['augmented_text'] = data['cleaned_text'].apply(augment_text)

input_ids = tokens['input_ids']
attention_masks = tokens['attention_mask']

def map_sentiment(value):
    if value == 'high':
        return 1
    elif value == 'medium':
        return 0.5
    elif value == 'low':
        return 0
    else:
        return None
    
data['sentiment_intensity'] = data['sentiment_intensity'].apply(map_sentiment)

data = data.dropna(subset=['sentiment_intensity']).reset_index(drop=True)

labels = torch.tensor(data['sentiment_intensity'].tolist())


train_val_inputs, test_inputs, train_val_masks, test_masks, train_val_labels, test_labels = train_test_split(
    input_ids, attention_masks, labels, test_size=0.15, random_state=42
)

# Second split: 20% for validation set from remaining data
train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = train_test_split(
    train_val_inputs, train_val_masks, train_val_labels, test_size=0.2, random_state=42
)

# Create TensorDataset objects for each set, including attention masks
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
val_dataset = TensorDataset(val_inputs, val_masks, val_labels)
test_dataset = TensorDataset(test_inputs, test_masks, test_labels)

# Create DataLoader objects
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)
test_dataloader = DataLoader(test_dataset, batch_size=16)

print("Training, validation, and test sets are prepared with attention masks!")