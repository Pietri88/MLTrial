import pandas as pd
import torch
import re

data = pd.read_csv('customer_data.csv')

# Define mapping for labels
label_mapping = {'Bronze': 0, 'Silver': 1, 'Gold': 2}  # Assign numbers to each category

# Convert membership_level to numeric labels
data['label'] = data['membership_level'].map(label_mapping)

# Convert labels to PyTorch tensor
labels = torch.tensor(data['label'].tolist())
data['cleaned_text'] = ["Hello, I am a Bronze member!", 
                        "Silver membership offers perks.", 
                        "Gold members get premium benefits.", 
                        "Silver members enjoy discounts.", 
                        "Bronze is the starting tier."]


def clean_text(text):
    text = text.lower()  # Konwersja do małych liter
    text = re.sub(r'http\S+', '', text)  # Usunięcie URL
    text = re.sub(r'\s+', ' ', text).strip()  # Usunięcie nadmiarowych białych znaków
    text = re.sub(r'[^\w\s]', '', text)  # Usunięcie znaków interpunkcyjnych
    return text

# Apply cleaning function to your dataset
data['cleaned_text'] = data['text'].apply(clean_text)
print(data['cleaned_text'].head())



from transformers import BertTokenizer

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the cleaned text
tokens = tokenizer(
    data['cleaned_text'].tolist(), padding=True, truncation=True, return_tensors='pt', max_length=128
)

print(tokens['input_ids'][:5])  # Check the first 5 tokenized examples


# Check for missing data
print(data.isnull().sum())

# Option 1: Drop rows with missing data
data = data.dropna()

# Option 2: Fill missing values with a placeholder
data['cleaned_text'].fillna('missing', inplace=True)

import torch
from torch.utils.data import TensorDataset, DataLoader

# Create PyTorch tensors from the tokenized data
input_ids = tokens['input_ids']
attention_masks = tokens['attention_mask']
labels = torch.tensor(data['label'].tolist())

# Create a DataLoader for training
dataset = TensorDataset(input_ids, attention_masks, labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

print("DataLoader created successfully!")

from sklearn.model_selection import train_test_split

# First, split data into a combined training + validation set and a test set
train_val_inputs, test_inputs, train_val_labels, test_labels = train_test_split(
    input_ids, labels, test_size=0.1, random_state=42
)

# Now, split the combined set into training and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    train_val_inputs, train_val_labels, test_size=0.15, random_state=42
)

# Create DataLoader objects for training, validation, and test sets
train_dataset = TensorDataset(train_inputs, train_labels)
val_dataset = TensorDataset(val_inputs, val_labels)
test_dataset = TensorDataset(test_inputs, test_labels)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)
test_dataloader = DataLoader(test_dataset, batch_size=16)

print("DataLoader objects for training, validation, and test sets created successfully!")