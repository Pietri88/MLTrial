import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Załaduj zbiór danych
data = pd.read_csv('data/tweet_emotion_intensity.csv')

# Jeśli kolumna z etykietami nazywa się "labels", a chcemy "label", zmień nazwę:
if "labels" in data.columns and "label" not in data.columns:
    data = data.rename(columns={"labels": "label"})

# Sprawdź unikalne etykiety oraz ich liczbę
unique_labels = data['label'].unique()
num_labels = len(unique_labels)
print("Unikalne etykiety:", unique_labels)
print("Liczba etykiet:", num_labels)

# Tokenizator - użyjemy BERTa
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Funkcja do tokenizacji pojedynczego przykładu (tekstu)
def tokenize_function(text):
    return tokenizer(text, padding="max_length", truncation=True, max_length=128)

# Tokenizuj kolumnę 'tweet' i dodaj wynik do nowych kolumn: input_ids oraz attention_mask
data["tokenized"] = data["tweet"].apply(lambda x: tokenize_function(x))
data["input_ids"] = data["tokenized"].apply(lambda x: x["input_ids"])
data["attention_mask"] = data["tokenized"].apply(lambda x: x["attention_mask"])
data = data.drop(columns=["tokenized"])

# Podziel zbiór danych na: treningowy (70%), walidacyjny (15%) i testowy (15%)
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")
print(f"Test set size: {len(test_data)}")

# Utwórz niestandardowy Dataset opakowujący DataFrame
class TweetDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        return {
            'input_ids': torch.tensor(row['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(row['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(row['label'], dtype=torch.long)
        }

# Opakuj zbiory danych w obiekty Dataset
train_dataset = TweetDataset(train_data)
val_dataset = TweetDataset(val_data)
test_dataset = TweetDataset(test_data)

# Załaduj wstępnie wytrenowany model BERT do klasyfikacji sekwencji
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# Ustaw argumenty treningowe
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",  # Ostrzeżenie: w przyszłych wersjach użyj eval_strategy
    logging_dir='./logs',
)

# Utwórz obiekt Trainer, który obsługuje proces dostrajania
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=lambda p: {
        'accuracy': accuracy_score(p.label_ids, p.predictions.argmax(axis=-1)),
        'f1': f1_score(p.label_ids, p.predictions.argmax(axis=-1), average='weighted'),
        'precision': precision_score(p.label_ids, p.predictions.argmax(axis=-1), average='weighted'),
        'recall': recall_score(p.label_ids, p.predictions.argmax(axis=-1), average='weighted'),
    }
)

# Rozpocznij dostrajanie modelu
trainer.train()

# Oceń model na zbiorze testowym
predictions_output = trainer.predict(test_dataset)
predictions = predictions_output.predictions.argmax(axis=-1)  # Zakładamy zadanie klasyfikacyjne

# Oblicz metryki oceny
accuracy_val = accuracy_score(test_data['label'], predictions)
f1_val = f1_score(test_data['label'], predictions, average='weighted')
precision_val = precision_score(test_data['label'], predictions, average='weighted')
recall_val = recall_score(test_data['label'], predictions, average='weighted')

print(f"Test Accuracy: {accuracy_val}")
print(f"Test F1 Score: {f1_val}")
print(f"Test Precision: {precision_val}")
print(f"Test Recall: {recall_val}")

# (Opcjonalnie) Użyj wyszukiwania hiperparametrów do optymalizacji dostrajania
best_model = trainer.hyperparameter_search(
    direction="maximize",
    n_trials=10
)
