# 1. Instalacja wymaganych bibliotek (jeśli używasz Google Colab odkomentuj poniższe)
# !pip install transformers datasets scikit-learn

# 2. Importy
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import os

# 3. Ustawienia środowiska
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MLFLOW_TRACKING_URI"] = "disable"
os.environ["HF_MLFLOW_LOGGING"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4. Przykładowa baza danych opinii
data = {
    "text": [
        "This movie was absolutely fantastic! The story was gripping and the acting was superb.",
        "Terrible movie. Waste of time. The plot made no sense.",
        "I really enjoyed this film, would definitely recommend it!",
        "Not my type of movie, very boring and predictable.",
        "An excellent performance by the lead actor. A must-watch!",
        "I fell asleep halfway through. Very dull.",
        "Great visuals and music, but the story was weak.",
        "One of the worst films I've ever seen.",
        "Loved the character development and the twists!",
        "Bad acting and worse dialogue. Skip this one."
    ],
    "label": [1, 0, 1, 0, 1, 0, 0, 0, 1, 0]  # 1 = pozytywna, 0 = negatywna
}

# 5. Przygotowanie zbioru danych
df = pd.DataFrame(data)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 6. Tokenizacja
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# 7. Załaduj model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.to(device)

# 8. Argumenty treningowe
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    report_to="none",
)

# 9. Trener
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# 10. Trening modelu
trainer.train()

# 11. Ewaluacja
results = trainer.evaluate()
print(f"\n✅ Wyniki ewaluacji:")
for key, value in results.items():
    print(f"{key}: {value:.4f}")
