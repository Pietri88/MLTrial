from lora import LoRALayer
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import precision_score, recall_score, f1_score

from lora import adjust_lora_rank
# Załaduj zbiór danych
data = pd.read_csv('data/tweet_emotion_intensity.csv')

# Podziel zbiór danych na: treningowy (70%), walidacyjny (15%) i testowy (15%)
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")
print(f"Test set size: {len(test_data)}")


model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Apply LoRA to specific layers (e.g., attention layers)
for name, module in model.named_modules():
    if 'attention' in name:
        module.apply(LoRALayer)

# Freeze the rest of the model
for param in model.base_model.parameters():
    param.requires_grad = False


# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    logging_dir='./logs',
)

# Initialize Trainer for fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)

# Start fine-tuning the model
trainer.train()


# Evaluate the model on the test set
results = trainer.evaluate(eval_dataset=test_data)

# Extract predictions and true labels
predictions = trainer.predict(test_data).predictions.argmax(-1)
true_labels = test_data['label']

# Calculate accuracy, precision, recall, and F1 score
accuracy = results['eval_accuracy']
precision = precision_score(true_labels, predictions, average='weighted')
recall = recall_score(true_labels, predictions, average='weighted')
f1 = f1_score(true_labels, predictions, average='weighted')

# Print all evaluation metrics
print(f"Test Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

adjust_lora_rank(model, rank=4)  # Experiment with different rank values

# Experiment with additional parameters
alpha = 16 # Scaling factor for LoRA
dropout_rate = 0.1 # Dropout rate for regularization
use_bias = True # Whether to include bias in the model layers

# Example of modifying these parameters
if hasattr(model.config, 'alpha'):
    model.config.alpha = alpha
else:
    print("Warning: model.config does not have attribute 'alpha'")

if hasattr(model.config, 'hidden_dropout_prob'):
    model.config.hidden_dropout_prob = dropout_rate
else:
    print("Warning: model.config does not have attribute 'hidden_dropout_prob'")

if hasattr(model.config, 'use_bias'):
    model.config.use_bias = use_bias
else:
    print("Warning: model.config does not have attribute 'use_bias'")

print(f"Alpha: {alpha}")
print(f"Dropout Rate: {dropout_rate}")
print(f"Using Bias: {use_bias}")