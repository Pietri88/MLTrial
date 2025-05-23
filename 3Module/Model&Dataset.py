from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from nlpaug.augmenter.word import BackTranslationAug

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)


back_translation_aug = BackTranslationAug(from_model_name='facebook/wmt19-en-de', to_model_name='facebook/wmt19-de-en')

text = "The weather is great today."

augmented_text = back_translation_aug.augment(text)

print("Original text:", text)
print("Augmented text:", augmented_text)

dataset = load_dataset('imdb')

train_data, val_data = train_test_split(dataset['train'], test_size=0.2)

# Convert the data into the format required for tokenization
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_train = train_data.map(tokenize_function, batched=True)
tokenized_val = val_data.map(tokenize_function, batched=True)


tokenized_train = tokenizer(
    train_data['text'], padding=True, truncation=True, return_tensors="pt"
)
tokenized_val = tokenizer(
    val_data['text'], padding=True, truncation=True, return_tensors="pt"
)