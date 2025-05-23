from sklearn.metrics import accuracy_score

y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 1, 0, 0, 1]

# Obliczenie dokładności
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")


from sklearn.metrics import precision_score

# Obliczenie precyzji
precision = precision_score(y_true, y_pred)
print(f"Precision: {precision}")


from sklearn.metrics import recall_score

# Obliczenie czułości
recall = recall_score(y_true, y_pred)
print(f"Recall: {recall}")


from sklearn.metrics import f1_score

# Obliczenie miary F1
f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1}")


from sklearn.metrics import confusion_matrix

# Generowanie macierzy pomyłek
matrix = confusion_matrix(y_true, y_pred)
print(matrix)


from sklearn.metrics import roc_auc_score

# Obliczenie wyniku ROC-AUC
roc_auc = roc_auc_score(y_true, y_pred)
print(f"ROC-AUC: {roc_auc}")


import torch
import torch.nn as nn

# Define the cross-entropy loss function
# CrossEntropyLoss is used for classification tasks where the model outputs class probabilities.
# It combines LogSoftmax and Negative Log Likelihood Loss into one function, making it efficient for such tasks.
loss_fn = nn.CrossEntropyLoss()

# Example prediction and actual class (as tensors)
# Here, we create a tensor called 'output' representing the predicted scores (unnormalized) for two data points.
# Each row corresponds to a data point, and the values represent the scores for each class.
# Note that CrossEntropyLoss internally applies the softmax function to these scores to obtain probabilities.
output = torch.tensor([[0.5, 1.5], [2.0, 0.5]])

# 'target' is a tensor representing the actual classes for the two data points.
# In this example, the first data point belongs to class 1, and the second data point belongs to class 0.
# These class indices are zero-based, meaning 0 represents the first class, 1 represents the second, and so on.
target = torch.tensor([1, 0])

# Calculate loss
# The CrossEntropyLoss function will take the predicted scores ('output') and the actual labels ('target')
# to compute the loss value, which quantifies how well the model's predictions match the actual labels.
# Lower loss values indicate better predictions, while higher values indicate more errors.
loss = loss_fn(output, target)

# Print the computed loss value
# '.item()' is used to get the Python scalar value from the tensor containing the loss.
print(f"Loss: {loss.item()}")