# Import bibliotek
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn import tree
import matplotlib
matplotlib.use('TkAgg')




# Przykładowy zbiór danych: Godziny nauki, wynik z poprzedniego egzaminu i wynik końcowy
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Nie zdał, 1 = Zdał
}

# Konwersja do DataFrame
df = pd.DataFrame(data)


# Definiowanie cech (X) i zmiennej docelowej (y)
x= df[['StudyHours', 'PrevExamScore']]  # Cechy
y = df['Pass']  # Zmienna docelowa

# Podział na 80% danych treningowych i 20% testowych
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"Dane treningowe: {x_train.shape}, {y_train.shape}")
print(f"Dane testowe: {x_test.shape}, {y_test.shape}")


logreg_model = LogisticRegression()

logreg_model.fit(x_train, y_train)

y_pred_logreg = logreg_model.predict(x_test)


tree_model = DecisionTreeClassifier(random_state=42)

tree_model.fit(x_train, y_train)

y_pred_tree = tree_model.predict(x_test)

accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print("Regresja logistyczna:")
print(f"Dokładność: {accuracy_logreg}")
print("Macierz pomyłek:")
print(confusion_matrix(y_test, y_pred_logreg))
print("Raport klasyfikacji:")
print(classification_report(y_test, y_pred_logreg))

accuracy_tree = accuracy_score(y_test, y_pred_tree)
print("\nDrzewo decyzyjne:")
print(f"Dokładność: {accuracy_tree}")
print("Macierz pomyłek:")
print(confusion_matrix(y_test, y_pred_tree))
print("Raport klasyfikacji:")
print(classification_report(y_test, y_pred_tree))


plt.figure(figsize=(12,8))
tree.plot_tree(tree_model, feature_names=['StudyHours', 'PrevExamScore'], class_names=['Nie zdał', 'Zdał'], filled=True)
plt.title('Drzewo decyzyjne dla klasyfikacji zdania/niezdania')
plt.show()