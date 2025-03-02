import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn import tree


# Przykładowe dane: liczba godzin nauki, wcześniejsze wyniki egzaminów i etykiety zdania/niezdania
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Nie zdał, 1 = Zdał
}

# Konwersja do DataFrame
df = pd.DataFrame(data)

x = df[['StudyHours', 'PrevExamScore']]
y = df['Pass']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("Dane treningowe", x_train.shape, y_train.shape)
print("Dane testwe", x_test.shape, y_test.shape)

model = DecisionTreeClassifier(random_state=42)

model.fit(x_train, y_train)

print("Głębokość drzewa", model.get_depth())
print("Liczba liści", model.get_n_leaves())

y_pred = model.predict(x_test)

print("Przewidziane wyniki (zdał, nie zdał)", y_pred)
print("Rzeczyiwste wyniki", y_test.values)

accuracy = accuracy_score(y_test, y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)

class_report = classification_report(y_test, y_pred)

print("dokładność modelu", accuracy)
print("Macierz pomyłek", conf_matrix)
print("Raport klasyfikacji", class_report)


model_tuned = DecisionTreeClassifier(max_depth=3, random_state=42)

model_tuned.fit(x_train, y_train)

y_pred_tuned = model_tuned.predict(x_test)


plt.figure(figsize=(12,8))
tree.plot_tree(model_tuned, feature_names=['StudyHours', 'PrevExamScores'], class_names=['Nie zdał', 'Zdał'], filled=True)
plt.title("Drzewo decyzyjne klasyfikujące zdanie egzaminu")
plt.show()