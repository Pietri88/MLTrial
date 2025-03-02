import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)
print(df.head())


x = df[['StudyHours']]  # Feature(s)
y = df['Pass']          # Target variable (0 = Fail, 1 = Pass)

# Split data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"Training data: {x_train.shape}, {y_train.shape}")
print(f"Testing data: {x_test.shape}, {y_test.shape}")

model = LogisticRegression()
model.fit(x_train, y_train)

print(f"Wyraz wolnt (intercept): {model.intercept_}")
print(f"Wspólcznynnik regresji: {model.coef_[0]}")

y_pred = model.predict(x_test)

print("Przewidywane wyniki:", y_pred)
print("Rzeczywiste wynik:", y_test.values)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Dokładność: {accuracy}")
print("Macierz pomyłek")
print(conf_matrix)
print("raport klasyfikacji")
print(class_report)

study_hours_range = np.linspace(x.min(), x.max(), 100)
y_prob = model.predict_proba(study_hours_range.reshape(-1, 1))[:, 1]

plt.scatter(x_test, y_test, color='blue', label='Rzeczywsite dane')
plt.plot(study_hours_range, y_prob, color='red', label='Krzywa regresji logistycznej')
plt.xlabel('Liczba godzin nauki')
plt.ylabel('Prawdopodobieństwo zdania')
plt.title('Regresja logistyczna: Liczba godzin nauki vs. zdał/nie zdał')
plt.legend()
plt.show()


