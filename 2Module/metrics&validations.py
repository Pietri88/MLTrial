import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_error


# Przykładowy zestaw danych: Godziny nauki, poprzednie wyniki egzaminów i etykiety pass/fail
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Nie zdał, 1 = Zdał
}

df = pd.DataFrame(data)

# Cechy i zmienna docelowa
x = df[['StudyHours', 'PrevExamScore']]
y = df['Pass']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Zainicjuj i wytrenuj model regresji logistycznej
model = LogisticRegression()
model.fit(x_train, y_train)

# Dokonaj przewidywań na zestawie testowym
y_pred = model.predict(x_test)

# Oblicz metryki
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model = LogisticRegression()
cv_scores = cross_val_score(model, x, y, cv=5, scoring='accuracy')

print("Cross-valdiation accuration", cv_scores)

scoring = ['accuracy', 'precision', 'recall', 'f1']

cv_results = cross_validate(model, x,y,cv=5,scoring=scoring)
print("Cross-validatiion accuracy:", np.mean(cv_results['test_accuracy']))


x_reg = df[['StudyHours']]
y_reg = df['PrevExamScore']

reg_model = LinearRegression()

cv_scores_r2 = cross_val_score(reg_model, x_reg, y_reg, cv=5, scoring='r2')

print(f'Cross-validation R-squared scores: {cv_scores_r2}')
print(f'Mean R-squared score: {np.mean(cv_scores_r2)}')