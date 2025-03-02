import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# Przykładowy zbiór danych
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Nie zaliczył, 1 = Zaliczył
}

df = pd.DataFrame(data)


X = df[['StudyHours', 'PrevExamScore']]
y = df['Pass']

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print(model.summary())

significance_level = 0.05

while True:
    model = sm.OLS(y, X).fit()

    p_values = model.pvalues

    max_p_value = p_values.max()

    if max_p_value > significance_level:
        excluded_feature = p_values.idxmax()
        X = X.drop(columns=[excluded_feature])
        print("Usunięto cechę ", excluded_feature, " p= ", max_p_value)
    else:
        break



print(model.summary())