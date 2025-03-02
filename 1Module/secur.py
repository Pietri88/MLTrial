import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import hashlib

# Walidacja i czyszczenie danych
def validate_data(df):
    if df.isnull().values.any():
        raise ValueError("Dataset contains null values. Please clean the data before processing.")
    
    if not all(df.dtypes.apply(lambda x: x.kind in 'biufc')):  # Sprawdzamy czy dane są numeryczne
        raise ValueError("Dataset contains non-numeric values. Please convert data before processing.")

    return df

# Bezpieczne ładowanie danych
try:
    data = validate_data(pd.read_csv('user_data.csv'))
    data["rating_category"] = pd.cut(data["average_rating"], bins=[0, 2, 4, 5], labels=[0, 1, 2])
except Exception as e:
    print(f"Błąd podczas ładowania danych: {e}")
    exit(1)

# Podział na cechy i etykiety
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Generowanie losowego seed'a jako liczba całkowita
random_seed = int.from_bytes(os.urandom(4), "big")

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# Trenowanie modelu regresji logistycznej
try:
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
except Exception as e:
    print(f"Błąd podczas trenowania modelu: {e}")
    exit(1)

# Zapis modelu do pliku z hashem dla weryfikacji integralności
filename = 'finalized_model.sav'
try:
    with open(filename, 'wb') as model_file:
        encrypted_model = pickle.dumps(model)
        model_file.write(encrypted_model)
    
    # Tworzymy hash dla weryfikacji
    model_hash = hashlib.sha256(encrypted_model).hexdigest()
    with open(f"{filename}.hash", "w") as hash_file:
        hash_file.write(model_hash)
except Exception as e:
    print(f"Błąd podczas zapisu modelu: {e}")
    exit(1)

# Odczyt modelu i weryfikacja integralności
try:
    with open(filename, 'rb') as model_file:
        loaded_model = pickle.loads(model_file.read())
    
    # Wczytanie zapisanego hash'a
    with open(f"{filename}.hash", "r") as hash_file:
        stored_hash = hash_file.read().strip()
    
    # Sprawdzamy, czy model nie został zmodyfikowany
    if hashlib.sha256(pickle.dumps(loaded_model)).hexdigest() != stored_hash:
        raise ValueError("Model integrity check failed. The model may have been tampered with.")
except Exception as e:
    print(f"Błąd podczas ładowania modelu: {e}")
    exit(1)

# Ocena modelu
result = loaded_model.score(X_test, y_test)
print(f'Model Accuracy: {result:.2f}')
