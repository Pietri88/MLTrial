import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import hashlib
import cryptography.fernet

# Validate and sanitize input data
def validate_data(df):
    if df.isnull().values.any():
        raise ValueError("Dataset contains null values. Please clean the data before processing.")
    return df

# Load dataset
data = pd.read_csv('user_data.csv')

# Usuwanie niepotrzebnych kolumn
data = data.drop(['Unnamed: 0', 'user_id', 'room_id'], axis=1)

# Split the dataset into features and target
X = data[['hygeine_rating', 'staff_rating', 'comfort_rating']] # Wybór cech
y_original = data['average_rating'] # Oryginalna kolumna average_rating
y = y_original.round().astype(int)  # Zaokrąglona kolumna docelowa

# Sprawdzenie rozkładu klas
print("Rozkład klas w zmiennej docelowej (y):")
print(y.value_counts())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Skalowanie cech
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Logistic Regression z skalowaniem i Grid Search ---
print("\nLogistic Regression with Scaled Features and Grid Search:")
param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search_lr = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_lr, cv=5, scoring='accuracy') # zwiększ max_iter
grid_search_lr.fit(X_train_scaled, y_train)

print("Najlepsze parametry Logistic Regression:", grid_search_lr.best_params_)
best_lr_model = grid_search_lr.best_estimator_
y_pred_lr = best_lr_model.predict(X_test_scaled)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f'Accuracy Logistic Regression (Scaled, GridSearch): {accuracy_lr:.2f}')
print(classification_report(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))


# --- Random Forest ---
print("\nRandom Forest:")
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train) # Używamy nieskalowanych cech dla RandomForest
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Accuracy Random Forest: {accuracy_rf:.2f}')
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))


# --- Zapisanie najlepszego modelu (np. Random Forest jeśli lepszy) ---
best_model_to_save = rf_model # Zmień na best_lr_model jeśli LR lepszy
filename = 'finalized_model.sav'
pickle.dump(best_model_to_save, open(filename, 'wb'))
print(f"\nModel zapisany jako {filename}")

# --- Kod dotyczący szyfrowania i weryfikacji integralności (pozostaje bez zmian) ---
# Encrypt model before saving (przykład, można dostosować)
key = cryptography.fernet.Fernet.generate_key()
cipher = cryptography.fernet.Fernet(key)

# Save the encrypted model to disk
filename_encrypted = 'finalized_model_encrypted.sav' # Zmień nazwę pliku
encrypted_model = cipher.encrypt(pickle.dumps(best_model_to_save))
with open(filename_encrypted, 'wb') as f:
    f.write(encrypted_model)

# Load the encrypted model from disk and verify its integrity
with open(filename_encrypted, 'rb') as f:
    encrypted_model = f.read()
    decrypted_model = cipher.decrypt(encrypted_model)

loaded_model = pickle.loads(decrypted_model)

# Compute hash of the loaded model
loaded_model_hash = hashlib.sha256(decrypted_model).hexdigest()

# Verify that the loaded model's hash matches the original
original_model_hash = hashlib.sha256(pickle.dumps(best_model_to_save)).hexdigest()
if loaded_model_hash != original_model_hash:
    raise ValueError("Model integrity check failed. The model may have been tampered with.")

result = loaded_model.score(X_test, y_test)
print(f'\nAccuracy of loaded and verified model: {result:.2f}')