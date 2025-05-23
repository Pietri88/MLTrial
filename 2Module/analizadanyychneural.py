import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, names=columns)


print(data.head())

print(data.isnull().sum())

print(data.describe())

print(data['Outcome'].value_counts())

X= data.drop('Outcome', axis=1)

y=data['Outcome']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Pierwsza warstwa ukryta
    layers.Dense(32, activation='relu'),  # Druga warstwa ukryta
    layers.Dense(1, activation='sigmoid')  # Warstwa wyjściowa dla klasyfikacji binarnej
])

# Kompilacja modelu
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))


# Ocena modelu
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Dokładność na zbiorze testowym: {test_accuracy}')

# Wykres dokładności
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Dokładność modelu')
plt.ylabel('Dokładność')
plt.xlabel('Epoka')
plt.legend(['Trening', 'Test'], loc='upper left')
plt.show()