import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Wczytanie danych
data = pd.read_csv('user_data.csv')

# Usunięcie zbędnych kolumn
df = data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'user_id'])

# Wyświetlenie pierwszych wierszy
print(df.head())

# Skalowanie danych
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['average_rating', 'room_id']])

# Konwersja do DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=['average_rating', 'room_id'])
print(df_scaled.head())

# Klasteryzacja DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=2)
dbscan.fit(df_scaled)

# Dodanie etykiet klastrów do oryginalnego DataFrame
df['Cluster'] = dbscan.labels_

# Wyświetlenie pierwszych wierszy z przypisanymi klastrami
print(df.head())

# Wizualizacja wyników
plt.scatter(df['average_rating'], df['room_id'], c=df['Cluster'], cmap='rainbow')
plt.title('Grupowanie DBSCAN: Ocena ogólna vs Komfort')
plt.xlabel('Średnia ocena')
plt.ylabel('id')
plt.show()
