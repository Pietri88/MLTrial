import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
os.environ["OMP_NUM_THREADS"] = "1"

# Przykładowy zbiór danych: Roczny dochód klienta, wynik wydatków oraz wiek
data = {'AnnualIncome': [15, 16, 17, 18, 19, 20, 22, 25, 30, 35],
        'SpendingScore': [39, 81, 6, 77, 40, 76, 94, 5, 82, 56],
        'Age': [20, 22, 25, 24, 35, 40, 30, 21, 50, 31]}

df = pd.DataFrame(data)

# Wyświetlenie pierwszych kilku wierszy zbioru danych
print(df.head())

# Normalizacja zbioru danych za pomocą StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Przekształcenie danych znormalizowanych z powrotem do DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=['AnnualIncome', 'SpendingScore', 'Age'])
print(df_scaled.head())

# Zastosowanie k-means clustering z k=3
kmeans = KMeans(n_clusters=3, random_state=42)
df_scaled['KMeans_Cluster'] = kmeans.fit_predict(df_scaled)

# Wyświetlenie przypisanych klastrów
print(df_scaled.head())

# Zastosowanie DBSCAN z predefiniowanymi parametrami
dbscan = DBSCAN(eps=0.7, min_samples=2)
df_scaled['DBSCAN_Cluster'] = dbscan.fit_predict(df_scaled)

# Wyświetlenie przypisanych klastrów oraz punktów odstających (-1)
print(df_scaled.head())

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Przekształcenie wyników PCA z powrotem do DataFrame dla łatwiejszej obsługi
df_pca = pd.DataFrame(df_pca, columns=['PCA1', 'PCA2'])
print(df_pca.head())

# Wizualizacja klastrów k-means
plt.scatter(df_scaled['AnnualIncome'], df_scaled['SpendingScore'], c=df_scaled['KMeans_Cluster'], cmap='viridis')
plt.title('Klasteryzacja K-Means klientów')
plt.xlabel('Roczny dochód (w tysiącach)')
plt.ylabel('Wynik wydatków (1-100)')
plt.show()

# Wizualizacja klastrów DBSCAN
plt.scatter(df_scaled['AnnualIncome'], df_scaled['SpendingScore'], c=df_scaled['DBSCAN_Cluster'], cmap='rainbow')
plt.title('Klasteryzacja DBSCAN klientów')
plt.xlabel('Roczny dochód (w tysiącach)')
plt.ylabel('Wynik wydatków (1-100)')
plt.show()


# Wizualizacja komponentów PCA
plt.scatter(df_pca['PCA1'], df_pca['PCA2'], c=df_scaled['KMeans_Cluster'], cmap='viridis')
plt.title('PCA - Redukcja wymiarowości z K-Means')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()
