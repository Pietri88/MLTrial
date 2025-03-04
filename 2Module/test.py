from sklearn.cluster import KMeans
import numpy as np

# Przykładowe dane klientów (liczba zakupów, łączne wydatki, liczba kategorii)
X = np.array([[5, 1000, 2], [10, 5000, 5], [2, 500, 1], [8, 3000, 3]])

# Tworzenie i trenowanie modelu k-means
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# Wyświetlenie centrów klastrów
print(kmeans.cluster_centers_)