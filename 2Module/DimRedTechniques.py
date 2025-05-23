import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# Create a sample dataset with customer annual income, spending score, and age
data = {
    'AnnualIncome': [
        15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 
        20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24, 24.5, 
        25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29, 29.5, 
        30, 30.5, 31, 31.5, 32, 32.5, 33, 33.5, 34, 34.5, 
        35,   # Normal points
        80, 85, 90  # Outliers
    ],
    'SpendingScore': [
        39, 42, 45, 48, 51, 54, 57, 60, 63, 66,
        69, 72, 75, 78, 81, 84, 87, 90, 93, 96,
        6, 9, 12, 15, 18, 21, 24, 27, 30, 33,
        5, 8, 11, 14, 17, 20, 23, 26, 29, 32,
        56,   # Normal points
        2, 3, 100  # Outliers
    ],
    'Age': [
        20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24, 24.5, 
        25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29, 29.5, 
        30, 30.5, 31, 31.5, 32, 32.5, 33, 33.5, 34, 34.5, 
        35, 35.5, 36, 36.5, 37, 37.5, 38, 38.5, 39, 39.5, 
        40,   # Normal points
        15, 60, 70  # Outliers
    ]
}

df = pd.DataFrame(data)

# Display the first few rows of the dataset
print(df.head())

# Normalizujemy dane
scaler = StandardScaler()
scaled = scaler.fit_transform(df)

# Konwertujemy na DataFrame, aby łatwiej było nimi zarządzać
df_scaled = pd.DataFrame(scaled, columns=['AnnualIncome', 'SpendingScore', 'Age'])

print(df_scaled.head())

pca = PCA(n_components=2)
df_pca = pca.fit_transform(scaled)

df_pca = pd.DataFrame(df_pca, columns=['PCA1', 'PCA2'])
print(df_pca.head())

plt.scatter(df_pca['PCA1'], df_pca['PCA2'])
plt.title('PCA - Redukcja Wymiarowości')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()


tsne= TSNE(n_components=2, perplexity=3,random_state=42)
df_tsne = tsne.fit_transform(scaled)

df_tsne = pd.DataFrame(df_tsne, columns=['t-SNE1', 't-SNE2'])
# Wizualizujemy komponenty t-SNE
plt.scatter(df_tsne['t-SNE1'], df_tsne['t-SNE2'])
plt.title('t-SNE - Redukcja Wymiarowości')
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')
plt.show()
