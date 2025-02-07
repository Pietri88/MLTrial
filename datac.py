import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import missingno as msno
from scipy import stats

df = pd.read_csv('scraped_data.csv')

print(df.head())
print(df.dtypes)  # Sprawdź typy danych (usunięto nawiasy)

msno.matrix(df)
msno.heatmap(df)  # Te wizualizacje możesz zostawić przed czyszczeniem, aby zobaczyć braki

# 1. Wypełnij brakujące dane NAJPIERW (poprawnie)
numeric_cols = df.select_dtypes(include=np.number).columns
df_filled = df.copy()  # Pracuj na kopii, aby nie modyfikować oryginału
df_filled[numeric_cols] = df_filled[numeric_cols].fillna(df_filled[numeric_cols].mean())

# Przykład obsługi kolumn nienumerycznych (jeśli są)
for col in df.columns:
    if df[col].dtype == 'object':  # Sprawdź, czy kolumna jest tekstowa
        df_filled[col] = df_filled[col].fillna(df_filled[col].mode()[0])  # Wypełnij modą

# 2. Wykryj outliery na WYPEŁNIONYCH danych (df_filled)
#    Oblicz Z-score tylko dla kolumn numerycznych
z_scores = np.abs(stats.zscore(df_filled[numeric_cols]))

#    Stwórz DataFrame bez outlierów (opcjonalnie, możesz też zastąpić outliery)
df_no_outliers = df_filled[(z_scores < 3).all(axis=1)]

# 3. Obsługa górnych wartości w 'Launched' (jeśli kolumna istnieje i jest numeryczna)
if 'Launched' in df_filled.columns and pd.api.types.is_numeric_dtype(df_filled['Launched']):
    upper_limit = df_filled['Launched'].quantile(0.95)
    # Stwórz NOWĄ kolumnę, aby nie nadpisywać oryginalnej
    df_filled['Launched_Capped'] = np.where(df_filled['Launched'] > upper_limit, upper_limit, df_filled['Launched'])

    #Alternatywa, jesli chcesz to zrobic na df_no_outliers, i ta kolumna jest numeryczna
    if 'Launched' in df_no_outliers.columns and pd.api.types.is_numeric_dtype(df_no_outliers['Launched']):
      upper_limit_no = df_no_outliers['Launched'].quantile(0.95)
      df_no_outliers['Launched_Capped'] = np.where(df_no_outliers['Launched'] > upper_limit_no, upper_limit_no, df_no_outliers['Launched'])
    
else:
    print("Kolumna 'Launched' nie istnieje lub nie jest numeryczna.")


# Teraz masz:
# - df: oryginalny DataFrame
# - df_filled: DataFrame z wypełnionymi brakującymi danymi
# - df_no_outliers: DataFrame z usuniętymi outlierami (na podstawie Z-score)
# - df_filled (i/lub df_no_outliers): z dodatkową kolumną 'Launched_Capped', jeśli 'Launched' istniała i była numeryczna
print(df_filled.head()) #sprawdz
print(df_no_outliers.head()) #sprawdz



# --- Dodanie kolejnych kroków z instrukcji ---

# Step 4: Handle outliers (już częściowo zrobione, ale rozszerzamy)
# Outliery już wykryliśmy i usunęliśmy w df_no_outliers.
# Teraz dodajemy alternatywne podejście: capping (ograniczanie wartości)
# Zastosujemy capping na df_filled, dla każdej kolumny numerycznej

for col in numeric_cols:
    if col != 'Launched':  # Launched już obsłużyliśmy
        upper_limit = df_filled[col].quantile(0.95)
        lower_limit = df_filled[col].quantile(0.05)
        df_filled[col + '_Capped'] = np.where(df_filled[col] > upper_limit, upper_limit,
                                            np.where(df_filled[col] < lower_limit, lower_limit, df_filled[col]))


# Step 5: Scale and normalize data
# Wykorzystamy MinMaxScaler (do zakresu [0, 1]) i StandardScaler (do średniej 0 i odch. stand. 1)
# Zastosujemy skalowanie do df_filled (po cappingu) ORAZ do df_no_outliers

# Min-Max Scaling (df_filled)
scaler_minmax_filled = MinMaxScaler()
numeric_cols_filled_capped = [col for col in df_filled.columns if '_Capped' in col or col == 'Launched_Capped']
df_filled_scaled = pd.DataFrame(scaler_minmax_filled.fit_transform(df_filled[numeric_cols_filled_capped]),
                                columns=numeric_cols_filled_capped, index=df_filled.index)

# Z-score Standardization (df_filled)
scaler_standard_filled = StandardScaler()
df_filled_standardized = pd.DataFrame(scaler_standard_filled.fit_transform(df_filled[numeric_cols_filled_capped]),
                                      columns=numeric_cols_filled_capped, index = df_filled.index)


# Min-Max Scaling (df_no_outliers - wersja z usuniętymi outlierami)
if 'Launched_Capped' in df_no_outliers.columns:
    cols_to_scale_no =  df_no_outliers.select_dtypes(include=np.number).columns.tolist()
else:
   cols_to_scale_no = [col for col in df_no_outliers.select_dtypes(include=np.number).columns if col != "Launched"]

scaler_minmax_no = MinMaxScaler()

df_no_outliers_scaled = pd.DataFrame(scaler_minmax_no.fit_transform(df_no_outliers[cols_to_scale_no]),
                                    columns=cols_to_scale_no, index = df_no_outliers.index)


# Z-score Standardization (df_no_outliers)
scaler_standard_no = StandardScaler()
df_no_outliers_standardized = pd.DataFrame(scaler_standard_no.fit_transform(df_no_outliers[cols_to_scale_no]),
                                            columns=cols_to_scale_no, index = df_no_outliers.index)




# Step 6: Encode categorical variables
# Kodowanie zmiennych kategorycznych (One-Hot Encoding)

# Najpierw połączmy przeskalowane dane numeryczne z oryginalnymi danymi tekstowymi, aby mieć wszystko w jednym DataFrame

# Dla df_filled:
df_filled_for_encoding = df_filled.drop(columns=numeric_cols_filled_capped).join(df_filled_scaled)
df_filled_encoded = pd.get_dummies(df_filled_for_encoding)  #domyślnie koduje wszystkie kolumny object

# Dla df_no_outliers:
df_no_outliers_for_encoding = df_no_outliers.drop(columns=cols_to_scale_no).join(df_no_outliers_scaled)
df_no_outliers_encoded = pd.get_dummies(df_no_outliers_for_encoding)


# Step 7: Save the cleaned and preprocessed data
# Zapisanie przetworzonych danych do plików CSV

df_filled_encoded.to_csv('cleaned_preprocessed_filled.csv', index=False)
df_no_outliers_encoded.to_csv('cleaned_preprocessed_no_outliers.csv', index=False)

print('Data cleaning and preprocessing complete. Files saved.')