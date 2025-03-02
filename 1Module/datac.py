import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import missingno as msno  # Import missingno
from scipy import stats

def load_data(filepath):
    """Wczytuje dane z pliku CSV."""
    df = pd.read_csv(filepath)
    print(df.head())
    print(df.dtypes)
    msno.matrix(df)  # Wizualizacja braków danych
    msno.heatmap(df)
    return df

def handle_missing_values(df):
    """Wypełnia brakujące dane: średnią dla numerycznych, modą dla kategorycznych."""
    df_filled = df.copy()
    numeric_cols = df_filled.select_dtypes(include=np.number).columns
    df_filled[numeric_cols] = df_filled[numeric_cols].fillna(df_filled[numeric_cols].mean())

    for col in df_filled.columns:
        if df_filled[col].dtype == 'object':
            df_filled[col] = df_filled[col].fillna(df_filled[col].mode()[0])
    return df_filled

def handle_outliers(df):
    """Wykrywa i usuwa outliery (Z-score > 3)."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df_no_outliers = df[(z_scores < 3).all(axis=1)]
    return df_no_outliers

def handle_launched_column(df):
    """Przetwarza kolumnę 'Launched', konwertując na typ numeryczny i stosując capping."""
    if 'Launched' in df.columns:
        try:
            df['Launched'] = pd.to_numeric(df['Launched'], errors='coerce')
            df['Launched'] = df['Launched'].fillna(df['Launched'].mean())

            if pd.api.types.is_numeric_dtype(df['Launched']):
                upper_limit = df['Launched'].quantile(0.95)
                df['Launched_Capped'] = np.where(df['Launched'] > upper_limit, upper_limit, df['Launched'])
        except ValueError:
            print("Kolumna 'Launched' nie może być przekonwertowana na typ numeryczny.")
            df['Launched_Capped'] = df['Launched'] #dodanie Launched_Capped nawet jesli konwersja sie nie powiedzie.

    else:
        print("Kolumna 'Launched' nie istnieje.")
        df['Launched_Capped'] = None  # Or some other default value
    return df


def cap_numeric_columns(df):
    """Stosuje capping (5. i 95. percentyl) do kolumn numerycznych (oprócz 'Launched')."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if col != 'Launched' and col != 'Launched_Capped':  # Exclude 'Launched' and 'Launched_Capped'
            upper_limit = df[col].quantile(0.95)
            lower_limit = df[col].quantile(0.05)
            df[col + '_Capped'] = np.where(df[col] > upper_limit, upper_limit,
                                            np.where(df[col] < lower_limit, lower_limit, df[col]))
    return df

def scale_data(df, method='minmax'):
    """Skaluje dane numeryczne (MinMaxScaler lub StandardScaler).

    Args:
        df: DataFrame do przeskalowania.
        method: 'minmax' (domyślnie) lub 'standard'.

    Returns:
        DataFrame: Przeskalowany DataFrame.  Zwraca pusty DataFrame, jeśli nie ma nic do skalowania.
    """

    numeric_cols_capped = [col for col in df.columns if '_Capped' in col or col == "Launched_Capped"]
    if 'Launched_Capped' in df.columns and 'Launched' in numeric_cols_capped:
      numeric_cols_capped.remove("Launched")
    if not numeric_cols_capped:
        print(f"Brak kolumn numerycznych do skalowania metodą {method}.")
        return pd.DataFrame(index=df.index)

    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Nieobsługiwana metoda skalowania.  Wybierz 'minmax' lub 'standard'.")

    df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols_capped]),
                                columns=numeric_cols_capped, index=df.index)
    return df_scaled



def encode_categorical_variables(df, scaled_df):
    """Koduje zmienne kategoryczne (One-Hot Encoding)."""
    
    numeric_cols_capped = [col for col in df.columns if '_Capped' in col or col == "Launched_Capped"]
    if 'Launched_Capped' in df.columns and 'Launched' in numeric_cols_capped:
       numeric_cols_capped.remove("Launched")
    
    df_for_encoding = df.drop(columns=numeric_cols_capped, errors='ignore').join(scaled_df)

    df_encoded = pd.get_dummies(df_for_encoding)

    return df_encoded


def save_processed_data(df, filepath):
    """Zapisuje przetworzone dane do pliku CSV."""
    df.to_csv(filepath, index=False)
    print(f'Data cleaning and preprocessing complete. File saved as {filepath}')


def preprocess_data(input_filepath, output_filepath_filled, output_filepath_no_outliers):
    """Główna funkcja do przetwarzania danych.

    Args:
        input_filepath: Ścieżka do pliku CSV z danymi wejściowymi.
        output_filepath_filled: Ścieżka do pliku CSV, gdzie zostaną zapisane dane z wypełnionymi brakami i cappingiem.
        output_filepath_no_outliers: Ścieżka do pliku CSV, gdzie zostaną zapisane dane z usuniętymi outlierami.
    """

    # 1. Wczytaj dane
    df = load_data(input_filepath)

    # 2. Wypełnij brakujące dane
    df_filled = handle_missing_values(df)

    # 3. Obsłuż outliery (usuwanie)
    df_no_outliers = handle_outliers(df_filled.copy()) # Pracuj na kopii

    # 4. Obsłuż kolumnę 'Launched' (df_filled)
    df_filled = handle_launched_column(df_filled)

    # 5. Obsłuż kolumnę 'Launched (df_no_outliers)
    df_no_outliers = handle_launched_column(df_no_outliers)


    # 6. Capping pozostałych kolumn numerycznych (df_filled)
    df_filled = cap_numeric_columns(df_filled)


    #7. Skalowanie danych - df_filled (minmax i standard)
    df_filled_scaled_minmax = scale_data(df_filled, method='minmax')
    df_filled_scaled_standard = scale_data(df_filled, method='standard')

    #8. Skalowanie danych - df_no_outliers (minmax i standard)
    df_no_outliers_scaled_minmax = scale_data(df_no_outliers, method='minmax')
    df_no_outliers_scaled_standard = scale_data(df_no_outliers, method='standard')

    #9. Kodowanie zmiennych kategorycznych (df_filled)
    df_filled_encoded = encode_categorical_variables(df_filled, df_filled_scaled_minmax) #uzywam wersji minmax do enkodowania

    # 10. Kodowanie zmiennych kategorycznych (df_no_outliers)
    df_no_outliers_encoded = encode_categorical_variables(df_no_outliers, df_no_outliers_scaled_minmax)

    # 11. Zapisz przetworzone dane
    save_processed_data(df_filled_encoded, output_filepath_filled)
    save_processed_data(df_no_outliers_encoded, output_filepath_no_outliers)



# Przykładowe użycie:
if __name__ == "__main__":
    preprocess_data('scraped_data.csv', 'cleaned_preprocessed_filled.csv', 'cleaned_preprocessed_no_outliers.csv')