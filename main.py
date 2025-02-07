# main.py
import datac  # Lub import datac as dt, jeśli używasz nazwy datac.py
import pandas as pd
import numpy as np

# 1. Generowanie przykładowych danych (zgodnie z instrukcją)
np.random.seed(0)
dummy_data = {
    'Feature1': np.random.normal(100, 10, 100).tolist() + [np.nan, 200],
    'Feature2': np.random.randint(0, 100, 102).tolist(),
    'Category': ['A', 'B', 'C', 'D'] * 25 + [np.nan, 'A'],
    'Target': np.random.choice([0, 1], 102).tolist()
}
df_dummy = pd.DataFrame(dummy_data)

# Zapisz przykładowe dane do pliku CSV (aby funkcja load_data mogła je wczytać)
df_dummy.to_csv('dummy_data.csv', index=False)


# 2. i 3.  Użycie funkcji preprocessingu
if __name__ == "__main__":
    input_filepath = 'dummy_data.csv'          # Ścieżka do pliku z danymi wejściowymi
    output_filepath_filled = 'preprocessed_dummy_data_filled.csv'    # Ścieżka do pliku wyjściowego (wypełnione braki)
    output_filepath_no_outliers = 'preprocessed_dummy_data_no_outliers.csv'  # Ścieżka do pliku wyjściowego (bez outlierów)

    # Wywołaj główną funkcję preprocessingu
    datac.preprocess_data(input_filepath, output_filepath_filled, output_filepath_no_outliers)

    # 4. Weryfikacja (opcjonalnie, ale ZALECA się to zrobić)
    # Wczytaj przetworzone dane z plików CSV
    df_filled = pd.read_csv(output_filepath_filled)
    df_no_outliers = pd.read_csv(output_filepath_no_outliers)


    # Sprawdź brakujące wartości
    print("\nBrakujące wartości w df_filled:\n", df_filled.isnull().sum())
    print("\nBrakujące wartości w df_no_outliers:\n", df_no_outliers.isnull().sum())

    # Podsumowanie statystyczne
    print("\nStatystyki df_filled:\n", df_filled.describe())
    print("\nStatystyki df_no_outliers:\n", df_no_outliers.describe())

    # Wyświetl pierwsze kilka wierszy
    print("\nPierwsze wiersze df_filled:\n", df_filled.head())
    print("\nPierwsze wiersze df_no_outliers:\n", df_no_outliers.head())

    # Sprawdź nazwy kolumn (aby zobaczyć, jak zostały zakodowane zmienne kategoryczne)
    print("\nKolumny df_filled:\n", df_filled.columns)
    print("\nKolumny df_no_outliers:\n", df_no_outliers.columns)