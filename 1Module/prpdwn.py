import requests
from bs4 import BeautifulSoup
import os

url = 'https://moniaki.pl/poradnik-41-sposobow-na-oszczedzanie/'
response = requests.get(url)
response.raise_for_status()  # Sprawdź, czy żądanie HTTP się powiodło

soup = BeautifulSoup(response.content, 'html.parser')

print('Webpage Title:', soup.title.text)

article_page_div = soup.find('div', class_='article-page')
if article_page_div:
    # Znajdź wszystkie tagi 'a' (linki) wewnątrz 'article-page_div'
    document_link_tags = article_page_div.find_all('a', href=True) # Znajdujemy wszystkie tagi <a> z atrybutem href

    if document_link_tags:
        print('Document links found:')
        for link_tag in document_link_tags:
            document_url = link_tag['href'] # Pobierz adres URL dokumentu z atrybutu 'href'
            if document_url.endswith(('.pdf', '.zip')): # Sprawdź, czy URL kończy się na .pdf lub .zip (możesz dodać inne rozszerzenia)
                print(f'- {document_url}')

                try:
                    document_response = requests.get(document_url, stream=True) # Pobierz dokument, stream=True dla dużych plików
                    document_response.raise_for_status() # Sprawdź, czy pobieranie zakończyło się sukcesem

                    file_name = os.path.basename(document_url) # Wygeneruj nazwę pliku na podstawie URL
                    file_path = file_name # Możesz chcieć zapisać w innym folderze, wtedy zmień ścieżkę

                    with open(file_path, 'wb') as document_file: # Otwórz plik w trybie zapisu binarnego
                        for chunk in document_response.iter_content(chunk_size=8192): # Iteruj po pobranej treści w częściach
                            document_file.write(chunk) # Zapisz każdą część do pliku

                    print(f'  Downloaded successfully and saved as: {file_name}')

                except requests.exceptions.RequestException as e: # Obsłuż błędy pobierania
                    print(f'  Failed to download: {document_url}')
                    print(f'  Error: {e}')
            else:
                print(f'- Skipped (not a document): {document_url}') # Pomiń linki, które nie są dokumentami

    else:
        print('No document links found within article-page div.')
else:
    print('No div with class "article-page" found.')