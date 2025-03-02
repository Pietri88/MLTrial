import requests
from bs4 import BeautifulSoup
import pandas as pd
import time 

url = 'https://en.wikipedia.org/wiki/Cloud-computing_comparison'
try:
    response = requests.get(url)
    response.raise_for_status()  # Raises an HTTPError for bad responses
except requests.exceptions.HTTPError as err:
    print('HTTP error occurred:', err)
except Exception as err:
    print('Other error occurred:', err)

soup = BeautifulSoup(response.content, 'html.parser')

print('Webpage Title:', soup.title.text)

# Locate the table that contains the product data
table = soup.find('table', {'class': 'wikitable'})  # Replace with the actual id or class name

# Extract the rows of the table
rows = table.find_all('tr')

# Initialize an empty list to store the data
data = []

for row in rows[1:]:
    cols = row.find_all('td')
    if len(cols) == 3:  # Ensure all three columns are present
        product_name = cols[0].text.strip() if cols[0] else 'N/A'
        price = cols[1].text.strip() if cols[1] else 'N/A'
        rating = cols[2].text.strip() if cols[2] else 'N/A'
        data.append([product_name, price, rating])
    else:
        print('Skipping a row with missing data.')

# Convert the list to a pandas DataFrame
df = pd.DataFrame(data, columns=['Product Name', 'Price', 'Rating'])

time.sleep(2)  # Adds a 2-second delay before the next request

# Save the DataFrame to a CSV file
df.to_csv('scraped_products.csv', index=False)

print('Data successfully saved to scraped_products.csv')