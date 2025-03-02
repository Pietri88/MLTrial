import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://en.wikipedia.org/wiki/Cloud-computing_comparison"
response = requests.get(url)

soup = BeautifulSoup(response.content, "html.parser")

print(soup.title.text)

table = soup.find('table')

rows = table.find_all('tr')

headers = [header.text.strip() for header in rows[0].find_all('th')]

data = []
for row in rows[1:]:
    cols = row.find_all('td')
    cols = [col.text.strip() for col in cols]
    data.append(cols)

df = pd.DataFrame(data, columns=headers)

print(df.head())

df.to_csv('scraped_data.csv', index=False)