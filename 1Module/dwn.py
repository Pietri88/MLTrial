import requests
from bs4 import BeautifulSoup
import os

# Step 1: Send an HTTP request to the webpage
url = 'https://www.sitereportpro.co.uk/sample-pdf-reports/'
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
else:
    print('Failed to retrieve the webpage.')
    exit()

# Step 2: Find all document links from the table
pdf_links = []
table = soup.find('div', class_='wpb_raw_html').find('table') # Find the table within the div with class 'wpb_raw_html'

if table:
    a_tags = table.find_all('a')
    for a_tag in a_tags:
        href = a_tag.get('href')
        if href and href.startswith('https://www.sitereportpro.co.uk/download/'): # Filter links to only download URLs
            pdf_links.append(href)
else:
    print("Table with PDF links not found.")
    exit()

# Print the document links to verify
print('Found PDF links:')
for link in pdf_links:
    print(link)

# Step 3: Download the PDFs
if pdf_links:
    for pdf_url in pdf_links:
        pdf_response = requests.get(pdf_url, stream=True)

        if pdf_response.status_code == 200:
            # Extract filename from URL or use a default name
            filename = os.path.basename(pdf_url.split('?')[0]) # Remove query parameters from filename
            if not filename:
                filename = 'sample_report.pdf'
            if not filename.endswith('.pdf'):
                filename += '.pdf' # Ensure .pdf extension

            filepath = os.path.join(os.getcwd(), filename) # Save to current directory
            with open(filepath, 'wb') as pdf_file:
                for chunk in pdf_response.iter_content(chunk_size=1024):
                    if chunk:
                        pdf_file.write(chunk)
            print(f'Downloaded PDF: {filename}')
        else:
            print(f'Failed to download PDF from: {pdf_url}')
else:
    print('No PDF links found to download.')