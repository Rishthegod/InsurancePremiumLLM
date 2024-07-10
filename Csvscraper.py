import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL to scrape
url = "https://www.wcirb.com/products-and-services/classification-search/dairy-farms"

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract the Phraseology
    phraseology = soup.find('div', class_='field-phraseology-wrapper').find('h2', class_='mb-20 label-wrapper h3 field-label-above').find_next_sibling(text=True).strip()
    
    # Extract the Footnote
    footnote_div = soup.find('div', class_='footnote-wrapper')
    footnote = ' '.join(p.text.strip() for p in footnote_div.find_all('p'))
    
    # Extract the Latest Pure Premium Rate
    latest_pure_premium_rate = soup.find('div', class_='field field--name-field-classification-code field--type-entity-reference field--label-hidden field__item').find('div', class_='field-content').text.strip()
    
    # Create a DataFrame
    data = {
        'Phraseology': [phraseology],
        'Footnote': [footnote],
        'Latest Pure Premium Rate': [latest_pure_premium_rate]
    }
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv('wcirb_dairy_farms.csv', index=False)
    
    print("Data scraped and saved to wcirb_dairy_farms.csv")
else:
    print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
