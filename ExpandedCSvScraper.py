import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Function to extract keywords from text
def extract_keywords(text):
    if text.strip() == "":
        return "N/A"
    vectorizer = CountVectorizer(max_features=5, stop_words='english')
    X = vectorizer.fit_transform([text])
    keywords = vectorizer.get_feature_names_out()
    return ', '.join(keywords)

# Function to scrape individual business type page
def scrape_business_type_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract business type description
    description_div = soup.find('div', class_='field-phraseology-wrapper')
    description = description_div.get_text(strip=True) if description_div else "N/A"
    
    # Extract latest pure premium rate
    rate_div = soup.find('div', class_='field field--name-field-classification-code field--type-entity-reference field--label-hidden field__item')
    rate = rate_div.get_text(strip=True) if rate_div else "N/A"
    
    # Extract footnotes and keywords
    footnotes = soup.find_all('div', class_='footnote-wrapper')
    footnotes_text = ' '.join([footnote.get_text(strip=True) for footnote in footnotes])
    keywords = extract_keywords(footnotes_text)
    
    return description, rate, footnotes_text, keywords

# Function to scrape the base URL
def scrape_base_url(base_url):
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract links to business type pages
    business_type_links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if 'classification-search' in href and not href.endswith('classification-search'):
            business_type_links.append(f"https://www.wcirb.com{href}")
    
    data = []
    for link in business_type_links:
        try:
            description, rate, footnotes_text, keywords = scrape_business_type_page(link)
            data.append({
                'Business Type': description,
                'Latest Pure Premium Rate': rate,
                'Footnotes': footnotes_text,
                'Keywords': keywords
            })
        except Exception as e:
            print(f"Failed to scrape {link}: {e}")
    
    return pd.DataFrame(data)

# Base URL
base_url = "https://www.wcirb.com/products-and-services/classification-search"

# Scrape the data
df = scrape_base_url(base_url)

# Save to CSV
df.to_csv('business_types_data.csv', index=False)
print("Data scraped and saved to business_types_data.csv")
