import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time



# Function to extract keywords from text, focusing on nouns and proper nouns
def extractKeywords(text):
    """ 
    Extract keywords from the text using TF-IDF vectorization.
    
    Parameters:
        - text (str): The text from which to extract keywords.
    Returns:
        - str: The top 10 keywords separated by commas.
    """
    if text.strip() == "":
        return "N/A"
    vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
    X = vectorizer.fit_transform([text])
    keywords = vectorizer.get_feature_names_out()
    return ', '.join(keywords)
   

# Function to scrape individual business type page
def scrapeBusinessType(url, session):
    """
    Scrape the individual business type page for the description, latest pure premium rate, footnotes, and keywords.
    
    Parameters:
        - url (str): The URL of the business type page.
        - session (requests.Session): The requests session object.
    Returns:
        - tuple: A tuple containing the description, rate, footnotes, and keywords.
    """
    response = session.get(url)
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
    keywords = extractKeywords(description + " " + footnotes_text)
    
    return description, rate, footnotes_text, keywords

# Function to scrape the base URL
def scrape_base_url(base_url, output_file):
    """
    Scrape the base URL for all of the business types and save the data to a CSV file.
    
    
    Parameters:
        - base_url (str): The base URL to scrape.
        - output_file (str): The name of the output CSV file.
    Output:
        - None (saves the data to a CSV file)
    """
    #init data array
    data = []
    session = requests.Session()
    retry = Retry(connect=5, backoff_factor=1)
    '''sometimes the connection times out, so we have to retry'''
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    for page in range(58):  # There are 58 pages starting from 0 index (for now)
        """The page number is appended to the url to get the next page of results.
        The page number is a string of commas, so the first page is ",,,0", the second page is ",,,1", and so on.
        """
        try:
            page_url = f"{base_url}?page=%2C%2C%2C{page}"
            response = session.get(page_url, timeout=20)
            if response.status_code != 200:
                print(f"Failed to retrieve page {page + 1}. Status code: {response.status_code}")
                continue
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract links to business type pages
            business_type_links = [f"https://www.wcirb.com{a['href']}" for a in soup.find_all('a', href=True) if 'classification-search' in a['href'] and not a['href'].endswith('classification-search')]
            
            if not business_type_links:
                print(f"No business type links found on page {page + 1}.")
                continue
            
            for link in business_type_links:
                try:
                    """Scrape each business type page for the Phreaseology, Latest Pure Premium Rate, Footnote, and keywords."""

                    description, rate, footnotes_text, keywords = scrapeBusinessType(link, session)
                    data.append({
                        'Phraseology': description,
                        'Latest Pure Premium Rate': rate,
                        'Footnote': footnotes_text,
                        'Keywords': keywords
                    })
                except Exception as e:
                    print(f"Failed to scrape {link}: {e}")
            
            print(f"Scraped page {page + 1}")
            time.sleep(1)  # Avoid getting banned by the website (to bypass the rate limiting)

            # Save data after every scrape so even in case of an error or something, the data is not all lost. 
            if data:
                SaveToCSV(data, output_file)
                data = []  # Clear data after saving
        except requests.exceptions.RequestException as e:
            print(f"Error while accessing page {page + 1}: {e}")
            break

    # Final save for any remaining data
    if data:
        SaveToCSV(data, output_file)

# Function to save data to CSV
def SaveToCSV(data, output_file):
    """
    Function to save the data to a CSV file.
    
    Parameters:
        - data (list): The list of dictionaries containing the data.
        - output_file (str): The name of the output CSV file.
    Output:
        - None (saves the data to a CSV file)"""
    df = pd.DataFrame(data)
    df = df.dropna(how='all')  # Remove rows with all N/A values
    df = df.replace("N/A", "")  # Replace "N/A" with empty strings
    df.to_csv(output_file, mode='a', header=not pd.io.common.file_exists(output_file), index=False)

# Base URL and output file
base_url = "https://www.wcirb.com/products-and-services/classification-search"
output_file = 'business_types_data.csv'

# Scrape the data
scrape_base_url(base_url, output_file)

print("Data scraped and saved to business_types_data.csv")





"""
    # Future Possible Implementation using nltk tokenizing to have better keywords:
    #currently running this code gives worse keyword selection than the TF-IDF vectorization (in my opinion)

    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag
    import string

    # Download nltk data
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')

    if text.strip() == "":
        return "N/A"
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    keywords = [word for word, pos in tagged_words if pos in ('NN', 'NNS', 'NNP', 'NNPS') and word.lower() not in stopwords.words('english') and word not in string.punctuation]
    return ', '.join(keywords[:10])  # Limit to top 10 keywords"""