from ExpandedCSVScraper import *
from DataVectorizer import *



def scrape_data():
    # Base URL and output file
    base_url = "https://www.wcirb.com/products-and-services/classification-search"
    output_file = 'business_types_data.csv'
    # Scrape the data
    scrape_base_url(base_url, output_file)

    print("Data scraped and saved to business_types_data.csv")

def vectorize():
    input_file = 'business_types_data.csv'
    output_features_file = 'features.npy'
    output_labels_file = 'labels.npy'
    
    # Load and preprocess data
    texts, labels = load_and_preprocess_data(input_file)
    
    # Vectorize text
    X = vectorize_text(texts)
    
    # Encode labels
    y = encode_labels(labels)
    
    # Save processed data
    save_data(X, y, output_features_file, output_labels_file)
    
    print("Data has been preprocessed and saved.")


if __name__ == "__main__":
    response = input("Do you want to scrape the data? (y/n): ")
    if response.lower() == 'y':
        scrape_data()

    response = input("Do you want to vectorize the data? (y/n): ")
    if response.lower() == 'y':
        vectorize()
    
