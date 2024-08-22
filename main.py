from webScraper.ExpandedCSVScraper import *
from inference import *





def scrape_data():
    # Base URL and output file
    base_url = "https://www.wcirb.com/products-and-services/classification-search"
    output_file = 'business_types_data.csv'
    # Scrape the data
    scrape_base_url(base_url, output_file)

    print("Data scraped and saved to business_types_data.csv")



if __name__ == "__main__":
    response = input("Do you want to scrape the data? (y/n): ")
    if response.lower() == 'y':
        scrape_data()

    response = input("Enter your input text: ")

    llm = InsurancePremiumLLM()
    
    output = llm.predict(response)
    print("Model Output:", output)

    
