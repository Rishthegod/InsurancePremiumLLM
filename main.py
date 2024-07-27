from ExpandedCSVScraper import *

# Base URL and output file
base_url = "https://www.wcirb.com/products-and-services/classification-search"
output_file = 'business_types_data.csv'

# Scrape the data
scrape_base_url(base_url, output_file)

print("Data scraped and saved to business_types_data.csv")
