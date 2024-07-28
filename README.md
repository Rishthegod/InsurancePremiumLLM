<h1>BTIS WCIRB Scraper for LLM Training</h1>

Repo Contents:
- ExpandedCSvScraper.py
  - Main Python program for scraping.
  - Scrapes all 58 pages from the WCIRB website and accesses the sub-links for all 691 business types
  - Scrapes the Phraseology, Latest Pure Premium Rate, and Footnote for each Business Type
  - From the Footnote, uses TF-IDF vectorization to extract keywords from Footnote text, useful for training an LLM later.
- business_types_data.csv
  - Output of the main scraper: ExpandedCSvScraper.py, as a CSV file.
  - TODO: import into MongoDB, potentially more data cleaning later.
- DairyfarmScraper.py
  - Scrapes only the Dairy-farm: base code for scraping a subpage from the base URL 
  - Initial Intern assignment: scrape only the dairy farm
- wcirb_dairy_farms.csv
  - Output of DairyfarmScraper.py
