import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from transformers import BertTokenizer

# Load the dataset
file_path = 'business_types_data.csv'  # Update with the actual file path
df = pd.read_csv(file_path)

# 1. Preprocessing the 'Latest Pure Premium Rate'
def preprocess_premium_rate(rate_str):
    """Remove the dollar sign and convert to float."""
    if isinstance(rate_str, str):
        return float(rate_str.replace('$', ''))
    return rate_str

df['Latest Pure Premium Rate'] = df['Latest Pure Premium Rate'].apply(preprocess_premium_rate)

# 2. Handling the 'Classification Code'
def normalize_classification_code(code):
    """Normalize classification codes by splitting out parenthetical subsections."""
    main_code = re.findall(r'\d+', code)[0]  # Extract the main number part
    subsection = re.findall(r'\((\d+)\)', code)
    if subsection:
        return (int(main_code), int(subsection[0]))
    return (int(main_code),)

df['Classification Code'] = df['Classification Code'].apply(normalize_classification_code)

# Label encode the classification codes
label_encoder = LabelEncoder()
df['Classification Code Encoded'] = label_encoder.fit_transform(df['Classification Code'].astype(str))

# 3. Text Tokenization for BERT/RoBERTa
# Combine Phraseology and Footnote for tokenization
text_data = df['Phraseology'] + " " + df['Footnote']

# Load BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the text data
bert_tokenized = text_data.apply(lambda x: bert_tokenizer.encode_plus(
    x,
    add_special_tokens=True,
    max_length=512,
    truncation=True,
    padding='max_length',
    return_attention_mask=True,
    return_tensors='tf'
))

# Extract input ids and attention masks
input_ids = [tokenized['input_ids'].numpy()[0] for tokenized in bert_tokenized]
attention_masks = [tokenized['attention_mask'].numpy()[0] for tokenized in bert_tokenized]

# Add tokenized data to the DataFrame
df['Input IDs'] = input_ids
df['Attention Masks'] = attention_masks

# 4. Feature Engineering for Traditional Models
# Example: Using TF-IDF on the combined text data
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_features = tfidf_vectorizer.fit_transform(text_data).toarray()

# Add TF-IDF features to the DataFrame (as an example, you might store it separately)
df_tfidf = pd.DataFrame(tfidf_features, columns=tfidf_vectorizer.get_feature_names_out())
df_tfidf['Classification Code Encoded'] = df['Classification Code Encoded']

# Save the preprocessed data
df.to_csv('preprocessed_business_types_data.csv', index=False)
df_tfidf.to_csv('tfidf_features.csv', index=False)

# Save the label encoder and BERT tokenizer for use in model training
import pickle
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Save the BERT tokenizer settings
bert_tokenizer.save_pretrained('bert_tokenizer')

# Save TF-IDF vectorizer for use with traditional models
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
