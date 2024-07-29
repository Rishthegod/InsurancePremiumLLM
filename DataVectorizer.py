import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

def preprocess_text(text):
    """ 
    Preprocess text by removing special characters and converting to lowercase.
    
    Parameters:
        - text (str): The text to preprocess.
    Returns:
        - str: The preprocessed text.
    """
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

def load_and_preprocess_data(input_file):
    """ 
    Load and preprocess data from CSV.
    
    Parameters:
        - input_file (str): The input CSV file.
    Returns:
        - tuple: A tuple containing the preprocessed descriptions and labels.
    """
    df = pd.read_csv(input_file)
    
    # Preprocess descriptions and footnotes
    df['Phraseology'] = df['Phraseology'].apply(preprocess_text)
    df['Footnote'] = df['Footnote'].apply(preprocess_text)
    
    # Combine Phraseology and Footnote for feature extraction
    df['combined_text'] = df['Phraseology'] + ' ' + df['Footnote']
    
    # Extract features and labels
    texts = df['combined_text'].values
    labels = df['Classification Code'].values
    
    return texts, labels

def vectorize_text(texts):
    """ 
    Vectorize text using TF-IDF.
    
    Parameters:
        - texts (list): The list of texts to vectorize.
    Returns:
        - np.ndarray: The TF-IDF vectorized text.
    """
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    return X.toarray()

def encode_labels(labels):
    """ 
    Encode labels using LabelEncoder.
    
    Parameters:
        - labels (list): The list of labels to encode.
    Returns:
        - np.ndarray: The encoded labels.
    """
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    return y

def save_data(X, y, output_features_file, output_labels_file):
    """ 
    Save data to files.
    
    Parameters:
        - X (np.ndarray): The feature matrix.
        - y (np.ndarray): The labels.
        - output_features_file (str): The output file for features.
        - output_labels_file (str): The output file for labels.
    Returns:
        - None
    """
    np.save(output_features_file, X)
    np.save(output_labels_file, y)


#if __name__ == '__main__':
 #   main()
