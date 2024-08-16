import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import re

# Load the CSV file
data = pd.read_csv('business_types_data.csv')

# Preprocess the 'Latest Pure Premium Rate' column
data['Latest Pure Premium Rate'] = data['Latest Pure Premium Rate'].replace('[\$,]', '', regex=True).astype(float)

# Preprocess the 'Classification Code' column
data['Classification Code'] = data['Classification Code'].apply(lambda x: re.sub(r'\((\d+)\)', r'.\1', x))
data['Classification Code'] = data['Classification Code'].astype(str)

# Combine 'Phraseology' and 'Footnote' for input text
data['text'] = data['Phraseology'] + " " + data['Footnote']

# Tokenization
max_words = 10000
max_len = 200

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(data['text'])
sequences = tokenizer.texts_to_sequences(data['text'])
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# Convert 'Classification Code' to numerical labels
labels = data['Classification Code'].astype('category').cat.codes

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Define data loader if necessary (for large datasets or batch processing)
batch_size = 32

train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

# Save tokenizer for use in LLM training and inference
import pickle
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Additional preprocessing functions if necessary
def preprocess_input(text):
    """Preprocess the input text for prediction"""
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    return padded_sequence

# Ensure compatibility with M1 MacBook Air
# If necessary, ensure that the environment is set up to use the CPU or M1 GPU correctly
# For M1 MacBook Air, using TensorFlow's built-in support for Apple Silicon
# Print out the shapes to verify
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_val.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_val.shape)