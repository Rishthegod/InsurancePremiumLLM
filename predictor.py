import tensorflow as tf
from tensorflow.python.keras import layers, Model
from keras.layers import Bidirectional, LSTM, Dense, Dropout, Input, Embedding, GlobalMaxPooling1D
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Load the CSV file
data = pd.read_csv('business_types_data.csv')

# Preprocess the 'Latest Pure Premium Rate' column
data['Latest Pure Premium Rate'] = data['Latest Pure Premium Rate'].replace('[\$,]', '', regex=True).astype(float)

# Preprocess the 'Classification Code' column
data['Classification Code'] = data['Classification Code'].apply(lambda x: re.sub(r'\((\d+)\)', r'.\1', x))
data['Classification Code'] = data['Classification Code'].astype(str)

# Combine 'Phraseology' and 'Footnote' for input text
data['text'] = data['Phraseology'] + " "+ data['Footnote']+ " " + data['Phraseology'] 

# Tokenization
max_words = 10000
max_len = 200

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

sequences = tokenizer.texts_to_sequences(data['text'])
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# Convert 'Classification Code' to numerical labels
labels = data['Classification Code'].astype('category').cat.codes

# Compute class weights to handle imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights_dict = dict(enumerate(class_weights))

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Define data loader if necessary (for large datasets or batch processing)
batch_size = 32

train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

# Create the model
vocab_size = 10000  # Max number of words in the tokenizer
embedding_dim = 128  # Embedding dimensions for each token
input_length = 200  # Max length of input sequences
num_classes = len(np.unique(labels))  # Number of classification codes

def create_llm_model(vocab_size, embedding_dim, input_length, num_classes):
    inputs = Input(shape=(input_length,))
    x = Embedding(vocab_size, embedding_dim, input_length=input_length)(inputs)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)  # Add dropout for regularization
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

model = create_llm_model(vocab_size, embedding_dim, input_length, num_classes)

# Compile the model with standard sparse categorical crossentropy
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Training the model
epochs = 10

history = model.fit(train_data,
                    validation_data=val_data,
                    epochs=epochs,
                    class_weight=class_weights_dict)  # Use class weights

# Save the trained model
model.save('trained_llm_model.h5')

# Optionally, save the training history
import json
with open('training_history.json', 'w') as f:
    json.dump(history.history, f)
