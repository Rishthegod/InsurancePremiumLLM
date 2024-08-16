import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from keras import layers, Model
from keras.layers import Bidirectional, LSTM, Dense, Dropout, Input, Embedding

# Load and preprocess the data
df = pd.read_csv('business_types_data.csv')

# Tokenize the Phraseology and Footnote columns
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Phraseology'] + " " + df['Footnote'])
sequences = tokenizer.texts_to_sequences(df['Phraseology'] + " " + df['Footnote'])
word_index = tokenizer.word_index

# Pad sequences to ensure uniform input size
max_seq_length = max([len(seq) for seq in sequences])
data = pad_sequences(sequences, maxlen=max_seq_length)

# Prepare the target variables as strings
df['Classification Code'] = df['Classification Code'].astype(str)
df['Latest Pure Premium Rate'] = df['Latest Pure Premium Rate'].astype(str)
df['Pure Premium Rate Effective Date'] = df['Pure Premium Rate Effective Date'].astype(str)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    data, 
    df[['Classification Code', 'Latest Pure Premium Rate', 'Pure Premium Rate Effective Date']], 
    test_size=0.2, 
    random_state=42
)

# Define the model
input_layer = Input(shape=(max_seq_length,))
embedding_layer = Embedding(len(word_index) + 1, 128)(input_layer)
bi_lstm_layer = Bidirectional(LSTM(64))(embedding_layer)
dense_layer = Dense(64, activation='relu')(bi_lstm_layer)
dropout_layer = Dropout(0.5)(dense_layer)

classification_code_output = Dense(len(df['Classification Code'].unique()), activation='softmax', name='classification_code')(dropout_layer)
pure_premium_rate_output = Dense(len(df['Latest Pure Premium Rate'].unique()), activation='softmax', name='pure_premium_rate')(dropout_layer)
pure_premium_rate_effective_date_output = Dense(len(df['Pure Premium Rate Effective Date'].unique()), activation='softmax', name='pure_premium_rate_effective_date')(dropout_layer)

model = Model(inputs=input_layer, outputs=[classification_code_output, pure_premium_rate_output, pure_premium_rate_effective_date_output])
model.compile(
    optimizer='adam', 
    loss={
        'classification_code': 'sparse_categorical_crossentropy', 
        'pure_premium_rate': 'sparse_categorical_crossentropy', 
        'pure_premium_rate_effective_date': 'sparse_categorical_crossentropy'
    }, 
    metrics=['accuracy']
)

# Convert target variables to integer labels
class_code_map = {code: idx for idx, code in enumerate(df['Classification Code'].unique())}
rate_map = {rate: idx for idx, rate in enumerate(df['Latest Pure Premium Rate'].unique())}
date_map = {date: idx for idx, date in enumerate(df['Pure Premium Rate Effective Date'].unique())}

y_train_class = y_train['Classification Code'].map(class_code_map).values
y_train_rate = y_train['Latest Pure Premium Rate'].map(rate_map).values
y_train_date = y_train['Pure Premium Rate Effective Date'].map(date_map).values

# Train the model
model.fit(
    X_train, 
    {
        'classification_code': y_train_class, 
        'pure_premium_rate': y_train_rate, 
        'pure_premium_rate_effective_date': y_train_date
    }, 
    epochs=10, 
    batch_size=32, 
    validation_split=0.2
)

# Save the model
model.save('insurance_premium_llm.h5')

# To use the model for inference, you will need to reverse the mappings
class_code_reverse_map = {v: k for k, v in class_code_map.items()}
rate_reverse_map = {v: k for k, v in rate_map.items()}
date_reverse_map = {v: k for k, v in date_map.items()}
