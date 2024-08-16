import tensorflow as tf
from tensorflow.python.keras import layers, Model
from keras.layers import Bidirectional, LSTM, Dense, Dropout, Input, Embedding

# Define the model architecture
def create_llm_model(vocab_size, embedding_dim, input_length, num_classes):
    model = tf.keras.Sequential([
        Embedding(vocab_size, embedding_dim, input_length=input_length),
        Bidirectional(LSTM(128, return_sequences=True)),
        Bidirectional(LSTM(64)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    # Parameters (these should match those used in the preprocessing script)
    vocab_size = 10000  # Max number of words in the tokenizer
    embedding_dim = 128  # Embedding dimensions for each token
    input_length = 200  # Max length of input sequences
    num_classes = 691  # Number of classification codes

    # Create the model
    model = create_llm_model(vocab_size, embedding_dim, input_length, num_classes)

    # Print the model summary
    model.summary()

    # Save the model architecture to a file
    model.save('llm_model.h5')
