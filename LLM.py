import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load the preprocessed data
df = pd.read_csv('preprocessed_business_types_data.csv')

# Initialize Label Encoder
label_encoder = LabelEncoder()
df['Classification Code Encoded'] = label_encoder.fit_transform(df['Classification Code'])

# Load the BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize and pad the input text for BERT
X_bert_input = df['Phraseology'] + ' ' + df['Footnote']
bert_tokens = bert_tokenizer(
    X_bert_input.tolist(),
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='tf'
)
X_bert_input_ids = bert_tokens['input_ids'].numpy()  # Convert to NumPy arrays
X_bert_attention_masks = bert_tokens['attention_mask'].numpy()  # Convert to NumPy arrays

# Prepare the labels
y = df['Classification Code Encoded'].values

# Split the data into training and testing sets
X_train_ids, X_test_ids, X_train_masks, X_test_masks, y_train, y_test = train_test_split(
    X_bert_input_ids, X_bert_attention_masks, y, test_size=0.2, random_state=42
)

# Define the BERT model for feature extraction
def create_bert_features(input_ids, attention_masks):
    bert_outputs = bert_model(input_ids, attention_mask=attention_masks)
    return bert_outputs.last_hidden_state[:, 0, :]  # Use [CLS] token output

# Initialize BERT model
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Extract BERT features
X_train_bert_features = create_bert_features(X_train_ids, X_train_masks).numpy()  # Convert to NumPy arrays
X_test_bert_features = create_bert_features(X_test_ids, X_test_masks).numpy()  # Convert to NumPy arrays

# Create TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf_vectorizer.fit_transform(X_bert_input)
X_train_tfidf = X_tfidf[:X_train_ids.shape[0]].toarray()
X_test_tfidf = X_tfidf[X_train_ids.shape[0]:].toarray()

# Combine BERT and TF-IDF features
X_train_combined = np.concatenate([X_train_bert_features, X_train_tfidf], axis=1)
X_test_combined = np.concatenate([X_test_bert_features, X_test_tfidf], axis=1)

# Initialize the RandomForest classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the RandomForest classifier on the combined features
rf_clf.fit(X_train_combined, y_train)

# Evaluate the RandomForest classifier
y_pred_rf = rf_clf.predict(X_test_combined)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {rf_accuracy:.4f}')

# Generate the classification report
print(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))

# BERT Classifier model definition
class BERTClassifier(tf.keras.Model):
    def __init__(self, bert_model):
        super(BERTClassifier, self).__init__()
        self.bert_model = bert_model
        self.dense = tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')

    def call(self, inputs):
        input_ids, attention_masks = inputs
        bert_outputs = self.bert_model(input_ids, attention_mask=attention_masks)
        cls_output = bert_outputs.last_hidden_state[:, 0, :]
        return self.dense(cls_output)

# Initialize BERT classifier
bert_classifier = BERTClassifier(bert_model)
bert_classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                        loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the BERT model
bert_classifier.fit([X_train_ids, X_train_masks], y_train, epochs=3, batch_size=16)

# Evaluate the BERT model
y_pred_bert = np.argmax(bert_classifier.predict([X_test_ids, X_test_masks]), axis=1)
bert_accuracy = accuracy_score(y_test, y_pred_bert)
print(f'BERT Accuracy: {bert_accuracy:.4f}')

# Ensemble Voting Classifier combining BERT and RandomForest
voting_clf = VotingClassifier(
    estimators=[('bert', bert_classifier), ('rf', rf_clf)],
    voting='soft'
)

# Train the Voting Classifier
voting_clf.fit(X_train_combined, y_train)

# Evaluate the Voting Classifier
y_pred_voting = voting_clf.predict(X_test_combined)
voting_accuracy = accuracy_score(y_test, y_pred_voting)
print(f'Voting Classifier Accuracy: {voting_accuracy:.4f}')

# Generate the classification report for Voting Classifier
print(classification_report(y_test, y_pred_voting, target_names=label_encoder.classes_))

# Save the Voting Classifier model
with open('voting_classifier.pkl', 'wb') as f:
    pickle.dump(voting_clf, f)

# Save the RandomForest model separately
with open('random_forest.pkl', 'wb') as f:
    pickle.dump(rf_clf, f)

# Save the BERT model
bert_classifier.save_weights('bert_classifier_weights.h5')
