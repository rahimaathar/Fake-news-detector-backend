import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

def preprocess_text(text):
    """Basic text preprocessing"""
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = ''.join([char for char in text if char.isalpha() or char.isspace()])
        return text
    return ""

print("Loading and preparing data...")

# Load the datasets
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')

# Add labels
fake_df['label'] = 1  # 1 for fake news
true_df['label'] = 0  # 0 for real news

# Combine the datasets
df = pd.concat([fake_df, true_df], ignore_index=True)

# Combine title and text columns if they exist
if 'title' in df.columns and 'text' in df.columns:
    df['text'] = df['title'] + ' ' + df['text']

# Preprocess the text
print("Preprocessing text...")
df['text'] = df['text'].apply(preprocess_text)

# Remove rows with empty text
df = df[df['text'].str.len() > 0]

print(f"Total number of articles: {len(df)}")
print(f"Number of fake news articles: {len(df[df['label'] == 1])}")
print(f"Number of real news articles: {len(df[df['label'] == 0])}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], 
    df['label'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['label']  # Ensure balanced split
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Create and fit vectorizer
print("\nCreating and fitting vectorizer...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
print("Training model...")
model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    C=1.0,
    class_weight='balanced'
)
model.fit(X_train_vec, y_train)

# Evaluate model
train_score = model.score(X_train_vec, y_train)
test_score = model.score(X_test_vec, y_test)
print(f"\nModel Performance:")
print(f"Training accuracy: {train_score:.2f}")
print(f"Testing accuracy: {test_score:.2f}")

# Save model and vectorizer
print("\nSaving model and vectorizer...")
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Model and vectorizer saved successfully!") 