# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from urllib.parse import urlparse
import re

# Function to extract features from a URL
def extract_features(url):
    """
    Extracts features from a URL.
    
    Args:
        url (str): The URL to analyze.
        
    Returns:
        dict: A dictionary of features, or None if invalid.
    """
    try:
        if not isinstance(url, str):
            raise ValueError("URL must be a string")
        parsed = urlparse(url)
        features = {}
        features['length'] = len(url)
        features['num_digits'] = sum(c.isdigit() for c in url)
        features['num_special_chars'] = len(re.findall(r'[!@#$%^&*(),.?]', url))
        features['has_secure'] = 1 if parsed.scheme == 'https' else 0
        features['url'] = url  # Store URL for TF-IDF
        return features
    except ValueError as e:
        print(f"Error processing URL: {e}")
        return None

# Function to create a sample dataset
def create_sample_dataset():
    """
    Creates a small dataset of URLs with labels.
    
    Returns:
        pd.DataFrame: A DataFrame with URL features and labels.
    """
    urls = [
        ('https://www.google.com', 0),
        ('http://login-paypal.example.com', 1),
        ('https://amazon.co.uk', 0),
        ('http://secure-bankofamerica-login.com', 1),
        ('https://github.com', 0),
        ('http://netflix-secure-login.com', 1)
    ]
    data = []
    for url, label in urls:
        features = extract_features(url)
        if features:
            features['label'] = label
            data.append(features)
    return pd.DataFrame(data)

# Main function to train and evaluate the model
def main():
    """
    Trains a logistic regression model to detect phishing URLs.
    """
    df = create_sample_dataset()
    X = df[['length', 'num_digits', 'num_special_chars', 'has_secure']]
    y = df['label']
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(df['url'])
    X_tfidf_df = pd.DataFrame(X_tfidf.toarray())
    
    # Combine features
    X = pd.concat([X, X_tfidf_df], axis=1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

# Entry point
if __name__ == "__main__":
    main()