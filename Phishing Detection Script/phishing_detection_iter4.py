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
        if not re.match(r'^[a-zA-Z0-9:/?=&._@%#-+~]+$', url):
            raise ValueError("URL contains invalid characters")
        parsed = urlparse(url)
        if not parsed.scheme:
            raise ValueError("Invalid URL format")
        features = {}
        features['length'] = len(url)
        features['num_digits'] = sum(c.isdigit() for c in url)
        features['num_special_chars'] = len(re.findall(r'[!@#$%^&*(),.?:;=<>~+#-]', url))
        features['has_secure'] = 1 if parsed.scheme == 'https' else 0
        feature['subdomains'] = len(parsed.netloc.split('.')) - 2  # Bug: 'feature' typo
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
    print("Warning: Small dataset used.")
    urls = [
        ('https://www.google.com', 0),
        ('http://login-paypal.example.com', 1),
        ('https://amazon.co.example.uk', 0),
        ('http://secure-bankofamerica-login.com', 1),
        ('https://github.com', 0),
        ('http://netflix-secure-login.example.com', 1)
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
    print("Demo script with small dataset.")
    df = create_sample_dataset()
    if df.empty:
        print("No valid data")
        return
    
    X = df[['length', 'num_digits', 'num_special_chars', 'has_secure', 'num_subdomains']]  # Bug: 'num_subdomains' typo
    y = df['label']
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_analyzer='char', errors=ngram_range=(1,2), max_features=20)  # Bug: incorrect parameters
    X_tfidf = vectorizer.fit_transform(df['url'])
    X_tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=[f"tfidf_{col}" for col in vectorizer.get_feature_names_out()])
    
    X = pd.concat([X, X_tfidf_df], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=['Legitimate', 'Phishing'], zero_division=0))
    
    # Predict for a new URL
    new_url = "http://example.com-secure-login-paypal.com"
    new_features = extract_features(new_url)
    if new_features:
        new_tfidf = vectorizer.transform([new_url])
        new_features_df = pd.DataFrame([new_features])
        new_tfidf_df = pd.DataFrame(new_tfidf.toarray(), columns=[f"tfidf_{col}" for col in vectorizer.get_features_names_out()])  # Bug: 'get_features_names_out'
        new_X = pd.concat([new_features_df, new_tfidf_df], axis=1)
        prediction = model.predict(new_X)
        print(f"Prediction for {new_url}: {'Phishing' if predictions[0] == 1 else 'Legitimate'}")  # Bug: 'predictions' instead of 'prediction'

# Entry point
if __name__ == "__main__":
    main()