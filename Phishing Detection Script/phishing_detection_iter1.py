# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from urllib.parse import urlparse

# Function to extract basic features from a URL
def extract_features(url):
    """
    Extracts basic features from a URL.
    
    Args:
        url (str): The URL to analyze.
        
    Returns:
        dict: A dictionary of features.
    """
    parsed = urlparse(url)
    features = {}
    features['length'] = len(url)
    features['has_secure'] = 1 if parsed.scheme == 'https' else 0
    return features

# Function to create a sample dataset
def create_sample_dataset():
    """
    Creates a small dataset of URLs with labels.
    
    Returns:
        pd.DataFrame: A DataFrame with URL features and labels.
    """
    urls = [
        ('https://www.google.com', 0),
        ('http://login-paypal.com', 1),
        ('https://github.com', 0),
        ('http://secure-bank-login.com', 1)
    ]
    data = []
    for url, label in urls:
        features = extract_features(url)
        features['label'] = label
        data.append(features)
    return pd.DataFrame(data)

# Main function to train and evaluate the model
def main():
    """
    Trains a logistic regression model to detect phishing URLs.
    """
    df = create_sample_dataset()
    X = df[['length', 'has_secure']]
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    
    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))

# Entry point
if __name__ == "__main__":
    main()