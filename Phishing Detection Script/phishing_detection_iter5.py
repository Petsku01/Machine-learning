# Import required libraries for data processing, machine learning, and URL parsing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from urllib.parse import urlparse
import re

# Function to extract features from a URL for phishing detection
def extract_features(url):
    """
    Extracts numerical features from a URL, such as length and special character count.
    
    Args:
        url (str): The URL to analyze.
        
    Returns:
        dict: A dictionary of features, or None if the URL is invalid.
    """
    try:
        # Validate that the input is a string
        if not isinstance(url, str):
            raise ValueError("URL must be a string")
        # Check for allowed characters to prevent malicious input
        if not re.match(r'^[a-zA-Z0-9:/?=&._@%#-+~\[\]]+$', url):
            raise ValueError("URL contains invalid characters")
        # Parse the URL to ensure it's valid
        parsed = urlparse(url)
        # Restrict schemes to http and https
        if parsed.scheme not in ['http', 'https']:
            raise ValueError("Only http and https schemes are supported")
        # Check for complex netloc formats (e.g., IPv6, excluding ports)
        if ':' in parsed.netloc and not parsed.netloc.startswith('[') and not parsed.netloc.split(':')[-1].isdigit():
            raise ValueError("Complex netloc formats (e.g., IPv6) not supported")
        # Initialize feature dictionary
        features = {}
        # Feature: Total length of the URL
        features['length'] = len(url)
        # Feature: Number of digits in the URL
        features['num_digits'] = sum(c.isdigit() for c in url)
        # Feature: Number of special characters in the URL
        features['num_special_chars'] = len(re.findall(r'[!@#$%^&*(),.?:;=<>~+#-]', url))
        # Feature: Whether the URL uses HTTPS (1 for yes, 0 for no)
        features['has_https'] = 1 if parsed.scheme.lower() == 'https' else 0
        # Feature: Number of subdomains (based on netloc, ensuring non-negative)
        features['subdomains'] = max(0, len(parsed.netloc.split('.')) - 2)
        return features
    except (ValueError, TypeError) as e:
        # Log error and return None for invalid URLs
        print(f"Error processing URL: {e}")
        return None

# Function to create a sample dataset of URLs with labels
def create_sample_dataset():
    """
    Creates a small sample dataset of URLs labeled as legitimate (0) or phishing (1).
    
    Returns:
        pd.DataFrame: A DataFrame containing URL features and labels.
    """
    # Warn about the small dataset size
    print("Warning: Using a small sample dataset. For production, use a larger dataset.")
    # Define sample URLs with labels (0 = legitimate, 1 = phishing)
    urls = [
        ('https://www.google.com', 0),
        ('http://login-paypal.example.com', 1),
        ('https://amazon.co.example.uk', 0),
        ('http://secure-bankofamerica-login.com', 1),
        ('https://github.com', 0),
        ('http://netflix-secure-login.example.com', 1)
    ]
    data = []
    # Extract features for each URL and append to data list
    for url, label in urls:
        features = extract_features(url)
        if features:
            features['urls'] = url  # Store the original URL for TF-IDF
            features['label'] = label  # Store the label
            data.append(features)
        else:
            # Log skipped URLs
            print(f"Skipping invalid URL: {url}")
    # Convert data to DataFrame
    return pd.DataFrame(data)

# Main function to train and evaluate the phishing detection model
def main():
    """
    Trains a logistic regression model to detect phishing URLs and evaluates it.
    Also demonstrates prediction on a new URL.
    """
    # Warn about the demo nature of the script
    print("Note: This is a demo with a small dataset. Results may not be reliable.")
    # Create the sample dataset
    df = create_sample_dataset()
    # Check if the dataset is empty
    if df.empty:
        print("No valid data to process")
        return
    
    # Prepare numerical features for training
    X = df[['length', 'num_digits', 'num_special_chars', 'has_https', 'subdomains']]
    # Extract labels
    y = df['label']
    
    # Initialize TF-IDF vectorizer for URL text features
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,2), max_features=20)
    # Transform URLs into TF-IDF features
    X_tfidf = vectorizer.fit_transform(df['urls'])
    # Convert TF-IDF features to DataFrame with prefixed column names
    X_tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=[f"tfidf_{col}" for col in vectorizer.get_feature_names_out()])
    
    # Combine numerical and TF-IDF features
    X = pd.concat([X, X_tfidf_df], axis=1)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    # Check if test set is too small
    if len(X_test) < 2:
        print("Test set too small for meaningful evaluation")
        return
    
    # Warn about small test set size
    print("Warning: Test set has only", len(X_test), "samples, results may be unreliable.")
    
    # Initialize and train logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    predictions = model.predict(X_test)
    # Print accuracy
    print("Accuracy:", accuracy_score(y_test, predictions))
    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=['Legitimate', 'Phishing'], zero_division=0))
    
    # Example prediction for a new URL
    new_url = "http://example.com-secure-login-paypal.com"
    new_features = extract_features(new_url)
    if new_features:
        # Transform new URL into TF-IDF features
        new_tfidf = vectorizer.transform([new_url])
        # Convert numerical features to DataFrame with explicit index
        new_features_df = pd.DataFrame([new_features], index=[0])
        # Convert TF-IDF features to DataFrame with prefixed columns and align
        new_tfidf_df = pd.DataFrame(new_tfidf.toarray(), columns=[f"tfidf_{col}" for col in vectorizer.get_feature_names_out()], index=[0])
        new_tfidf_df = new_tfidf_df.reindex(columns=X_tfidf_df.columns, fill_value=0)
        # Combine features
        new_X = pd.concat([new_features_df, new_tfidf_df], axis=1)
        # Predict phishing status
        prediction = model.predict(new_X)
        print(f"\nPrediction for {new_url}: {'Phishing' if prediction[0] == 1 else 'Legitimate'}")
    else:
        # Handle invalid URL
        print(f"Cannot predict for invalid URL: {new_url}")

# Entry point for the script
if __name__ == "__main__":
    main()