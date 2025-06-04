# Imports TextBlob for sentiment analysis
from textblob import TextBlob

def analyze_sentiment(text):
    """Analyze sentiment of text and return polarity, subjectivity, category.
    
    Args:
        text (str): Text to analyze.
    
    Returns:
        tuple: (polarity, subjectivity, category) or (None, None, error_message).
    """
    # Check if input is a non-empty string
    if not text or not isinstance(text, str):
        return None, None, "Invalid input: Provide a non-empty string."
    
    # Createa a TextBlob object for the sentiment analysis
    blob = TextBlob(text)
    
    # Get blob sentiment metrics
    sentiment = blob.sentiment
    
    # Categorize sentiment based on polarity
    polarity = sentiment.polarity
    if polarity > 0:
        category = "Positive"
    elif polarity < 0:
        category = "Negative"
    else:
        category = "Neutral"
    
    return polarity, sentiment.subjectivity, category

def main():
    """Prompts the user for text input and analyze sentiment until user exits."""
    while True:
        # Prompt user for text or to exit
        print("Enter text to analyze (or 'exit' to quit):")
        try:
            text = input().strip()
        except EOFError:
            print("Input error: Please run in an interactive Python environment.")
            break
        
        # Check if user wants to exit
        if text.lower() == "exit":
            print("Exiting sentiment analysis.")
            break
        
        # Analyze sentiment of given user input
        polarity, subjectivity, category = analyze_sentiment(text)
        
        # Print results with simple strings
        if polarity is not None:
            print("\nResults:")
            print("Text: " + text)
            print("Polarity: " + str(round(polarity, 2)))
            print("Subjectivity: " + str(round(subjectivity, 2)))
            print("Category: " + category)
        else:
            print(category)

# Run program with error handling
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error: " + str(e))
