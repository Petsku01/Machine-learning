# Import TextBlob for sentiment analysis
from textblob import TextBlob

def analyze_sentiment(text):
    """Analyze sentiment of text and return polarity, subjectivity, category.
    
    Args:
        text (str): Text to analyze.
    
    Returns:
        tuple: (polarity, subjectivity, category) or (None, None, error_message).
    """
    # Check if input is a non-empty string and not too long (max 500 characters)
    if not text or not isinstance(text, str):
        return None, None, "Invalid input: Provide a non-empty string."
    if len(text) > 500:
        return None, None, "Input too long: Maximum 500 characters allowed."
    
    # Create TextBlob object for sentiment analysis
    blob = TextBlob(text)
    
    # Get sentiment from blob metrics
    sentiment = blob.sentiment
    
    # Categorize the sentiment based on polarity
    polarity = sentiment.polarity
    if polarity > 0:
        category = "Positive"
    elif polarity < 0:
        category = "Negative"
    else:
        category = "Neutral"
    
    return polarity, sentiment.subjectivity, category

def display_summary(results):
    """Display a summary of all analyzed texts.
    
    Args:
        results (list): List of tuples (text, polarity, subjectivity, category).
    """
    print("\nSummary of Analyzed Texts:")
    print("==========================")
    for idx, (text, polarity, subjectivity, category) in enumerate(results, 1):
        print("Analysis " + str(idx) + ":")
        print("Text: " + text)
        print("Polarity: " + str(round(polarity, 2)))
        print("Subjectivity: " + str(round(subjectivity, 2)))
        print("Category: " + category)
        print("--------------------------")

def main():
    """Prompt user for text input, analyze sentiment, and show summary."""
    # List to store analysis results
    results = []
    
    while True:
        # Prompt user for text
        print("\nEnter text to analyze (max 500 characters, or 'exit' to quit):")
        try:
            text = input().strip()
        except EOFError:
            print("Input error: Please run in an interactive Python environment.")
            break
        
        # Check if user wants to exit
        if text.lower() == "exit":
            break
        
        # Analyze sentiment of user input
        polarity, subjectivity, category = analyze_sentiment(text)
        
        # Print results with simple strings
        if polarity is not None:
            print("\nResults:")
            print("Text: " + text)
            print("Polarity: " + str(round(polarity, 2)))
            print("Subjectivity: " + str(round(subjectivity, 2)))
            print("Category: " + category)
            # Store result for summary
            results.append((text, polarity, subjectivity, category))
        else:
            print(category)  # Print error message
        
        # Ask if user wants to continue
        print("\nContinue analyzing? (y/n):")
        try:
            choice = input().strip().lower()
            if choice != "y":
                break
        except EOFError:
            print("Input error: Exiting.")
            break
    
    # Display summary if any analyses were performed
    if results:
        display_summary(results)
    else:
        print("\nNo texts analyzed.")

# Run program with error handling
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error: " + str(e))
