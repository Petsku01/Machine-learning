# Import the TextBlob library for natural language processing and sentiment analysis
from textblob import TextBlob

# Define a sample text string to analyze for sentiment
text = "This product is great!"

# Create a TextBlob object to process the text for sentiment analysis
blob = TextBlob(text)

# Extract sentiment metrics (polarity and subjectivity) from the text
sentiment = blob.sentiment

# Print the polarity score, ranging from -1 (negative) to 1 (positive)
print(f"Polarity: {sentiment.polarity}")

# Print the subjectivity score, ranging from 0 (factual) to 1 (opinion-based)
print(f"Subjectivity: {sentiment.subjectivity}")
