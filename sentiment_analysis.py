import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Sample Reviews
data = {
    'Review': [
        'This product is amazing',
        'I love this phone',
        'Worst experience ever',
        'Very bad quality',
        'This is a mobile phone',
        'Excellent performance',
        'Not worth the money'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Function for Sentiment Analysis
def get_sentiment(text):
    analysis = TextBlob(text)

    if analysis.sentiment.polarity > 0:
        return 'Positive'

    elif analysis.sentiment.polarity < 0:
        return 'Negative'

    else:
        return 'Neutral'

# Apply Sentiment Function
df['Sentiment'] = df['Review'].apply(get_sentiment)

# Print Results
print("SENTIMENT ANALYSIS RESULTS")
print(df)

# Count Sentiments
sentiment_counts = df['Sentiment'].value_counts()

print("\nSENTIMENT COUNTS")
print(sentiment_counts)

# Create Pie Chart
plt.pie(
    sentiment_counts,
    labels=sentiment_counts.index,
    autopct='%1.1f%%'
)

# Chart Title
plt.title("Sentiment Analysis Results")

# Show Graph
plt.show()