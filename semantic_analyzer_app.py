import streamlit as st
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import numpy as np

# Function to get sentiment category
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive", polarity
    elif polarity < 0:
        return "Negative", polarity
    else:
        return "Neutral", polarity

# Function to generate a word cloud
def generate_wordcloud(text):
    stopwords = set(["the", "and", "to", "of", "in", "it", "for", "on", "is", "with", "at", "this", "a", "an", "as", "by", "from", "be"])
    wordcloud = WordCloud(stopwords=stopwords, max_words=100, width=800, height=400, background_color="white").generate(text)
    return wordcloud

# Streamlit UI
st.title("Sentiment Analysis with TextBlob and WordCloud")
st.write("Enter a text below to analyze the sentiment and see a word cloud.")

# Text input
text_input = st.text_area("Enter your text here:")

if text_input:
    # Perform sentiment analysis
    sentiment, polarity = get_sentiment(text_input)
    
    # Show sentiment results
    st.subheader("Sentiment Analysis")
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Polarity: {polarity:.2f}")
    
    # Plot pie chart for sentiment distribution
    sentiment_data = {
        "Positive": 0,
        "Negative": 0,
        "Neutral": 0
    }
    
    # Count sentiment categories
    if sentiment == "Positive":
        sentiment_data["Positive"] += 1
    elif sentiment == "Negative":
        sentiment_data["Negative"] += 1
    else:
        sentiment_data["Neutral"] += 1

    # Create Pie Chart
    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots()
    ax.pie(sentiment_data.values(), labels=sentiment_data.keys(), autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)
    
    # Generate word cloud
    st.subheader("Word Cloud")
    wordcloud = generate_wordcloud(text_input)
    st.image(wordcloud.to_array(), use_column_width=True)
    
    # Display word frequencies
    word_list = text_input.split()
    word_counts = Counter(word_list)
    word_df = pd.DataFrame(word_counts.most_common(10), columns=["Word", "Frequency"])
    
    st.subheader("Top 10 Frequent Words")
    st.dataframe(word_df)

