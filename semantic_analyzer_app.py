import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
from textblob import download_corpora

# Ensure corpora is available for TextBlob
try:
    download_corpora.download_all()
except:
    pass

# App title
st.title("📊 Sentiment Analyzer with Visualizations")

# Text input from user
user_input = st.text_area("Enter text to analyze:", height=150)

# If text is entered
if user_input:
    # Split into sentences
    sentences = [s.strip() for s in user_input.split('.') if s.strip()]
    sentiments = []

    for sentence in sentences:
        blob = TextBlob(sentence)
        polarity = blob.sentiment.polarity

        if polarity > 0:
            sentiment_label = "Positive"
        elif polarity < 0:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        sentiments.append({
            "Sentence": sentence,
            "Polarity": polarity,
            "Sentiment": sentiment_label
        })

    df = pd.DataFrame(sentiments)

    # Display table
    st.subheader("📄 Sentiment Breakdown")
    st.dataframe(df)

    # Pie chart
    st.subheader("📈 Sentiment Distribution")
    sentiment_counts = df['Sentiment'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

    # WordCloud
    st.subheader("☁️ Word Cloud")
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(
        stopwords=stopwords,
        background_color='white',
        max_words=100,
        width=800,
        height=400
    ).generate(user_input)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.imshow(wordcloud, interpolation='bilinear')
    ax2.axis('off')
    st.pyplot(fig2)

else:
    st.info("Please enter some text to see the analysis.")
