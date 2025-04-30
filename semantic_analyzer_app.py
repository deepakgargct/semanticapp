import streamlit as st
from textblob import TextBlob
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS

# Set title for the app
st.title("Advanced Sentiment & Subjectivity Analysis with TextBlob")

# User input for a list of texts
text_input = st.text_area("Enter text for analysis:", "", height=200)
submit_button = st.button("Analyze Text")

# Store results for sentiment trend analysis
if 'sentiment_data' not in st.session_state:
    st.session_state.sentiment_data = []

# Function to analyze sentiment and subjectivity
def analyze_sentiment(text):
    blob = TextBlob(text)
    
    # Sentiment analysis: polarity (-1 to 1) and subjectivity (0 to 1)
    sentiment = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Classify sentiment as positive, neutral, or negative
    if sentiment > 0:
        sentiment_label = "Positive"
    elif sentiment < 0:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"
    
    return sentiment_label, sentiment, subjectivity

# If the button is clicked, analyze the sentiment and add data for trend
if submit_button and text_input:
    sentiment_label, sentiment, subjectivity = analyze_sentiment(text_input)
    
    # Add sentiment data to session state
    st.session_state.sentiment_data.append({
        'text': text_input,
        'sentiment_label': sentiment_label,
        'sentiment_score': sentiment,
        'subjectivity': subjectivity
    })
    
    # Display the results
    st.subheader("Analysis Results")
    st.write(f"Sentiment: **{sentiment_label}**")
    st.write(f"Sentiment Polarity: **{sentiment}**")
    st.write(f"Subjectivity: **{subjectivity}**")
    
    # Display sentiment color
    if sentiment > 0:
        st.markdown(f"<div style='color: green; font-size: 20px;'>Sentiment is Positive with polarity {sentiment}</div>", unsafe_allow_html=True)
    elif sentiment < 0:
        st.markdown(f"<div style='color: red; font-size: 20px;'>Sentiment is Negative with polarity {sentiment}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='color: gray; font-size: 20px;'>Sentiment is Neutral with polarity {sentiment}</div>", unsafe_allow_html=True)

# **Advanced Visualization 1**: Sentiment Trend Analysis (Plotly Graph)
if len(st.session_state.sentiment_data) > 1:
    sentiment_scores = [data['sentiment_score'] for data in st.session_state.sentiment_data]
    sentiment_labels = [data['sentiment_label'] for data in st.session_state.sentiment_data]
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=list(range(1, len(sentiment_scores) + 1)),
                             y=sentiment_scores,
                             mode='lines+markers',
                             name='Sentiment Score',
                             marker=dict(color='blue')))
    
    fig.update_layout(title="Sentiment Trend Over Multiple Entries",
                      xaxis_title="Text Entries",
                      yaxis_title="Sentiment Score",
                      showlegend=True)
    
    st.plotly_chart(fig)

# **Advanced Visualization 2**: Sentiment Pie Chart (Matplotlib)
if len(st.session_state.sentiment_data) > 0:
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    
    for data in st.session_state.sentiment_data:
        sentiment_counts[data['sentiment_label']] += 1
    
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%', startangle=90, colors=['green', 'red', 'gray'])
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Sentiment Distribution')
    st.pyplot(fig)

# **Advanced Visualization 3**: Word Cloud (using WordCloud with customization)
if len(st.session_state.sentiment_data) > 0:
    all_text = ' '.join([data['text'] for data in st.session_state.sentiment_data])
    
    # Custom stopwords list, can be extended by the user
    custom_stopwords = set(STOPWORDS).union({'the', 'and', 'is', 'to', 'of'})
    
    # Create the word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=custom_stopwords,
        max_words=100,
        colormap='viridis'
    ).generate(all_text)
    
    st.subheader("Word Cloud of Analyzed Texts")
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    st.pyplot()

# Provide instructions
else:
    st.write("Please enter some text for sentiment analysis.")

