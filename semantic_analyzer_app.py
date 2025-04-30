import streamlit as st
from transformers import pipeline
import torch

# Initialize the Hugging Face transformer model pipeline
@st.cache_resource
def load_model():
    # Load a zero-shot classification pipeline for semantic analysis
    nlp_zero_shot = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
    return nlp_zero_shot

# Define categories for zero-shot classification
categories = ["politics", "business", "sports", "technology", "entertainment", "health"]

# Streamlit app layout
st.title("Semantic Analysis with Hugging Face Transformers")

st.write(
    "This app uses Hugging Face's Zero-Shot Classification model to analyze the semantics of a given text. "
    "It can classify the text into categories such as politics, business, technology, etc."
)

# Input text area for user
text = st.text_area("Enter your text:", height=200)

if text:
    st.write("Analyzing the input text...")

    # Load the model
    nlp_zero_shot = load_model()

    # Perform Zero-Shot Classification
    result = nlp_zero_shot(text, candidate_labels=categories)

    # Display the result
    st.write("Predicted Categories:")
    st.write(f"Text: {text}")
    st.write(f"Predicted labels: {result['labels']}")
    st.write(f"Scores: {result['scores']}")
else:
    st.write("Please enter some text to analyze.")

# Optionally, you can also include an entity analysis using spaCy if needed.
