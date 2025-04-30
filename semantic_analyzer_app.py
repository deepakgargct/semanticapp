import streamlit as st
from transformers import pipeline
import torch
import pandas as pd  # Import pandas to fix the error

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

    # Display the structured output
    st.subheader("Text Analysis Result")

    # Display the original input text
    st.write(f"**Input Text:**")
    st.write(f"{text}")

    # Create a structured output for predicted categories
    st.write("### Predicted Categories and Scores:")
    result_df = pd.DataFrame({
        "Category": result['labels'],
        "Score": result['scores']
    })

    st.write(result_df)

    # Optionally, add entity recognition using spaCy
    import spacy
    nlp_spacy = spacy.load("en_core_web_sm")
    doc = nlp_spacy(text)
    
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    if entities:
        st.write("### Detected Entities:")
        entities_df = pd.DataFrame(entities, columns=["Entity", "Label"])
        st.write(entities_df)
    else:
        st.write("No entities detected.")
else:
    st.write("Please enter some text to analyze.")
