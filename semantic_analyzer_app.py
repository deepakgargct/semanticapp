import streamlit as st
from transformers import pipeline

# Initialize Hugging Face pipelines
nlp_zero_shot = pipeline('zero-shot-classification')
nlp_ner = pipeline('ner')

# Streamlit app title and layout
st.set_page_config(page_title="📊 Enhanced Semantic Content Analyzer", layout="wide")
st.title("📊 Enhanced Semantic Content Analyzer")

# Text input area
text_input = st.text_area("Enter text for semantic analysis:")

# Predefined candidate labels for zero-shot classification (you can expand this list)
labels = ["Technology", "Science", "Politics", "Sports", "Entertainment", "Business", "Healthcare", "Culture"]

# Function to perform entity recognition
def extract_entities(text):
    return nlp_ner(text)

# Function to perform zero-shot classification
def classify_text(text):
    return nlp_zero_shot(text, candidate_labels=labels)

# When text is entered, perform both entity analysis and zero-shot classification
if text_input:
    # Entity Recognition
    with st.spinner('Extracting named entities...'):
        entities = extract_entities(text_input)
    
    st.write("### Named Entity Recognition (NER):")
    for entity in entities:
        st.write(f"- **{entity['word']}**: {entity['entity_group']} (Confidence: {entity['score']:.4f})")
    
    # Zero-shot Classification
    with st.spinner('Performing semantic classification...'):
        classification_result = classify_text(text_input)
    
    st.write("### Semantic Classification Results:")
    st.write(f"Predicted label: **{classification_result['labels'][0]}**")
    st.write(f"Score: {classification_result['scores'][0]:.4f}")

    # Show all candidate labels and their respective scores
    st.write("#### All candidate labels and scores:")
    for label, score in zip(classification_result['labels'], classification_result['scores']):
        st.write(f"- **{label}**: {score:.4f}")

# Footer
st.markdown(
    """
    ---
    This app uses the Hugging Face Transformers library for both:
    - **Zero-Shot Semantic Classification** to categorize text into predefined labels.
    - **Named Entity Recognition (NER)** to extract entities such as persons, organizations, and locations from the text.
    """
)
