import streamlit as st
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import spacy
import subprocess
import sys

# Ensure spaCy model is downloaded
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

# Load models
@st.cache_resource
def load_models():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    nlp = load_spacy_model()
    return model, nlp

model, nlp = load_models()

# Streamlit UI
st.set_page_config(page_title="Semantic Analyzer", layout="wide")
st.title("📊 Semantic Content Analyzer")

st.markdown("""
This app compares the semantic similarity between a **target keyword** and your page content to help improve contextual relevance.
""")

# Sidebar input
with st.sidebar:
    st.header("Settings")
    keyword = st.text_input("🔍 Target Keyword", "data science")
    top_n = st.slider("🔝 Top N Similar Sentences", min_value=1, max_value=20, value=5)

# Text input
content = st.text_area("📝 Paste Your Page Content Here:", height=300)

if st.button("Analyze"):
    if not content.strip():
        st.warning("Please paste your content before analyzing.")
        st.stop()

    # Split into sentences
    doc = nlp(content)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    if not sentences:
        st.error("No valid sentences found in content.")
        st.stop()

    # Encode keyword and content
    keyword_embedding = model.encode(keyword, convert_to_tensor=True)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    # Compute cosine similarities
    similarities = util.cos_sim(keyword_embedding, sentence_embeddings)[0].cpu().numpy()

    # Create DataFrame with results
    df = pd.DataFrame({
        "Sentence": sentences,
        "Semantic Similarity": similarities
    }).sort_values(by="Semantic Similarity", ascending=False).head(top_n)

    st.subheader("🔎 Top Matches")
    st.dataframe(df.style.format({"Semantic Similarity": "{:.3f}"}), use_container_width=True)

    st.subheader("📈 Similarity Score Distribution")
    st.line_chart(similarities)
