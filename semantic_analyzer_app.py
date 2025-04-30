import streamlit as st
import spacy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from fpdf import FPDF
import base64
import io

# Load spaCy English model
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    return nlp

nlp = load_models()

st.set_page_config(page_title="Semantic Analyzer", layout="wide")
st.title("📊 Semantic Content Analyzer")
st.markdown("Compare two pieces of text, extract keywords/entities, and get improvement suggestions.")

# Sidebar controls
similarity_threshold = st.sidebar.slider("Semantic Similarity Threshold", 0.0, 1.0, 0.75, 0.01)

# Text input
text1 = st.text_area("✏️ Text 1 (Reference)", height=200)
text2 = st.text_area("✏️ Text 2 (To Compare)", height=200)

if text1 and text2:
    doc1 = nlp(text1)
    doc2 = nlp(text2)

    # TF-IDF similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    similarity_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    # Display similarity
    st.metric("🔁 Semantic Similarity Score", f"{similarity_score:.2f}", delta=None)

    if similarity_score < similarity_threshold:
        st.warning("Semantic similarity is lower than the threshold — consider improving relevance.")
    else:
        st.success("Texts are semantically similar!")

    # Visualization
    fig, ax = plt.subplots()
    sns.heatmap([[similarity_score]], annot=True, fmt=".2f", cmap="coolwarm", xticklabels=['Text 2'], yticklabels=['Text 1'], ax=ax)
    st.pyplot(fig)

    # Entity extraction
    ents1 = [(ent.text, ent.label_) for ent in doc1.ents]
    ents2 = [(ent.text, ent.label_) for ent in doc2.ents]

    st.subheader("📌 Named Entities")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Text 1 Entities**")
        st.dataframe(pd.DataFrame(ents1, columns=["Entity", "Label"]))
    with col2:
        st.markdown("**Text 2 Entities**")
        st.dataframe(pd.DataFrame(ents2, columns=["Entity", "Label"]))

    # Keyword extraction (noun chunks)
    chunks1 = list(set(chunk.text.lower() for chunk in doc1.noun_chunks))
    chunks2 = list(set(chunk.text.lower() for chunk in doc2.noun_chunks))

    st.subheader("🔑 Key Topics (Noun Phrases)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Text 1 Key Phrases**")
        st.write(chunks1)
    with col2:
        st.markdown("**Text 2 Key Phrases**")
        st.write(chunks2)

    # Recommendations
    st.subheader("🛠 Recommendations to Improve Semantic Similarity")
    recommendations = []

    missing_entities = [ent for ent in ents1 if ent not in ents2]
    if missing_entities:
        recommendations.append(f"Text 2 is missing some important entities from Text 1: {[e[0] for e in missing_entities]}.")

    missing_chunks = [chunk for chunk in chunks1 if chunk not in chunks2]
    if missing_chunks:
        recommendations.append(f"Text 2 might lack coverage on topics: {missing_chunks[:5]}...")

    verbs1 = [token.lemma_ for token in doc1 if token.pos_ == "VERB"]
    verbs2 = [token.lemma_ for token in doc2 if token.pos_ == "VERB"]
    missing_verbs = [v for v in verbs1 if v not in verbs2]
    if missing_verbs:
        recommendations.append(f"Text 2 may underuse action verbs like: {missing_verbs[:5]}.")

    if abs(len(text1.split()) - len(text2.split())) > 30:
        recommendations.append("There’s a significant word count difference — try expanding the shorter text.")

    if recommendations:
        for rec in recommendations:
            st.markdown(f"🔧 {rec}")
    else:
        st.success("Text 2 covers most semantic elements of Text 1!")

    # Export to Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        pd.DataFrame(ents1, columns=["Entity", "Label"]).to_excel(writer, sheet_name='Entities_Text1', index=False)
        pd.DataFrame(ents2, columns=["Entity", "Label"]).to_excel(writer, sheet_name='Entities_Text2', index=False)
        pd.DataFrame({"Chunk_Text1": chunks1}).to_excel(writer, sheet_name='Chunks_Text1', index=False)
        pd.DataFrame({"Chunk_Text2": chunks2}).to_excel(writer, sheet_name='Chunks_Text2', index=False)
        pd.DataFrame({"Recommendations": recommendations}).to_excel(writer, sheet_name='Suggestions', index=False)
        writer.save()
    st.download_button("📥 Download Full Report (.xlsx)", data=output.getvalue(), file_name="semantic_analysis.xlsx")

    # Optional: PDF Report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Semantic Analysis Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Similarity Score: {similarity_score:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Threshold: {similarity_threshold}", ln=True)
    pdf.add_page()
    pdf.cell(200, 10, txt="Recommendations:", ln=True)
    for rec in recommendations:
        pdf.multi_cell(0, 10, txt=f"- {rec}")

    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    st.download_button("📄 Download PDF Report", data=pdf_output.getvalue(), file_name="semantic_analysis.pdf")
