import streamlit as st
import spacy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
from fpdf import FPDF

# Load spaCy model
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

# App UI
st.title("🧠 Semantic Analyzer (Offline-Friendly with Reports)")
text1 = st.text_area("Text 1")
text2 = st.text_area("Text 2")
threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.5)

if st.button("Analyze") and text1 and text2:
    docs = [text1, text2]

    # TF-IDF similarity
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(docs)
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    st.metric("Similarity Score", f"{similarity_score:.4f}")
    if similarity_score < threshold:
        st.error("❌ Low similarity – consider improving the content.")
    else:
        st.success("✅ Good similarity.")

    # Named Entities
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    ents1 = [(ent.text, ent.label_) for ent in doc1.ents]
    ents2 = [(ent.text, ent.label_) for ent in doc2.ents]

    st.subheader("🔍 Named Entities")
    col1, col2 = st.columns(2)
    col1.write("**Text 1 Entities:**")
    col1.write(ents1 or "None")
    col2.write("**Text 2 Entities:**")
    col2.write(ents2 or "None")

    # Heatmap
    df = pd.DataFrame(tfidf_matrix.toarray(), index=["Text 1", "Text 2"], columns=vectorizer.get_feature_names_out())
    st.subheader("🧩 TF-IDF Heatmap")
    plt.figure(figsize=(10, 4))
    sns.heatmap(df.T, cmap="YlGnBu", annot=False)
    st.pyplot(plt)

    # --- Excel Report ---
    output_excel = BytesIO()
    with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="TFIDF_Matrix")
        pd.DataFrame({"Text 1 Entities": [str(e) for e in ents1]}).to_excel(writer, sheet_name="Entities_1")
        pd.DataFrame({"Text 2 Entities": [str(e) for e in ents2]}).to_excel(writer, sheet_name="Entities_2")
        pd.DataFrame({"Similarity Score": [similarity_score]}).to_excel(writer, sheet_name="Summary")
    output_excel.seek(0)
    st.download_button("📥 Download Excel Report", data=output_excel, file_name="semantic_report.xlsx")

    # --- PDF Report ---
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Semantic Similarity Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Similarity Score: {similarity_score:.4f}", ln=True)
    pdf.cell(200, 10, txt="--- Entities ---", ln=True)
    pdf.multi_cell(0, 10, txt=f"Text 1 Entities: {ents1 or 'None'}")
    pdf.multi_cell(0, 10, txt=f"Text 2 Entities: {ents2 or 'None'}")
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    st.download_button("📄 Download PDF Summary", data=pdf_output, file_name="semantic_summary.pdf", mime="application/pdf")
