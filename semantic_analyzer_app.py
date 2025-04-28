# semantic_analyzer_app.py

# Install needed if not already:
# pip install -r requirements.txt

import streamlit as st
from sentence_transformers import SentenceTransformer, util
import spacy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from sklearn.feature_extraction.text import TfidfVectorizer
from bertopic import BERTopic
from fpdf import FPDF
from io import BytesIO

# ---- 1. CACHE MODELS ----
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    nlp = spacy.load("en_core_web_sm")
    return embedder, nlp

embedder, nlp = load_models()

# ---- 2. PAGE SETTINGS ----
st.set_page_config(page_title="Semantic Analyzer Pro", layout="wide")
st.title("🔥 Semantic Content Analyzer Pro")

# ---- 3. INPUT AREA ----
st.subheader("Paste Your Texts (separate each by '###')")
text_input = st.text_area("Enter your texts below", height=300)

# Semantic threshold input
threshold = st.slider("Semantic Similarity Threshold for Recommendations", 0.0, 1.0, 0.8, step=0.01)

if st.button("Analyze"):
    with st.spinner("Processing..."):

        # ---- 4. PROCESS TEXTS ----
        texts = [t.strip() for t in text_input.split('###') if t.strip()]
        texts_clean = [t.lower() for t in texts]

        # Embeddings
        embeddings = embedder.encode(texts_clean, convert_to_tensor=True)
        similarity_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()

        # Named Entities
        entities = {}
        for idx, doc in enumerate(nlp.pipe(texts)):
            ents = [(ent.text, ent.label_) for ent in doc.ents]
            entities[idx] = ents

        # Keywords (Top 5 per text)
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(texts_clean)
        feature_names = vectorizer.get_feature_names_out()
        keywords = []
        for row in X:
            top_indices = row.toarray().flatten().argsort()[-5:][::-1]
            keywords.append([feature_names[i] for i in top_indices])

        # Topics
        topic_model = BERTopic(umap_model=umap.UMAP(n_neighbors=5, min_dist=0.3))
        topics, _ = topic_model.fit_transform(texts_clean)

        # ---- 5. VISUALIZATIONS ----
        st.header("📈 Semantic Similarity Matrix")
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=False, yticklabels=False)
        st.pyplot(fig)

        st.header("📉 UMAP Projection of Texts")
        reducer = umap.UMAP()
        embedding_2d = reducer.fit_transform(embeddings.cpu())
        fig2, ax2 = plt.subplots()
        ax2.scatter(embedding_2d[:,0], embedding_2d[:,1])
        for i, txt in enumerate(range(len(texts))):
            ax2.annotate(txt, (embedding_2d[i,0], embedding_2d[i,1]))
        st.pyplot(fig2)

        st.header("🧠 Named Entities")
        for idx, ents in entities.items():
            st.write(f"Text {idx}:", ents)

        st.header("🔑 Extracted Keywords")
        for idx, kws in enumerate(keywords):
            st.write(f"Text {idx}:", kws)

        st.header("📚 Topics Detected")
        topics_df = topic_model.get_topic_info()
        st.dataframe(topics_df)

        # ---- 6. RECOMMENDATIONS ----
        st.header("💡 Recommendations to Improve Semantic Similarity")
        recommendations = []

        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                score = similarity_matrix[i][j]
                if score < threshold:
                    recs = []
                    missing_in_1 = [kw for kw in keywords[j] if kw not in keywords[i]]
                    missing_in_2 = [kw for kw in keywords[i] if kw not in keywords[j]]
                    if missing_in_1:
                        recs.append(f"Add to Text {i}: {missing_in_1}")
                    if missing_in_2:
                        recs.append(f"Add to Text {j}: {missing_in_2}")
                    recommendations.append({
                        "Text Pair": f"{i} - {j}",
                        "Semantic Score": round(float(score), 2),
                        "Recommendations": " | ".join(recs)
                    })

        if recommendations:
            recs_df = pd.DataFrame(recommendations)

            for _, rec in recs_df.iterrows():
                color = "green" if rec["Semantic Score"] > 0.7 else "orange" if rec["Semantic Score"] > 0.5 else "red"
                st.markdown(
                    f"<p style='color:{color};'>Recommendation for {rec['Text Pair']} (Score: {rec['Semantic Score']}): {rec['Recommendations']}</p>",
                    unsafe_allow_html=True
                )
        else:
            st.success("All texts have good semantic similarity above the threshold!")

        # ---- 7. DOWNLOADS ----

        st.subheader("📥 Export Reports")

        # Excel download
        def create_excel(similarity_matrix, entities, keywords, topics_df, recs_df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                pd.DataFrame(similarity_matrix).to_excel(writer, sheet_name="Similarity Matrix", index=False)
                
                entity_list = []
                for idx, ents in entities.items():
                    for ent_text, ent_label in ents:
                        entity_list.append({"Text ID": idx, "Entity": ent_text, "Label": ent_label})
                pd.DataFrame(entity_list).to_excel(writer, sheet_name="Entities", index=False)
                
                kw_list = []
                for idx, kws in enumerate(keywords):
                    for kw in kws:
                        kw_list.append({"Text ID": idx, "Keyword": kw})
                pd.DataFrame(kw_list).to_excel(writer, sheet_name="Keywords", index=False)
                
                topics_df.to_excel(writer, sheet_name="Topics", index=False)
                recs_df.to_excel(writer, sheet_name="Recommendations", index=False)

            output.seek(0)
            return output

        excel_file = create_excel(similarity_matrix, entities, keywords, topics_df, recs_df)
        st.download_button(
            label="📥 Download Full Report as Excel",
            data=excel_file,
            file_name="semantic_analysis_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # PDF download
        def create_pdf_report(recs_df):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, txt="Semantic Analysis Report", ln=True, align='C')

            pdf.set_font("Arial", size=12)
            pdf.ln(10)
            pdf.cell(200, 10, txt="Summary of Recommendations", ln=True)

            for index, row in recs_df.iterrows():
                pdf.ln(5)
                pdf.multi_cell(0, 10, f"Text Pair: {row['Text Pair']} | Score: {row['Semantic Score']}\nRecommendations: {row['Recommendations']}")

            output = BytesIO()
            pdf.output(output)
            output.seek(0)
            return output

        pdf_file = create_pdf_report(recs_df)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_file,
            file_name="semantic_analysis_report.pdf",
            mime="application/pdf",
        )
