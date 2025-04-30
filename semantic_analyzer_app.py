import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from transformers import tokenization_utils_base

# Initialize the zero-shot classification pipeline
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
nlp_zero_shot = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

# Initialize the NER pipeline
ner_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
ner_model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
nlp_ner = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer)

def analyze_zero_shot_classification(text, labels):
    """ Perform Zero-Shot Classification on the given text. """
    result = nlp_zero_shot(text, candidate_labels=labels)
    return result

def perform_ner_analysis(text):
    """ Perform Named Entity Recognition (NER) on the given text. """
    result = nlp_ner(text)
    return result

# Streamlit App Layout
st.title("Semantic Analysis and Entity Recognition App")

st.sidebar.header("Select Task")
task = st.sidebar.selectbox("Choose an analysis type", ("Zero-Shot Classification", "Named Entity Recognition"))

if task == "Zero-Shot Classification":
    st.header("Zero-Shot Classification")
    text_input = st.text_area("Enter Text for Classification", "I love the new features of this product!")
    labels_input = st.text_area("Enter Candidate Labels (comma separated)", "positive, negative, neutral")
    
    if st.button("Analyze"):
        labels = [label.strip() for label in labels_input.split(",")]
        if text_input:
            result = analyze_zero_shot_classification(text_input, labels)
            st.write("Classification Results:")
            st.write(result)
        else:
            st.warning("Please enter text for classification.")
            
elif task == "Named Entity Recognition":
    st.header("Named Entity Recognition (NER)")
    text_input_ner = st.text_area("Enter Text for NER", "Barack Obama was born in Hawaii.")
    
    if st.button("Analyze"):
        if text_input_ner:
            result_ner = perform_ner_analysis(text_input_ner)
            st.write("Named Entities:")
            st.write(result_ner)
        else:
            st.warning("Please enter text for NER.")

# Sidebar info and about
st.sidebar.subheader("About")
st.sidebar.info("""
This app provides two main features:
1. **Zero-Shot Classification**: Analyze text and classify it into predefined categories without any training.
2. **Named Entity Recognition (NER)**: Extract named entities such as persons, organizations, and locations from the text.
""")
