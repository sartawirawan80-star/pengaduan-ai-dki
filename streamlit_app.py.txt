import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load model & tokenizer
MODEL_NAME = "indobenchmark/indobert-base-p1"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=5  # jumlah kategori
    )
    return tokenizer, model

tokenizer, model = load_model()

# Daftar kategori layanan (contoh)
labels = [
    "Infrastruktur",
    "Kesehatan",
    "Pendidikan",
    "Transportasi",
    "Keamanan"
]

st.title("AI Klasifikasi Pengaduan Masyarakat DKI Jakarta")
st.write("Demo Model NLP Berbasis Transformer (IndoBERT)")

input_text = st.text_area("Masukkan Pengaduan:")

if st.button("Klasifikasikan"):
    if input_text.strip() == "":
        st.warning("Silakan masukkan teks pengaduan.")
    else:
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probs).item()

        st.success(f"Kategori Prediksi: {labels[predicted_class]}")
        st.write("Probabilitas:")
        for i, label in enumerate(labels):
            st.write(f"{label}: {probs[0][i].item():.4f}")

