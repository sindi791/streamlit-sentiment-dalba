
import streamlit as st
import pandas as pd
import pickle
from utils import preprocess_text

st.set_page_config(page_title="D'Alba Sentiment App", layout="wide")

st.markdown("""<h1 style='text-align: center; color: #7F27FF;'>Analisis Sentimen Komentar Instagram Produk D'Alba</h1>""", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #eee;'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload file komentar (.xlsx atau .csv)", type=["xlsx", "csv"])
model_choice = st.radio("🔍 Pilih Model Klasifikasi", ["SVM", "Naive Bayes"], horizontal=True)

if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        data = pd.read_excel(uploaded_file)
    else:
        data = pd.read_csv(uploaded_file)

    kolom_komentar = None
    for col in data.columns:
        if col.lower() in ['komentar', 'comment']:
            kolom_komentar = col
            break

    if not kolom_komentar:
        st.error("❌ Kolom komentar tidak ditemukan. Gunakan nama kolom 'komentar' atau 'comment'.")
    else:
        if st.button("🚀 Klasifikasikan"):
            data['cleaned'] = data[kolom_komentar].astype(str).apply(preprocess_text)

            try:
                vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
                model_svm = pickle.load(open("model_svm.pkl", "rb"))
                model_nb = pickle.load(open("model_nb.pkl", "rb"))
            except Exception as e:
                st.error(f"⚠️ Gagal memuat model: {e}")
                st.stop()

            X = vectorizer.transform(data['cleaned'])

            if X.shape[1] == 0:
                st.warning("⚠️ Teks tidak menghasilkan fitur. Coba data lain.")
                st.stop()

            prediction = model_svm.predict(X) if model_choice == "SVM" else model_nb.predict(X)
            data['sentimen'] = prediction

            st.success("✅ Klasifikasi berhasil!")
            st.dataframe(data[[kolom_komentar, 'sentimen']])

            st.subheader("📊 Distribusi Sentimen")
            st.bar_chart(data['sentimen'].value_counts())
