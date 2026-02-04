import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, accuracy_score

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="Dashboard Analisis Sentimen Maxim",
    layout="wide"
)

# =====================================================
# LOAD DATA & MODEL
# =====================================================
@st.cache_data
def load_data():
    return pd.read_csv("maxim_reviews_labeled.csv")

@st.cache_resource
def load_models():
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    model_xgb = joblib.load("model_xgb.pkl")
    model_rf = joblib.load("model_rf.pkl")
    return vectorizer, model_xgb, model_rf

df = load_data()
vectorizer, model_xgb, model_rf = load_models()

# =====================================================
# SIDEBAR NAVIGASI
# =====================================================
st.sidebar.title("📌 Menu Dashboard")

menu = st.sidebar.radio(
    "Pilih Halaman:",
    [
        "Overview",
        "Performa Model",
        "Confusion Matrix",
        "Word Cloud",
        "Data Ulasan",
        "Klasifikasi Ulasan Baru"
    ]
)

# =====================================================
# HALAMAN OVERVIEW
# =====================================================
if menu == "Overview":
    st.title("📊 Overview Analisis Sentimen")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Ulasan Dianalisis", f"{len(df)} Data")

    with col2:
        sentiment_counts = df["sentimen"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_counts.index,
               autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

# =====================================================
# HALAMAN PERFORMA MODEL
# =====================================================
elif menu == "Performa Model":
    st.title("📊 Perbandingan Performa Model")

    model_metrics = pd.DataFrame({
        "Model": ["XGBoost", "Random Forest"],
        "Akurasi": [0.87, 0.84],
        "Presisi": [0.86, 0.83],
        "Recall": [0.85, 0.82],
        "F1-Score": [0.85, 0.82]
    })

    metric = st.selectbox(
        "Pilih Metrik Evaluasi:",
        ["Akurasi", "Presisi", "Recall", "F1-Score"]
    )

    fig, ax = plt.subplots()
    ax.bar(model_metrics["Model"], model_metrics[metric])
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric)
    ax.set_title(f"Perbandingan {metric}")
    st.pyplot(fig)

# =====================================================
# HALAMAN CONFUSION MATRIX
# =====================================================
elif menu == "Confusion Matrix":
    st.title("📉 Confusion Matrix")

    model_choice = st.selectbox(
        "Pilih Model:",
        ["XGBoost", "Random Forest"]
    )

    y_true = df["sentimen_encoded"]
    X_tfidf = vectorizer.transform(df["content"].astype(str))

    y_pred = (
        model_xgb.predict(X_tfidf)
        if model_choice == "XGBoost"
        else model_rf.predict(X_tfidf)
    )

    labels = [2, 1, 0]
    label_names = ["Puas", "Netral", "Tidak Puas"]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    st.pyplot(fig)

    st.subheader("Tabel Confusion Matrix")
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("Evaluasi Model")
    report = classification_report(
        y_true, y_pred, target_names=label_names, output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(3), use_container_width=True)

    st.subheader("Perbandingan XGBoost vs Random Forest")

    y_pred_xgb = model_xgb.predict(X_tfidf)
    y_pred_rf = model_rf.predict(X_tfidf)

    comparison_df = pd.DataFrame({
        "Model": ["XGBoost", "Random Forest"],
        "Akurasi": [
            accuracy_score(y_true, y_pred_xgb),
            accuracy_score(y_true, y_pred_rf)
        ],
        "F1 Macro": [
            classification_report(y_true, y_pred_xgb, output_dict=True)["macro avg"]["f1-score"],
            classification_report(y_true, y_pred_rf, output_dict=True)["macro avg"]["f1-score"]
        ],
        "F1 Weighted": [
            classification_report(y_true, y_pred_xgb, output_dict=True)["weighted avg"]["f1-score"],
            classification_report(y_true, y_pred_rf, output_dict=True)["weighted avg"]["f1-score"]
        ],
    })

    st.dataframe(comparison_df.round(3), use_container_width=True)

# =====================================================
# HALAMAN WORD CLOUD
# =====================================================
elif menu == "Word Cloud":
    st.title("☁️ Word Cloud Berdasarkan Sentimen")

    sentiment_option = st.selectbox(
        "Pilih Sentimen:",
        df["sentimen"].unique()
    )

    text_data = " ".join(
        df[df["sentimen"] == sentiment_option]["content"].astype(str)
    )

    if text_data.strip():
        wc = WordCloud(width=800, height=400, background_color="white")
        wc.generate(text_data)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

# =====================================================
# HALAMAN DATA ULASAN
# =====================================================
elif menu == "Data Ulasan":
    st.title("📋 Data Ulasan Terklasifikasi")

    filter_sentiment = st.selectbox(
        "Filter Sentimen:",
        ["Semua"] + list(df["sentimen"].unique())
    )

    df_show = df if filter_sentiment == "Semua" else df[df["sentimen"] == filter_sentiment]

    st.dataframe(
        df_show[["content", "score", "sentimen"]],
        use_container_width=True
    )

# =====================================================
# HALAMAN KLASIFIKASI ULASAN BARU
# =====================================================
elif menu == "Klasifikasi Ulasan Baru":
    st.title("🧠 Klasifikasi Ulasan Baru")

    st.markdown(
        "Fitur ini digunakan untuk mengklasifikasikan ulasan baru "
        "ke dalam kategori **Puas, Netral, atau Tidak Puas**."
    )

    model_choice = st.selectbox(
        "Pilih Model:",
        ["XGBoost", "Random Forest"]
    )

    user_review = st.text_area(
        "Masukkan Teks Ulasan:",
        placeholder="Contoh: Driver ramah dan aplikasi mudah digunakan"
    )

    if st.button("🔍 Klasifikasikan"):
        if user_review.strip() == "":
            st.warning("Masukkan teks ulasan terlebih dahulu.")
        else:
            review_tfidf = vectorizer.transform([user_review])

            prediction = (
                model_xgb.predict(review_tfidf)[0]
                if model_choice == "XGBoost"
                else model_rf.predict(review_tfidf)[0]
            )

            label_map = {2: "Puas 😊", 1: "Netral 😐", 0: "Tidak Puas 😠"}
            st.success(f"Hasil Klasifikasi: **{label_map[prediction]}**")

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown(
    "<center>Dashboard Analisis Sentimen | Skripsi | 2026</center>",
    unsafe_allow_html=True
)
