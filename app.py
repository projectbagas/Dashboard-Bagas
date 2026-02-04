import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Dashboard Analisis Sentimen Ulasan Aplikasi Maxim",
    layout="wide"
)

st.title("📊 Dashboard Analisis Sentimen Ulasan Aplikasi Maxim")
st.markdown(
    "Perbandingan Algoritma **XGBoost** dan **Random Forest** "
    "dalam Klasifikasi Tingkat Kepuasan Pengguna Google Play Store"
)

# ===============================
# LOAD DATA & MODEL
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("maxim_reviews_labeled.csv")

@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

data = load_data()
vectorizer = load_model("tfidf_vectorizer.pkl")
model_xgb = load_model("model_xgb.pkl")
model_rf = load_model("model_rf.pkl")

# ===============================
# PREPROCESS & PREDIKSI
# ===============================
X = vectorizer.transform(data["review"])
y_true = data["label"]

y_pred_xgb = model_xgb.predict(X)
y_pred_rf = model_rf.predict(X)

labels = ["Tidak Puas", "Netral", "Puas"]

# ===============================
# SIDEBAR NAVIGASI
# ===============================
menu = st.sidebar.radio(
    "Menu Dashboard",
    [
        "Overview",
        "Performa Model",
        "Confusion Matrix",
        "Word Cloud",
        "Data Ulasan"
    ]
)

# ===============================
# OVERVIEW
# ===============================
if menu == "Overview":
    st.subheader("📌 Ringkasan Statistik")

    col1, col2 = st.columns(2)
    col1.metric("Total Ulasan", len(data))
    col2.metric("Jumlah Kelas Sentimen", data["label"].nunique())

    sentiment_count = data["label"].value_counts()

    fig, ax = plt.subplots()
    ax.pie(
        sentiment_count,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90
    )
    ax.set_title("Distribusi Tingkat Kepuasan Pengguna")
    st.pyplot(fig)

# ===============================
# PERFORMA MODEL
# ===============================
elif menu == "Performa Model":
    st.subheader("📊 Perbandingan Performa Model")

    metrics = {
        "Akurasi": [
            accuracy_score(y_true, y_pred_xgb),
            accuracy_score(y_true, y_pred_rf)
        ],
        "Presisi": [
            precision_score(y_true, y_pred_xgb, average="weighted"),
            precision_score(y_true, y_pred_rf, average="weighted")
        ],
        "Recall": [
            recall_score(y_true, y_pred_xgb, average="weighted"),
            recall_score(y_true, y_pred_rf, average="weighted")
        ],
        "F1-Score": [
            f1_score(y_true, y_pred_xgb, average="weighted"),
            f1_score(y_true, y_pred_rf, average="weighted")
        ],
    }

    df_metrics = pd.DataFrame(metrics, index=["XGBoost", "Random Forest"])

    st.dataframe(df_metrics)

    df_metrics.plot(kind="bar")
    plt.title("Perbandingan Metrik Evaluasi Model")
    plt.ylabel("Nilai")
    plt.xticks(rotation=0)
    st.pyplot(plt)

# ===============================
# CONFUSION MATRIX (PAKAI ANGKA)
# ===============================
elif menu == "Confusion Matrix":
    st.subheader("🧩 Confusion Matrix")

    def plot_cm(y_true, y_pred, title):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        ax.imshow(cm)

        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(title)

        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        plot_cm(y_true, y_pred_xgb, "Confusion Matrix XGBoost")
    with col2:
        plot_cm(y_true, y_pred_rf, "Confusion Matrix Random Forest")

# ===============================
# WORD CLOUD
# ===============================
elif menu == "Word Cloud":
    st.subheader("☁️ Analisis Kata (Word Cloud)")

    from wordcloud import WordCloud

    sentiment = st.selectbox(
        "Pilih Kategori Sentimen",
        labels
    )

    text = " ".join(
        data[data["label"] == labels.index(sentiment)]["review"]
    )

    wc = WordCloud(
        width=800,
        height=400,
        background_color="black"
    ).generate(text)

    fig, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig)

# ===============================
# DATA ULASAN
# ===============================
elif menu == "Data Ulasan":
    st.subheader("📄 Data Ulasan Terklasifikasi")

    data_display = data.copy()
    data_display["Prediksi XGBoost"] = y_pred_xgb
    data_display["Prediksi RF"] = y_pred_rf

    st.dataframe(data_display)
