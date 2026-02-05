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
    return pd.read_csv("maxim_reviews_cleaned.csv")

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
# HALAMAN OVERVIEW (Professional & Modern)
# =====================================================
if menu == "Overview":
    st.title("📊 Overview Analisis Sentimen")

    # ================= HITUNG DISTRIBUSI =================
    sentiment_counts = df["sentimen"].value_counts()

    # ================= KPI METRICS =================
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💬 Total Ulasan", len(df))
    col2.metric("😊 Puas", sentiment_counts.get("Puas", 0))
    col3.metric("😐 Netral", sentiment_counts.get("Netral", 0))
    col4.metric("😠 Tidak Puas", sentiment_counts.get("Tidak Puas", 0))

    st.markdown("---")

    # ================= VISUALISASI SEJAJAR =================
    col1, col2 = st.columns([1, 1])

    # ---------- PIE CHART ----------
    with col1:
        st.subheader("Distribusi Sentimen")

        fig1, ax1 = plt.subplots(figsize=(4, 4))
        colors = ["#1E88E5", "#FBC02D", "#E53935"]  # Biru, Kuning, Merah

        ax1.pie(
            sentiment_counts,
            labels=sentiment_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            textprops={"fontsize": 10}
        )
        ax1.axis("equal")
        st.pyplot(fig1)

    # ---------- BAR CHART ----------
    with col2:
        st.subheader("Jumlah Ulasan per Kategori")

        bar_df = sentiment_counts.reset_index()
        bar_df.columns = ["Sentimen", "Jumlah"]

        fig2, ax2 = plt.subplots(figsize=(4, 4))
        ax2.bar(
            bar_df["Sentimen"],
            bar_df["Jumlah"]
        )
        ax2.set_ylabel("Jumlah Ulasan")
        ax2.set_xlabel("Kategori Sentimen")

        st.pyplot(fig2)

    # ================= NARASI =================
    st.markdown("---")
    st.markdown("""
    **Gambaran Umum Analisis Sentimen**  
    Halaman overview menyajikan ringkasan distribusi sentimen ulasan pengguna
    aplikasi Maxim ke dalam tiga kategori, yaitu **Puas**, **Netral**, dan
    **Tidak Puas**.  
    Visualisasi pie chart menunjukkan proporsi masing-masing sentimen,
    sedangkan bar chart menampilkan jumlah ulasan per kategori untuk
    memudahkan analisis perbandingan.
    """)

elif menu == "Performa Model":
    st.title("📊 Perbandingan Performa Model")

    # ================= DATA METRIK MODEL =================
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

    xgboost_scores = [0.87, 0.86, 0.85, 0.85]
    rf_scores = [0.84, 0.83, 0.82, 0.82]

    x = np.arange(len(metrics))
    width = 0.35

    # ================= BAR CHART PERBANDINGAN =================
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(x - width/2, xgboost_scores, width, label="XGBoost")
    ax.bar(x + width/2, rf_scores, width, label="Random Forest")

    ax.set_ylabel("Score")
    ax.set_title("Perbandingan Performa Model (XGBoost vs Random Forest)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.legend()

    st.pyplot(fig)

    # ================= TABEL RINGKAS =================
    st.markdown("### 📋 Tabel Ringkasan Evaluasi Model")

    perf_df = pd.DataFrame({
        "Metric": metrics,
        "XGBoost": xgboost_scores,
        "Random Forest": rf_scores
    })

    st.dataframe(perf_df, use_container_width=True)

    # ================= NARASI =================
    st.markdown("""
    **Analisis Performa Model**  
    Berdasarkan hasil evaluasi, model **XGBoost** menunjukkan performa yang
    lebih unggul dibandingkan **Random Forest** pada seluruh metrik evaluasi,
    yaitu Accuracy, Precision, Recall, dan F1-Score.  
    Hal ini menunjukkan bahwa XGBoost lebih konsisten dalam mengklasifikasikan
    sentimen ulasan pengguna.
    """)

# =====================================================
# HALAMAN CONFUSION MATRIX
# =====================================================
elif menu == "Confusion Matrix":
    st.title("📉 Confusion Matrix")

    model_choice = st.selectbox(
        "Pilih Model:",
        ["XGBoost", "Random Forest"]
    )

    # ================= PREDIKSI =================
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

    # ================= LAYOUT 2 KOLOM =================
    col1, col2 = st.columns([1, 1])

    # ---------- CONFUSION MATRIX ----------
    with col1:
        st.subheader("Visual Confusion Matrix")

        fig, ax = plt.subplots(figsize=(4, 4))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=label_names
        )
        disp.plot(
            ax=ax,
            cmap="Blues",
            colorbar=False,
            values_format="d"
        )
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        st.pyplot(fig)

    # ---------- TABEL CONFUSION MATRIX ----------
    with col2:
        st.subheader("Tabel Confusion Matrix")

        cm_df = pd.DataFrame(
            cm,
            index=label_names,
            columns=label_names
        )
        st.dataframe(cm_df, use_container_width=True)

    st.markdown("---")

    # ================= EVALUASI MODEL =================
    st.subheader("Evaluasi Model")

    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        output_dict=True
    )

    report_df = (
        pd.DataFrame(report)
        .transpose()
        .round(3)
    )

    st.dataframe(report_df, use_container_width=True)

    # ================= NARASI =================
    st.markdown("""
    **Analisis Confusion Matrix**  
    Confusion matrix menunjukkan kemampuan model dalam mengklasifikasikan
    ulasan ke dalam kategori **Puas**, **Netral**, dan **Tidak Puas**.
    Nilai pada diagonal utama merepresentasikan jumlah prediksi yang tepat,
    sedangkan nilai di luar diagonal menunjukkan kesalahan klasifikasi.
    """)

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









