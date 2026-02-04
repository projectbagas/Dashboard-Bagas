# app_final.py - DASHBOARD MAXIM (WORK DI STREAMLIT CLOUD)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Cek plotly (fallback jika gagal)
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("📊 Plotly tidak tersedia, menggunakan Matplotlib")

st.set_page_config(page_title="Dashboard Maxim", page_icon="📱", layout="wide")

# CSS styling
st.markdown("""
<style>
.metric-card {background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); 
              padding: 1.5rem; border-radius: 15px; border-left: 6px solid #10B981;}
.main-title {font-size: 2.5rem !important; color: #10B981; text-align: center;}
</style>
""", unsafe_allow_html=True)

# DATA SAMPLE 2000 ULASAN MAXIM
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 2000
    labels = np.random.choice(['puas', 'netral', 'tidak puas'], n, p=[0.45, 0.30, 0.25])
    
    reviews = [
        'bagus cepat murah recommended mantap oke nyaman driver ramah',
        'error lemot hang sering crash aplikasi jelek buruk',
        'lumayan biasa aja kadang lag standar',
        'mantap bos cepat sampai recommended mantap',
        'penipuan mahal banget aplikasi scam lemot'
    ] * (n//5)
    
    return pd.DataFrame({
        'review_text': reviews[:n],
        'rating': np.random.randint(1, 6, n),
        'label': labels,
        'date': pd.date_range('2025-01-01', periods=n, freq='D')
    })

df = load_data()

# HASIL MODEL SKRIPSI
MODEL_METRICS = {
    'XGBoost': [0.92, 0.91, 0.92, 0.91],
    'Random Forest': [0.89, 0.88, 0.89, 0.88]
}
metrics_names = ['Akurasi', 'Precision', 'Recall', 'F1-Score']

# HEADER
st.markdown('<h1 class="main-title">📊 Dashboard Analisis Kepuasan Maxim</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="text-align: center; color: #1F2937;">Perbandingan XGBoost vs Random Forest - Google Play Store</h2>', unsafe_allow_html=True)

# SIDEBAR
page = st.sidebar.radio("📋 Navigasi:", ["📈 Overview", "⚔️ Model", "☁️ Word Cloud", "📋 Ulasan"])

# 1. OVERVIEW
if page == "📈 Overview":
    col1, col2, col3, col4 = st.columns(4)
    total = len(df)
    puas = len(df[df['label']=='puas'])
    netral = len(df[df['label']=='netral'])
    tidak_puas = len(df[df['label']=='tidak puas'])
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Ulasan", f"{total:,}", "✅ Teranalisis")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2: st.markdown('<div class="metric-card">', unsafe_allow_html=True); st.metric("Puas 👍", f"{puas:,}", f"{puas/total*100:.1f}%"); st.markdown('</div>', unsafe_allow_html=True)
    with col3: st.markdown('<div class="metric-card">', unsafe_allow_html=True); st.metric("Netral ➡️", f"{netral:,}", f"{netral/total*100:.1f}%"); st.markdown('</div>', unsafe_allow_html=True)
    with col4: st.markdown('<div class="metric-card">', unsafe_allow_html=True); st.metric("Tidak Puas 👎", f"{tidak_puas:,}", f"{tidak_puas/total*100:.1f}%"); st.markdown('</div>', unsafe_allow_html=True)
    
    # PIE CHART (Plotly atau Matplotlib)
    col1, col2 = st.columns([3,1])
    with col1:
        if PLOTLY_AVAILABLE:
            fig = px.pie(values=[puas, netral, tidak_puas], 
                        names=['Puas 👍', 'Netral ➡️', 'Tidak Puas 👎'],
                        color_discrete_sequence=['#10B981', '#F59E0B', '#EF4444'], hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(8,8))
            ax.pie([puas, netral, tidak_puas], labels=['Puas', 'Netral', 'Tidak Puas'],
                   colors=['#10B981', '#F59E0B', '#EF4444'], autopct='%1.1f%%')
            ax.set_title('Distribusi Kepuasan', fontsize=16, color='#10B981')
            st.pyplot(fig)

# 2. PERBANDINGAN MODEL
elif page == "⚔️ Model":
    st.subheader("📊 Perbandingan Metrik Evaluasi")
    
    # BAR CHART
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(x - width/2, MODEL_METRICS['XGBoost'], width, label='XGBoost', color='#10B981', alpha=0.8)
    ax.bar(x + width/2, MODEL_METRICS['Random Forest'], width, label='Random Forest', color='#3B82F6', alpha=0.8)
    
    ax.set_xlabel('Metrik'); ax.set_ylabel('Score'); ax.set_title('XGBoost vs Random Forest')
    ax.set_xticks(x); ax.set_xticklabels(metrics_names)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    # TABEL
    metrics_df = pd.DataFrame(MODEL_METRICS).T
    metrics_df['Metrik'] = metrics_names
    st.dataframe(metrics_df.round(3), use_container_width=True)

# 3. WORDCLOUD
elif page == "☁️ Word Cloud":
    st.subheader("☁️ Kata Kunci Ulasan")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        text_puas = ' '.join(df[df['label']=='puas']['review_text'].dropna())
        wc = WordCloud(width=400, height=300, colormap='Greens', background_color='white').generate(text_puas)
        plt.figure(figsize=(6,4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title('PUAS 👍', color='#10B981', fontsize=16)
        st.pyplot(plt.gcf())
    
    with col2:
        text_netral = ' '.join(df[df['label']=='netral']['review_text'].dropna())
        wc = WordCloud(width=400, height=300, colormap='Oranges', background_color='white').generate(text_netral)
        plt.figure(figsize=(6,4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title('NETRAL ➡️', color='#F59E0B', fontsize=16)
        st.pyplot(plt.gcf())
    
    with col3:
        text_negatif = ' '.join(df[df['label']=='tidak puas']['review_text'].dropna())
        wc = WordCloud(width=400, height=300, colormap='Reds', background_color='white').generate(text_negatif)
        plt.figure(figsize=(6,4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title('TIDAK PUAS 👎', color='#EF4444', fontsize=16)
        st.pyplot(plt.gcf())

# 4. ULASAN
elif page == "📋 Ulasan":
    st.subheader("📋 Daftar Ulasan Terklasifikasi")
    kategori = st.multiselect("Filter Kategori:", df['label'].unique(), default=df['label'].unique())
    df_show = df[df['label'].isin(kategori)].sort_values('rating', ascending=False)
    
    st.dataframe(df_show[['review_text', 'label', 'rating']].head(50), 
                column_config={
                    "review_text": st.column_config.TextColumn("Ulasan", width="medium"),
                    "rating": st.column_config.NumberColumn("⭐ Rating", format="%.1f")
                }, height=500, use_container_width=True)

st.markdown("---")
st.markdown("*Dashboard Skripsi - Universitas Nasional 2026 | 2000+ Ulasan Maxim Play Store*")
