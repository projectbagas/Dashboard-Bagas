# app_fixed.py - DASHBOARD ANTI-ERROR (TANPA PICKLE)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Dashboard Maxim", page_icon="📱", layout="wide")

# CSS styling
st.markdown("""
<style>
.main-header {font-size: 2.5rem !important; color: #10B981; text-align: center;}
.metric-card {background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); 
              padding: 1.5rem; border-radius: 15px; border-left: 6px solid #10B981;}
</style>
""", unsafe_allow_html=True)

# DATA DUMMY REALISTIS (2000 ulasan Maxim)
@st.cache_data
def load_sample_data():
    np.random.seed(42)
    labels = np.random.choice(['puas', 'netral', 'tidak puas'], 2000, p=[0.45, 0.30, 0.25])
    
    data = {
        'review_text': [
            'bagus cepat murah recommended mantap oke nyaman',
            'error lemot hang sering crash aplikasi jelek',
            'lumayan biasa aja kadang lag',
            'mantap bos cepat sampai driver ramah',
            'penipuan mahal banget aplikasi scam'
        ] * 400,
        'rating': np.random.randint(1, 6, 2000),
        'label': labels,
        'date': pd.date_range('2025-01-01', periods=2000, freq='D')
    }
    return pd.DataFrame(data)

df = load_sample_data()

# METRICS HASIL MODEL (dari skripsi Anda)
MODEL_RESULTS = {
    'XGBoost': {'Akurasi': 0.92, 'Precision': 0.91, 'Recall': 0.92, 'F1': 0.91},
    'Random Forest': {'Akurasi': 0.89, 'Precision': 0.88, 'Recall': 0.89, 'F1': 0.88}
}

# Header
st.markdown('<h1 class="main-header">📊 Dashboard Analisis Kepuasan Maxim</h1>', unsafe_allow_html=True)
st.markdown('<h2>Perbandingan XGBoost vs Random Forest - Google Play Store</h2>', unsafe_allow_html=True)

# Sidebar
page = st.sidebar.radio("Navigasi:", ["📈 Overview", "⚔️ Perbandingan Model", "☁️ Word Cloud", "📋 Ulasan"])

# 1. OVERVIEW
if page == "📈 Overview":
    col1, col2, col3, col4 = st.columns(4)
    total, puas, netral, tidak_puas = len(df), len(df[df['label']=='puas']), len(df[df['label']=='netral']), len(df[df['label']=='tidak puas'])
    
    with col1: st.markdown('<div class="metric-card">', unsafe_allow_html=True); st.metric("Total Ulasan", f"{total:,}"); st.markdown('</div>', unsafe_allow_html=True)
    with col2: st.markdown('<div class="metric-card">', unsafe_allow_html=True); st.metric("Puas 👍", f"{puas:,}", f"{puas/total*100:.1f}%"); st.markdown('</div>', unsafe_allow_html=True)
    with col3: st.markdown('<div class="metric-card">', unsafe_allow_html=True); st.metric("Netral ➡️", f"{netral:,}", f"{netral/total*100:.1f}%"); st.markdown('</div>', unsafe_allow_html=True)
    with col4: st.markdown('<div class="metric-card">', unsafe_allow_html=True); st.metric("Tidak Puas 👎", f"{tidak_puas:,}", f"{tidak_puas/total*100:.1f}%"); st.markdown('</div>', unsafe_allow_html=True)
    
    # PIE CHART
    fig = px.pie(values=[puas, netral, tidak_puas], names=['Puas', 'Netral', 'Tidak Puas'], 
                 color_discrete_sequence=['#10B981', '#F59E0B', '#EF4444'], hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

# 2. PERBANDINGAN MODEL
elif page == "⚔️ Perbandingan Model":
    metrics_df = pd.DataFrame(MODEL_RESULTS).T.reset_index()
    fig = px.bar(metrics_df, x='index', y=['Akurasi', 'Precision', 'Recall', 'F1'],
                title="Perbandingan Metrik Evaluasi", barmode='group',
                color_discrete_sequence=['#10B981', '#3B82F6'])
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(metrics_df.round(3))

# 3. WORDCLOUD
elif page == "☁️ Word Cloud":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        text_puas = ' '.join(df[df['label']=='puas']['review_text'])
        wc = WordCloud(colormap='Greens', width=400, height=300).generate(text_puas)
        fig, ax = plt.subplots(); ax.imshow(wc); ax.axis('off'); ax.set_title('PUAS 👍', color='#10B981')
        st.pyplot(fig)
    
    with col2:
        text_netral = ' '.join(df[df['label']=='netral']['review_text'])
        wc = WordCloud(colormap='Oranges', width=400, height=300).generate(text_netral)
        fig, ax = plt.subplots(); ax.imshow(wc); ax.axis('off'); ax.set_title('NETRAL ➡️', color='#F59E0B')
        st.pyplot(fig)
    
    with col3:
        text_negatif = ' '.join(df[df['label']=='tidak puas']['review_text'])
        wc = WordCloud(colormap='Reds', width=400, height=300).generate(text_negatif)
        fig, ax = plt.subplots(); ax.imshow(wc); ax.axis('off'); ax.set_title('TIDAK PUAS 👎', color='#EF4444')
        st.pyplot(fig)

# 4. ULASAN
elif page == "📋 Ulasan":
    kategori = st.multiselect("Filter:", df['label'].unique())
    df_filtered = df[df['label'].isin(kategori)]
    st.dataframe(df_filtered[['review_text', 'label', 'rating']], use_container_width=True)

st.markdown("---")
st.markdown("*Dashboard Skripsi Maxim - Universitas Nasional 2026*")
