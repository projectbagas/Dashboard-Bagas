# Dashboard Analisis Sentimen Ulasan Aplikasi Maxim

Dashboard ini dikembangkan sebagai bagian dari penelitian skripsi yang bertujuan
untuk menganalisis sentimen ulasan pengguna aplikasi Maxim berdasarkan data
ulasan di Google Play Store.

Analisis sentimen dilakukan menggunakan dua algoritma machine learning, yaitu
**XGBoost** dan **Random Forest**, dengan pendekatan pembobotan kata menggunakan
**TF-IDF**.

---

## 🎯 Tujuan Dashboard
Dashboard ini bertujuan untuk:
- Menyajikan ringkasan statistik sentimen ulasan pengguna
- Membandingkan performa algoritma XGBoost dan Random Forest
- Menampilkan confusion matrix sebagai evaluasi model
- Mengidentifikasi kata-kata dominan pada setiap kategori sentimen
- Menyediakan tabel ulasan terklasifikasi untuk validasi visual

---

## 📊 Fitur Dashboard
1. **Ringkasan Statistik**
   - Total ulasan yang dianalisis
   - Distribusi sentimen (Puas, Netral, Tidak Puas)

2. **Perbandingan Performa Model**
   - Grafik batang perbandingan Akurasi, Presisi, Recall, dan F1-Score

3. **Confusion Matrix**
   - Evaluasi performa klasifikasi untuk XGBoost dan Random Forest

4. **Word Cloud**
   - Visualisasi kata dominan berdasarkan kategori sentimen

5. **Tabel Ulasan Terklasifikasi**
   - Menampilkan ulasan asli, rating bintang, dan hasil klasifikasi sentimen

