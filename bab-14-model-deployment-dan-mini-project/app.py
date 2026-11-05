"""
Aplikasi Web: Prediksi Risiko Gagal Bayar Pinjaman (Loan Default Prediction)
Laboratorium Machine Learning - Universitas Indraprasta PGRI (Unindra)
Mata Kuliah: Machine Learning (Bab 14: Deployment & Mini Project)
Mahasiswa: Anggit Arisa Siwi Pratama (NPM: 202343570045)
Dosen Pengampu: Nurfidah Dwitiyanti, M.Si.
"""

import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─── 1. Konfigurasi Halaman & Tema ──────────────────────────────────────────
st.set_page_config(
    page_title="Unindra Credit Risk Analyzer | Bab 14",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0284C7;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── 2. Pemuatan Model Terlatih (Joblib Artifact) ────────────────────────────
CURRENT_DIR = Path(__file__).resolve().parent
MODEL_PATH = CURRENT_DIR / "models" / "model_loan_default_best.joblib"

@st.cache_resource
def load_trained_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    # Fallback dummy pipeline jika dijalankan terisolasi
    from sklearn.ensemble import RandomForestClassifier
    dummy = RandomForestClassifier(n_estimators=20, random_state=42)
    X_d = np.array([[5000000, 30, 3, 30000000, 0.2, 650], [15000000, 45, 10, 50000000, 0.45, 550]])
    y_d = np.array([0, 1])
    dummy.fit(X_d, y_d)
    return dummy

model = load_trained_model()

# ─── 3. Header & Informasi Akademik ─────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="main-header">Sistem Analisis Risiko Kredit dan Kelayakan Pinjaman</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Prototipe Web Deployment Model Machine Learning (Sesuai Sub-bab 14.14 Modul Unindra)</div>', unsafe_allow_html=True)

with col_h2:
    st.info("**Peneliti / Mahasiswa:**\nAnggit Arisa Siwi Pratama\nNPM: `202343570045` (XC Ekstensi)")

# ─── 4. Sidebar Form Input Nasabah ──────────────────────────────────────────
st.sidebar.header("Profil Calon Peminjam")
st.sidebar.markdown("Masukkan data calon nasabah perbankan:")

pendapatan_bulanan = st.sidebar.number_input(
    "Pendapatan Bulanan (Rp)",
    min_value=3_000_000,
    max_value=100_000_000,
    value=8_500_000,
    step=500_000,
    format="%d"
)

usia = st.sidebar.slider("Usia Peminjam (Tahun)", min_value=21, max_value=65, value=29)
lama_kerja = st.sidebar.slider("Lama Bekerja / Usaha (Tahun)", min_value=0, max_value=35, value=4)

jumlah_pinjaman = st.sidebar.number_input(
    "Jumlah Plafon Pinjaman Diajukan (Rp)",
    min_value=5_000_000,
    max_value=300_000_000,
    value=35_000_000,
    step=5_000_000,
    format="%d"
)

tenor_bulan = st.sidebar.selectbox("Tenor Pelunasan (Bulan)", options=[12, 24, 36, 48, 60], index=1)
skor_kredit = st.sidebar.slider("Skor Kredit Riwayat (BI Checking / SLIK)", min_value=350, max_value=850, value=675)

# Kalkulasi rasio beban cicilan (Debt Service Ratio)
estimasi_cicilan = jumlah_pinjaman / tenor_bulan
rasio_cicilan = round(float(estimasi_cicilan / pendapatan_bulanan), 3)

st.sidebar.markdown("---")
st.sidebar.metric(
    label="Estimasi Cicilan Bulanan",
    value=f"Rp {estimasi_cicilan:,.0f}".replace(",", "."),
    delta=f"Rasio: {rasio_cicilan*100:.1f}% Pendapatan",
    delta_color="inverse" if rasio_cicilan > 0.35 else "normal"
)

# ─── 5. Inferensi Prediksi ───────────────────────────────────────────────────
input_data = pd.DataFrame([[
    pendapatan_bulanan,
    usia,
    lama_kerja,
    jumlah_pinjaman,
    rasio_cicilan,
    skor_kredit
]], columns=["pendapatan_bulanan", "usia", "lama_kerja", "jumlah_pinjaman", "rasio_cicilan", "skor_kredit"])

st.subheader("Evaluasi Keputusan Komite Kredit")

col_btn, col_blank = st.columns([1, 4])
with col_btn:
    btn_predict = st.button("Jalankan Analisis Model", type="primary", use_container_width=True)

if btn_predict or "has_run" in st.session_state:
    st.session_state["has_run"] = True
    
    # Eksekusi inferensi dari model joblib
    prediksi_default = int(model.predict(input_data)[0])
    prob_default = float(model.predict_proba(input_data)[0][1])
    prob_lancar = 1.0 - prob_default
    
    col_res1, col_res2, col_res3 = st.columns([1.2, 1.2, 1.6])
    
    with col_res1:
        if prediksi_default == 0:
            st.success("### Risiko Rendah (Lolos)")
            st.caption("Nasabah memiliki probabilitas pelunasan yang baik dan profil keuangan stabil.")
        else:
            st.error("### Risiko Tinggi (Default)")
            st.caption("Nasabah terindikasi memiliki potensi kegagalan pembayaran kredit.")
            
    with col_res2:
        st.metric(
            label="Probabilitas Gagal Bayar (Default)",
            value=f"{prob_default * 100:.1f}%",
            delta=f"Tingkat Keyakinan: {max(prob_default, prob_lancar)*100:.1f}%"
        )
        st.progress(prob_default)
        
    with col_res3:
        st.markdown("**Pedoman Mitigasi Komite:**")
        if prob_default < 0.20:
            st.write("**Persetujuan Otomatis:** Ajukan plafon maksimal, tenor fleksibel.")
        elif prob_default < 0.40:
            st.write("**Persetujuan Bersyarat:** Memerlukan agunan tambahan atau pengurangan plafon 15%.")
        else:
            st.write("**Penolakan Rekomendasi:** Tingkat rasio cicilan atau riwayat skor kredit tidak memenuhi kriteria mitigasi risiko bank.")

    st.markdown("---")
    
    # Ringkasan Parameter Input
    st.markdown("#### Parameter Verifikasi Fitur")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pendapatan", f"Rp {pendapatan_bulanan:,.0f}".replace(",", "."))
    c2.metric("Plafon Pengajuan", f"Rp {jumlah_pinjaman:,.0f}".replace(",", "."))
    c3.metric("Rasio Beban Cicilan", f"{rasio_cicilan * 100:.1f}%")
    c4.metric("Skor Kredit SLIK", f"{skor_kredit}")

# ─── 6. Footer & Catatan Akademik ───────────────────────────────────────────
st.markdown("---")
st.caption("""
**Catatan Akademik**: Sistem inferensi ini merupakan luaran Praktikum Bab 14 Mata Kuliah Machine Learning, Universitas Indraprasta PGRI (Unindra). 
Model dibangun menggunakan ensemble Random Forest dengan pembobotan *multi-criteria decision* antara metrik F1-Score dan kemampuan interpretasi model.
""")

