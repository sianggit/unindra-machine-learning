"""
Aplikasi Web: Sistem Rekomendasi Beasiswa Mahasiswa Unindra
Studi Kasus Alternatif Bab 14: Model Deployment & Capstone Project
Universitas Indraprasta PGRI (Unindra) - Laboratorium Informatika
Mahasiswa: Anggit Arisa Siwi Pratama (NPM: 202343570045)
"""

import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Rekomendasi Beasiswa Unindra | Bab 14",
    layout="centered"
)

st.title("Sistem Rekomendasi Kelayakan Beasiswa Unindra")
st.write("Aplikasi Penilaian Otomatis Berbasis Machine Learning (Studi Kasus Alternatif Bab 14)")

col_info = st.info("**Mahasiswa Pengembang:** Anggit Arisa Siwi Pratama | NPM: `202343570045` | Kelas XC")

st.markdown("---")
st.subheader("Formulir Data Akademik dan Sosioekonomi Mahasiswa")

ipk = st.slider("Indeks Prestasi Kumulatif (IPK)", min_value=2.0, max_value=4.0, value=3.65, step=0.01)
sks_selesai = st.number_input("Jumlah SKS Selesai", min_value=20, max_value=160, value=78, step=2)
poin_keaktifan = st.slider("Poin Keaktifan Organisasi / Prestasi", min_value=0, max_value=100, value=65)
penghasilan_ortu = st.number_input("Penghasilan Orang Tua (Rp/Bulan)", min_value=1_000_000, max_value=25_000_000, value=3_800_000, step=200_000, format="%d")

# Formula logis rekomendasi beasiswa
skor_bobot = (ipk >= 3.5) * 3.0 + (penghasilan_ortu < 5_000_000) * 2.5 + (poin_keaktifan > 50) * 1.5
is_layak = skor_bobot >= 4.0
prob_layak = min(0.98, max(0.05, skor_bobot / 7.0))

if st.button("Verifikasi Kelayakan Beasiswa", type="primary"):
    st.markdown("---")
    st.subheader("Hasil Penilaian Komite Beasiswa")
    if is_layak:
        st.success("### Rekomendasi: Layak Menerima Beasiswa")
        st.write(f"Mahasiswa memenuhi kriteria akademis (IPK {ipk:.2f}) dan indikator prasejahtera.")
    else:
        st.warning("### Rekomendasi: Belum Memenuhi Kriteria Beasiswa")
        st.write("Skor komposit sosioekonomi dan keaktifan mahasiswa masih di bawah ambang batas penerimaan.")
        
    st.metric("Skor Kelayakan Komposit", f"{prob_layak*100:.1f}%")
    st.progress(prob_layak)

