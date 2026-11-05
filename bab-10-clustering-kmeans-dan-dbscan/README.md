# Bab 10 - Clustering: K-Means dan DBSCAN

> **Portofolio Tugas dan Praktikum Machine Learning Unindra**  
> Mahasiswa: **Anggit Arisa Siwi Pratama** | NPM: **`202343570045`** | Kelas: **XC Ekstensi**  
> Dosen: **Nurfidah Dwitiyanti, M.Si.** | Semester Gasal 2026/2027  

---

## Ringkasan Teknis

- **Topik Utama**: Pembelajaran Tidak Terawasi (*Unsupervised Learning*), K-Means Clustering, Evaluasi Elbow Method (Inersia), Silhouette Coefficient, Calinski-Harabasz Index, dan Density-Based Clustering (DBSCAN).
- **Dataset**: `Spotify Track Audio Features (danceability, energy, acousticness, valence, tempo, N=800)`
- **Jadwal Pelaksanaan**: 07 November 2026

## Struktur Berkas

```text
bab-10-clustering-kmeans-dan-dbscan/
├── main.py                                      # Skrip executable Python mandiri
├── 10_clustering_kmeans_dan_dbscan.ipynb        # Notebook modul utama Unindra
├── 10_alternatif_spotify_music_clustering_kmeans_dbscan.ipynb # Notebook studi kasus alternatif
├── LAPORAN_BAB_10_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf # Dokumen Laporan Resmi Cetak A4
├── visual_output/                               # Grafik visual hasil komputasi
│   └── 10_kmeans_elbow_silhouette.png
└── README.md                                    # Dokumentasi modul
```

## Panduan Menjalankan

```bash
# Menjalankan skrip simulasi mandiri
python main.py

# Membuka Jupyter Notebook
jupyter notebook 10_clustering_kmeans_dan_dbscan.ipynb
```

## Hasil Eksekusi Terminal

```text
[OK] Dataset Spotify Audio Tracks dimuat: 800 lagu, 5 fitur audio.

=======================================================
HASIL EVALUASI CLUSTERING K-MEANS & DBSCAN (BAB 10)
=======================================================
Jumlah Klaster Optimal K-Means : k = 4
Silhouette Score Tertinggi     : 0.3842
Calinski-Harabasz Index        : 412.85
DBSCAN Jumlah Klaster Terbentuk: 3 klaster
DBSCAN Titik Noise (Outlier)   : 28 titik (3.5%)

[OK] Grafik visualisasi disimpan ke visual_output/10_kmeans_elbow_silhouette.png
```

---
*Laboratorium Machine Learning - Teknik Informatika Universitas Indraprasta PGRI*
