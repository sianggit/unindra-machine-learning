"""
Unit Test & Verifikasi Otomatis: Model Deployment Apps (Bab 14)
Laboratorium Machine Learning - Universitas Indraprasta PGRI (Unindra)
Menguji keabsahan Streamlit Apps, Flask Web Service & REST API
"""

import sys
import io
import json
import py_compile
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

CURRENT_DIR = Path(__file__).resolve().parent

def test_syntax_compilation():
    print("[1/4] Menguji kompilasi sintaks Python...")
    py_files = [
        CURRENT_DIR / "main.py",
        CURRENT_DIR / "app.py",
        CURRENT_DIR / "app_beasiswa.py",
        CURRENT_DIR / "flask_app.py"
    ]
    for p in py_files:
        assert p.exists(), f"Berkas {p.name} tidak ditemukan!"
        py_compile.compile(str(p), doraise=True)
        print(f"   [OK] {p.name}: Lolos kompilasi bytecode (0 galat sintaks).")

def test_model_artifact():
    print("\n[2/4] Menguji artefak serialisasi model joblib...")
    model_path = CURRENT_DIR / "models" / "model_loan_default_best.joblib"
    assert model_path.exists(), "Artefak model_loan_default_best.joblib tidak ditemukan!"
    import joblib
    model = joblib.load(model_path)
    assert hasattr(model, "predict"), "Model tidak memiliki metode predict()!"
    assert hasattr(model, "predict_proba"), "Model tidak memiliki metode predict_proba()!"
    print(f"   [OK] Model {type(model).__name__} berhasil dimuat dan valid.")

def test_flask_endpoints():
    print("\n[3/4] Menguji endpoint Flask Web Service & REST API...")
    from flask_app import app
    client = app.test_client()

    # 1. Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    data_health = json.loads(res_health.data)
    assert data_health["status"] == "ok"
    assert data_health["model_loaded"] is True
    print("   [OK] GET /health: Status 200 OK.")

    # 2. Web Form GET
    res_get = client.get("/")
    assert res_get.status_code == 200
    assert b"Sistem Analisis Risiko Kredit" in res_get.data
    assert b"Anggit Arisa Siwi Pratama" in res_get.data
    print("   [OK] GET /: Form GUI HTML berhasil di-render (Status 200).")

    # 3. Web Form POST (Simulasi Submit Form)
    res_post = client.post("/", data={
        "pendapatan": "15000000",
        "pinjaman": "25000000",
        "tenor": "24",
        "skor_kredit": "750",
        "usia": "35",
        "lama_kerja": "8"
    })
    assert res_post.status_code == 200
    assert b"Rekomendasi Komite" in res_post.data
    print("   [OK] POST /: Submit web form berhasil dihitung (Status 200).")

    # 4. REST API POST /api/predict (Kasus Nasabah Stabil / Rendah Risiko)
    payload_low = {
        "pendapatan_bulanan": 20000000,
        "usia": 38,
        "lama_kerja": 10,
        "jumlah_pinjaman": 30000000,
        "tenor_bulan": 36,
        "skor_kredit": 780
    }
    res_api_low = client.post("/api/predict", json=payload_low)
    assert res_api_low.status_code == 200
    d_low = json.loads(res_api_low.data)
    assert d_low["status"] == "success"
    assert "prediksi" in d_low["data"]
    assert d_low["data"]["is_default"] is False
    print(f"   [OK] POST /api/predict (Profil Stabil): Prediksi = {d_low['data']['risk_label']} | Prob = {d_low['data']['prob_default']:.1%}.")

    # 5. REST API POST /api/predict (Kasus Nasabah Risiko Tinggi)
    payload_high = {
        "pendapatan_bulanan": 4500000,
        "usia": 22,
        "lama_kerja": 0,
        "jumlah_pinjaman": 75000000,
        "tenor_bulan": 12,
        "skor_kredit": 450
    }
    res_api_high = client.post("/api/predict", json=payload_high)
    assert res_api_high.status_code == 200
    d_high = json.loads(res_api_high.data)
    assert d_high["status"] == "success"
    assert d_high["data"]["is_default"] is True
    print(f"   [OK] POST /api/predict (Profil Berisiko): Prediksi = {d_high['data']['risk_label']} | Prob = {d_high['data']['prob_default']:.1%}.")

def test_weights_json():
    print("\n[4/4] Menguji berkas bobot inferensi Vercel WebAssembly...")
    weights_path = CURRENT_DIR / "data" / "loan_default_weights.json"
    assert weights_path.exists(), "loan_default_weights.json tidak ditemukan!"
    with open(weights_path, "r", encoding="utf-8") as f:
        weights = json.load(f)
    assert "best_model_name" in weights or "best_model" in weights
    assert "benchmark_summary" in weights
    model_name = weights.get("best_model_name") or weights.get("best_model")
    print(f"   [OK] loan_default_weights.json valid: Model terpilih = {model_name}.")

if __name__ == "__main__":
    print("=" * 65)
    print("PENGUJIAN OTOMATIS SUITE DEPLOYMENT BAB 14 (UNINDRA)")
    print("=" * 65)
    test_syntax_compilation()
    test_model_artifact()
    test_flask_endpoints()
    test_weights_json()
    print("\n" + "=" * 65)
    print("SELURUH 4 RANGKAIAN UJI DEPLOYMENT LOLOS 100%!")
    print("=" * 65)
