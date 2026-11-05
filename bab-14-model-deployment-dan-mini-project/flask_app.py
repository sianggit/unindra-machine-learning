"""
Aplikasi Web & REST API: Prediksi Risiko Gagal Bayar Pinjaman (Loan Default Prediction)
Laboratorium Machine Learning - Universitas Indraprasta PGRI (Unindra)
Mata Kuliah: Machine Learning (Bab 14: Deployment & Mini Project)
Implementasi Framework: Flask (Sub-bab 14.5 & 14.16)
Mahasiswa: Anggit Arisa Siwi Pratama (NPM: 202343570045)
Dosen Pengampu: Nurfidah Dwitiyanti, M.Si.
"""

import os
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# ─── 1. Pemuatan Artefak Model Terlatih ──────────────────────────────────────
CURRENT_DIR = Path(__file__).resolve().parent
MODEL_PATH = CURRENT_DIR / "models" / "model_loan_default_best.joblib"

def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    from sklearn.ensemble import RandomForestClassifier
    dummy = RandomForestClassifier(n_estimators=15, random_state=42)
    X_d = np.array([[5000000, 30, 3, 30000000, 0.2, 650], [15000000, 45, 10, 50000000, 0.45, 550]])
    y_d = np.array([0, 1])
    dummy.fit(X_d, y_d)
    return dummy

model = load_model()

# ─── 2. Fungsi Logika Inferensi Bersama ──────────────────────────────────────
def run_credit_inference(pendapatan, usia, lama_kerja, pinjaman, tenor, skor_kredit):
    estimasi_cicilan = pinjaman / tenor
    rasio_cicilan = round(float(estimasi_cicilan / pendapatan), 3)

    input_df = pd.DataFrame([[
        pendapatan, usia, lama_kerja, pinjaman, rasio_cicilan, skor_kredit
    ]], columns=["pendapatan_bulanan", "usia", "lama_kerja", "jumlah_pinjaman", "rasio_cicilan", "skor_kredit"])

    prediksi = int(model.predict(input_df)[0])
    prob_default = float(model.predict_proba(input_df)[0][1])
    prob_lancar = round(1.0 - prob_default, 4)

    if prob_default < 0.20:
        rekomendasi = "Persetujuan Otomatis: Ajukan plafon penuh dengan tenor fleksibel."
    elif prob_default < 0.40:
        rekomendasi = "Persetujuan Bersyarat: Memerlukan penambahan agunan atau penyesuaian plafon pinjaman."
    else:
        rekomendasi = "Penolakan Pengajuan: Tingkat rasio cicilan atau riwayat kredit belum memenuhi batas mitigasi risiko."

    risk_label = "Risiko Tinggi (Default)" if prediksi == 1 else "Risiko Rendah (Lolos)"

    return {
        "prediksi": prediksi,
        "is_default": bool(prediksi == 1),
        "risk_label": risk_label,
        "prob_default": round(prob_default, 4),
        "prob_lancar": prob_lancar,
        "rasio_cicilan": rasio_cicilan,
        "estimasi_cicilan": int(estimasi_cicilan),
        "rekomendasi": rekomendasi
    }

# ─── 3. Template HTML GUI Responsif (Tanpa Dependensi Eksternal) ────────────
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sistem Analisis Risiko Kredit (Flask) - Bab 14</title>
    <style>
        :root {
            --bg: #0B0E14;
            --panel: #111622;
            --border: #1E293B;
            --text: #F1F5F9;
            --mute: #94A3B8;
            --primary: #0284C7;
            --primary-hover: #0369A1;
            --success: #10B981;
            --danger: #EF4444;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background-color: var(--bg); color: var(--text); padding: 24px 16px; min-height: 100vh; }
        .container { max-width: 860px; margin: 0 auto; }
        .header { margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }
        .header h1 { font-size: 1.5rem; color: #38BDF8; margin-bottom: 4px; }
        .header p { font-size: 0.85rem; color: var(--mute); }
        .meta-chip { display: inline-block; font-size: 0.75rem; background: rgba(56, 189, 248, 0.1); color: #38BDF8; padding: 2px 8px; border-radius: 4px; margin-top: 8px; }
        .card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 20px; }
        .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 16px; }
        .form-group label { display: block; font-size: 0.8rem; font-weight: 600; color: var(--mute); margin-bottom: 6px; }
        .form-group input, .form-group select { width: 100%; padding: 8px 12px; background: #07090E; border: 1px solid var(--border); border-radius: 4px; color: var(--text); font-size: 0.9rem; }
        .form-group input:focus, .form-group select:focus { outline: none; border-color: var(--primary); }
        .btn-submit { display: block; width: 100%; padding: 10px; background: var(--primary); color: white; border: none; border-radius: 4px; font-weight: 600; cursor: pointer; transition: background 0.2s; }
        .btn-submit:hover { background: var(--primary-hover); }
        .result-box { padding: 16px; border-radius: 6px; margin-top: 16px; }
        .result-box.danger { background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); }
        .result-box.success { background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); }
        .result-title { font-size: 1.2rem; font-weight: 700; margin-bottom: 8px; }
        .result-title.danger { color: var(--danger); }
        .result-title.success { color: var(--success); }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 12px; }
        .stat-item { background: #07090E; padding: 10px; border-radius: 4px; border: 1px solid var(--border); }
        .stat-item .k { font-size: 0.75rem; color: var(--mute); }
        .stat-item .v { font-size: 1rem; font-weight: 700; color: var(--text); margin-top: 2px; }
        .api-note { font-size: 0.78rem; color: var(--mute); line-height: 1.5; margin-top: 8px; }
        code { background: #07090E; padding: 2px 6px; border-radius: 3px; font-size: 0.78rem; color: #38BDF8; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Sistem Analisis Risiko Kredit dan Kelayakan Pinjaman</h1>
            <p>Implementasi Micro-Web Service Flask (Sesuai Sub-bab 14.5 & 14.16 Modul Praktikum Machine Learning Unindra)</p>
            <div class="meta-chip">Peneliti: Anggit Arisa Siwi Pratama | NPM: 202343570045 | Kelas XC Ekstensi</div>
        </div>

        <div class="card">
            <form method="POST" action="/">
                <div class="form-grid">
                    <div class="form-group">
                        <label>Pendapatan Bulanan (Rp):</label>
                        <input type="number" name="pendapatan" value="{{ form.pendapatan if form else 8500000 }}" min="3000000" max="100000000" step="500000" required>
                    </div>
                    <div class="form-group">
                        <label>Plafon Pinjaman Diajukan (Rp):</label>
                        <input type="number" name="pinjaman" value="{{ form.pinjaman if form else 35000000 }}" min="5000000" max="300000000" step="5000000" required>
                    </div>
                    <div class="form-group">
                        <label>Tenor Pelunasan (Bulan):</label>
                        <select name="tenor">
                            {% for t in [12, 24, 36, 48, 60] %}
                            <option value="{{ t }}" {% if form and form.tenor == t %}selected{% elif not form and t == 24 %}selected{% endif %}>{{ t }} Bulan</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Skor Kredit Riwayat (SLIK / BI Checking):</label>
                        <input type="number" name="skor_kredit" value="{{ form.skor_kredit if form else 675 }}" min="350" max="850" step="5" required>
                    </div>
                    <div class="form-group">
                        <label>Usia Peminjam (Tahun):</label>
                        <input type="number" name="usia" value="{{ form.usia if form else 29 }}" min="21" max="65" required>
                    </div>
                    <div class="form-group">
                        <label>Masa Kerja / Usaha (Tahun):</label>
                        <input type="number" name="lama_kerja" value="{{ form.lama_kerja if form else 4 }}" min="0" max="35" required>
                    </div>
                </div>
                <button type="submit" class="btn-submit">Jalankan Analisis Risiko Kredit</button>
            </form>

            {% if result %}
            <div class="result-box {{ 'danger' if result.is_default else 'success' }}">
                <div class="result-title {{ 'danger' if result.is_default else 'success' }}">
                    {{ result.risk_label }}
                </div>
                <p style="font-size: 0.88rem; margin-bottom: 8px;"><b>Rekomendasi Komite:</b> {{ result.rekomendasi }}</p>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="k">Probabilitas Gagal Bayar</div>
                        <div class="v" style="color: {{ 'var(--danger)' if result.is_default else 'var(--success)' }}">{{ (result.prob_default * 100) | round(1) }}%</div>
                    </div>
                    <div class="stat-item">
                        <div class="k">Estimasi Cicilan Bulanan</div>
                        <div class="v">Rp {{ "{:,.0f}".format(result.estimasi_cicilan) }}</div>
                    </div>
                    <div class="stat-item">
                        <div class="k">Rasio Cicilan / Pendapatan</div>
                        <div class="v" style="color: {{ 'var(--danger)' if result.rasio_cicilan > 0.35 else 'var(--success)' }}">{{ (result.rasio_cicilan * 100) | round(1) }}%</div>
                    </div>
                    <div class="stat-item">
                        <div class="k">Tingkat Keyakinan Model</div>
                        <div class="v">{{ (([result.prob_default, result.prob_lancar] | max) * 100) | round(1) }}%</div>
                    </div>
                </div>
            </div>
            {% endif %}
        </div>

        <div class="card">
            <h3 style="font-size: 0.95rem; color: #38BDF8; margin-bottom: 6px;">REST API Endpoint Dokumentasi</h3>
            <p class="api-note">Service ini juga menyediakan endpoint programmatic HTTP POST pada <code>/api/predict</code> dengan payload JSON:</p>
            <pre style="background: #07090E; padding: 10px; border-radius: 4px; font-size: 0.78rem; color: #94A3B8; margin-top: 8px; overflow-x: auto;">curl -X POST http://localhost:5000/api/predict \\
  -H "Content-Type: application/json" \\
  -d '{"pendapatan_bulanan": 8500000, "usia": 29, "lama_kerja": 4, "jumlah_pinjaman": 35000000, "tenor_bulan": 24, "skor_kredit": 675}'</pre>
        </div>
    </div>
</body>
</html>
"""

# ─── 4. Routing Web GUI (HTML) ──────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    form_data = None
    if request.method == "POST":
        try:
            form_data = {
                "pendapatan": float(request.form.get("pendapatan", 8500000)),
                "pinjaman": float(request.form.get("pinjaman", 35000000)),
                "tenor": int(request.form.get("tenor", 24)),
                "skor_kredit": float(request.form.get("skor_kredit", 675)),
                "usia": int(request.form.get("usia", 29)),
                "lama_kerja": int(request.form.get("lama_kerja", 4)),
            }
            result = run_credit_inference(
                form_data["pendapatan"],
                form_data["usia"],
                form_data["lama_kerja"],
                form_data["pinjaman"],
                form_data["tenor"],
                form_data["skor_kredit"]
            )
        except Exception as e:
            result = {
                "is_default": True,
                "risk_label": f"Galat Input: {str(e)}",
                "rekomendasi": "Pastikan seluruh parameter angka telah diisi dengan benar.",
                "prob_default": 1.0,
                "prob_lancar": 0.0,
                "rasio_cicilan": 0.0,
                "estimasi_cicilan": 0
            }
    return render_template_string(HTML_TEMPLATE, form=form_data, result=result)

# ─── 5. Routing REST API Service (JSON) ──────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"status": "error", "message": "Payload JSON tidak valid atau kosong"}), 400

    try:
        pendapatan = float(data.get("pendapatan_bulanan", 8500000))
        usia = int(data.get("usia", 29))
        lama_kerja = int(data.get("lama_kerja", 4))
        pinjaman = float(data.get("jumlah_pinjaman", 35000000))
        tenor = int(data.get("tenor_bulan", 24))
        skor_kredit = float(data.get("skor_kredit", 675))

        res = run_credit_inference(pendapatan, usia, lama_kerja, pinjaman, tenor, skor_kredit)
        return jsonify({
            "status": "success",
            "data": res
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": bool(model is not None),
        "framework": "Flask 3.x",
        "chapter": "Bab 14 (Model Deployment & Mini Project)"
    }), 200

if __name__ == "__main__":
    print("[Flask] Server inferensi risiko kredit aktif pada: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
