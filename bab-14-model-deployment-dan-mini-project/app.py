"""
Sistem Analisis Risiko Kredit dan Kelayakan Pinjaman — FINAL (Executive Bento)
Design tokens follow the ML Lab main-page DESIGN.md.

Run:  streamlit run app.py
"""
import math
import os

import streamlit as st

st.set_page_config(
    page_title="Analisis Risiko Kredit",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────── Design tokens + styles ─────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root{
  --text-primary:#ededed; --text-secondary:#8a94a6; --text-tertiary:#5e6674;
  --surface-base:#000000; --surface-muted:#0f1319; --surface-raised:#090b0e;
  --border-subtle:rgba(255,255,255,.08); --border-strong:rgba(255,255,255,.22);
  --accent:#34d399; --accent-soft:rgba(52,211,153,.12);
  --warn:#fbbf24; --warn-soft:rgba(251,191,36,.12);
  --danger:#f87171; --danger-soft:rgba(248,113,113,.12);
  --radius-sm:8px; --radius-md:14px;
  --shadow-1:rgba(255,255,255,.06) 0 1px 0 0 inset, rgba(0,0,0,.5) 0 20px 40px -20px;
  --shadow-2:rgba(255,255,255,.25) 0 1px 0 0 inset, rgba(0,0,0,.25) 0 8px 24px -10px;
  --motion-fast:200ms;
}
html, body, [class*="css"], .stApp, button, input, select, textarea{
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif !important;
}
.stApp{
  background:radial-gradient(1100px 520px at 78% -8%, rgba(52,211,153,.08), transparent 60%), var(--surface-base);
  color:var(--text-primary);
}
header[data-testid="stHeader"]{background:transparent;}
footer, #MainMenu{visibility:hidden;}
.block-container{padding-top:3rem !important; padding-bottom:4rem !important; max-width:1180px;}

/* Sidebar */
section[data-testid="stSidebar"]{background:var(--surface-raised); border-right:1px solid var(--border-subtle);}
section[data-testid="stSidebar"] .block-container{padding-top:2rem !important;}
section[data-testid="stSidebar"] label p{color:var(--text-secondary) !important; font-size:12px !important; font-weight:500 !important; letter-spacing:.02em;}
.sb-title{font-size:15px;font-weight:600;color:var(--text-primary);margin:0 0 2px}
.sb-sub{font-size:12px;color:var(--text-tertiary);margin:0 0 18px;line-height:1.5}
.sb-group{font-size:10.5px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--text-tertiary);margin:22px 0 8px;padding-top:14px;border-top:1px solid var(--border-subtle)}

/* Inputs */
div[data-baseweb="input"], div[data-baseweb="select"] > div{
  background:var(--surface-muted) !important; border:1px solid var(--border-subtle) !important;
  border-radius:var(--radius-sm) !important; transition:border-color var(--motion-fast);
}
div[data-baseweb="input"]:hover, div[data-baseweb="select"] > div:hover{border-color:var(--border-strong) !important;}
div[data-baseweb="input"]:focus-within{border-color:var(--accent) !important; box-shadow:0 0 0 3px var(--accent-soft);}
div[data-testid="stSlider"] [role="slider"]{background:var(--accent) !important; box-shadow:0 0 0 4px var(--accent-soft);}
div[data-testid="stSlider"] [data-testid="stThumbValue"]{color:var(--accent) !important; font-weight:600;}

/* Primary button */
.stButton > button{
  background:var(--accent); color:#04130d; border:0; border-radius:var(--radius-sm);
  padding:.7rem 1.4rem; font-weight:600; font-size:14px; letter-spacing:.01em;
  box-shadow:var(--shadow-2); transition:transform var(--motion-fast), filter var(--motion-fast);
}
.stButton > button:hover{filter:brightness(1.08); transform:translateY(-1px); color:#04130d; border:0}
.stButton > button:active{transform:translateY(0)}
.stButton > button:focus-visible{outline:2px solid #fff; outline-offset:3px}

/* Components */
.eyebrow{font-size:11px;font-weight:600;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);margin-bottom:10px}
.page-title{font-size:38px;line-height:1.12;font-weight:700;letter-spacing:-.025em;margin:0}
.page-sub{font-size:15px;color:var(--text-secondary);margin:10px 0 0;line-height:1.6;max-width:620px}
.chip{display:inline-flex;gap:10px;align-items:center;padding:10px 14px;border:1px solid var(--border-subtle);background:var(--surface-muted);border-radius:var(--radius-md);box-shadow:var(--shadow-1)}
.chip .av{width:32px;height:32px;border-radius:50%;background:var(--accent-soft);color:var(--accent);display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px}
.chip .nm{font-size:13px;font-weight:600;line-height:1.2}
.chip .mt{font-size:11px;color:var(--text-tertiary);line-height:1.4}
.card{background:linear-gradient(180deg,var(--surface-muted),var(--surface-raised));border:1px solid var(--border-subtle);border-radius:var(--radius-md);padding:26px;box-shadow:var(--shadow-1)}
.card-label{font-size:11px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--text-tertiary);margin-bottom:12px}
.badge{display:inline-flex;align-items:center;gap:8px;padding:5px 12px;border-radius:999px;font-size:12px;font-weight:600}
.badge::before{content:"";width:7px;height:7px;border-radius:50%;background:currentColor}
.t-low{color:var(--accent);background:var(--accent-soft)}
.t-mid{color:var(--warn);background:var(--warn-soft)}
.t-high{color:var(--danger);background:var(--danger-soft)}
.verdict{font-size:34px;font-weight:700;letter-spacing:-.02em;line-height:1.15;margin:12px 0 6px}
.muted{color:var(--text-secondary);font-size:14px;line-height:1.6}
.metric{background:var(--surface-muted);border:1px solid var(--border-subtle);border-radius:var(--radius-md);padding:20px 22px}
.metric .l{font-size:12px;color:var(--text-secondary);margin-bottom:10px}
.metric .v{font-size:26px;font-weight:600;letter-spacing:-.02em}
.metric .s{font-size:11.5px;margin-top:8px;font-weight:500}
.rec{display:flex;gap:12px;padding:12px 0;border-top:1px solid var(--border-subtle);font-size:14px;line-height:1.55}
.rec:first-of-type{border-top:0}
.rec b{color:var(--text-secondary);font-weight:500;min-width:104px}
.section{font-size:19px;font-weight:600;letter-spacing:-.01em;margin:40px 0 16px}
.note{font-size:12px;line-height:1.7;color:var(--text-tertiary);border-top:1px solid var(--border-subtle);padding-top:20px;margin-top:48px}
.note b{color:var(--text-secondary)}
.empty{border:1px dashed var(--border-strong);border-radius:var(--radius-md);padding:44px;text-align:center;color:var(--text-secondary)}
.empty b{display:block;color:var(--text-primary);font-size:16px;margin-bottom:6px}
@media (max-width:760px){.page-title{font-size:28px}.verdict{font-size:26px}}
@media (prefers-reduced-motion:reduce){*{transition:none !important}}
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ───────────────────────── Helpers ─────────────────────────
def h(s: str) -> str:
    """Collapse HTML to one line so Markdown never treats indentation as a code block."""
    return "".join(line.strip() for line in s.splitlines())


def rp(v: float) -> str:
    return "Rp " + f"{v:,.0f}".replace(",", ".")


TIERS = {
    "low": dict(cls="t-low", color="#34d399", label="Risiko Rendah", status="Lolos",
                desc="Nasabah memiliki probabilitas pelunasan yang baik dan profil keuangan stabil.",
                recs=[("Keputusan", "Persetujuan otomatis"),
                      ("Plafon", "Dapat diajukan hingga maksimal"),
                      ("Tenor", "Fleksibel sesuai permintaan nasabah")]),
    "mid": dict(cls="t-mid", color="#fbbf24", label="Risiko Sedang", status="Tinjau",
                desc="Profil keuangan cukup, namun perlu verifikasi tambahan sebelum keputusan akhir.",
                recs=[("Keputusan", "Tinjauan manual oleh analis"),
                      ("Plafon", "Pertimbangkan penurunan 20–30%"),
                      ("Tenor", "Perpanjang untuk menurunkan cicilan")]),
    "high": dict(cls="t-high", color="#f87171", label="Risiko Tinggi", status="Tolak",
                 desc="Probabilitas gagal bayar melampaui ambang batas toleransi komite.",
                 recs=[("Keputusan", "Rekomendasi penolakan"),
                       ("Alternatif", "Tawarkan plafon lebih kecil / agunan"),
                       ("Tindak lanjut", "Evaluasi ulang setelah 6 bulan")]),
}


def tier_of(p: float) -> str:
    return "low" if p < 0.10 else "mid" if p < 0.30 else "high"


@st.cache_resource
def load_model():
    """Loads trained model if present; otherwise returns None (heuristic fallback)."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "model_kredit.pkl"),
        os.path.join(os.path.dirname(__file__), "models", "model_loan_default_best.joblib"),
        os.path.join(os.path.dirname(__file__), "model_loan_default_best.joblib"),
    ]
    for path in candidates:
        if os.path.exists(path):
            import joblib
            try:
                return joblib.load(path)
            except Exception:
                continue
    return None


def predict_default_proba(income, age, work_years, loan, tenor, score) -> float:
    """
    HOOK: align the feature order below with your training pipeline.
    Fallback heuristic keeps the UI runnable without the trained model.
    """
    dti = (loan / tenor) / income
    model = load_model()
    if model is not None:
        try:
            import pandas as pd
            x_df = pd.DataFrame(
                [[income, age, work_years, loan, round(dti, 3), score]],
                columns=["pendapatan_bulanan", "usia", "lama_kerja", "jumlah_pinjaman", "rasio_cicilan", "skor_kredit"]
            )
            return float(model.predict_proba(x_df)[0][1])
        except Exception:
            try:
                x = [[income, age, work_years, loan, tenor, score, dti]]
                return float(model.predict_proba(x)[0][1])
            except Exception:
                pass
    z = -3.5 + 9.0 * (dti - 0.172) - 0.012 * (score - 675) - 0.03 * (work_years - 4) - 0.01 * (age - 29)
    return min(max(1 / (1 + math.exp(-z)), 0.001), 0.999)


def gauge_svg(p: float, color: str) -> str:
    return h(f"""<svg viewBox="0 0 200 115" width="100%" style="max-width:340px" role="img"
      aria-label="Probabilitas gagal bayar {p*100:.1f} persen">
      <path d="M20 100 A80 80 0 0 1 180 100" pathLength="100" fill="none" stroke="rgba(255,255,255,.08)" stroke-width="12" stroke-linecap="round"/>
      <path d="M20 100 A80 80 0 0 1 180 100" pathLength="100" fill="none" stroke="{color}" stroke-width="12"
            stroke-linecap="round" stroke-dasharray="{max(p*100, 1.5):.2f} 100"/>
      <text x="100" y="84" text-anchor="middle" fill="#ededed" font-size="34" font-weight="700" font-family="Inter,sans-serif">{p*100:.1f}%</text>
      <text x="100" y="104" text-anchor="middle" fill="#8a94a6" font-size="9.5" font-family="Inter,sans-serif">PROBABILITAS GAGAL BAYAR</text></svg>""")


# ───────────────────────── Sidebar ─────────────────────────
with st.sidebar:
    st.markdown(h("""<p class="sb-title">Profil Calon Peminjam</p>
        <p class="sb-sub">Masukkan data calon nasabah perbankan untuk dievaluasi model.</p>
        <div class="sb-group" style="border-top:0;padding-top:0;margin-top:0">Keuangan</div>"""),
                unsafe_allow_html=True)
    income = st.number_input("Pendapatan bulanan (Rp)", min_value=1_000_000, value=8_500_000, step=500_000)
    loan = st.number_input("Plafon pinjaman diajukan (Rp)", min_value=1_000_000, value=35_000_000, step=1_000_000)
    tenor = st.selectbox("Tenor pelunasan (bulan)", [6, 12, 18, 24, 36, 48, 60], index=3)

    st.markdown('<div class="sb-group">Profil</div>', unsafe_allow_html=True)
    age = st.slider("Usia peminjam (tahun)", 18, 70, 29)
    work = st.slider("Lama bekerja / usaha (tahun)", 0, 40, 4)

    st.markdown('<div class="sb-group">Riwayat kredit</div>', unsafe_allow_html=True)
    score = st.slider("Skor kredit SLIK / BI Checking", 300, 850, 675)

# ───────────────────────── Compute ─────────────────────────
dti = (loan / tenor) / income
p = predict_default_proba(income, age, work, loan, tenor, score)
t = TIERS[tier_of(p)]
if "ran" not in st.session_state:
    st.session_state.ran = False

# ───────────────────────── Header ─────────────────────────
left, right = st.columns([3, 1.25], vertical_alignment="bottom")
with left:
    st.markdown(h("""<div class="eyebrow">Praktikum Bab 14 · Deployment Model</div>
        <h1 class="page-title">Sistem Analisis Risiko Kredit<br>dan Kelayakan Pinjaman</h1>
        <p class="page-sub">Prototipe web deployment model machine learning (sesuai sub-bab 14.14 Modul Unindra).</p>"""),
                unsafe_allow_html=True)
with right:
    st.markdown(h("""<div class="chip"><div class="av">AP</div><div>
        <div class="nm">Anggit Arisa Siwi Pratama</div>
        <div class="mt">NPM 202343570045 · XC Ekstensi</div></div></div>"""), unsafe_allow_html=True)

st.write("")
st.write("")
if st.button("Jalankan analisis model"):
    st.session_state.ran = True
st.write("")

# ───────────────────────── Decision bento ─────────────────────────
if not st.session_state.ran:
    st.markdown(h("""<div class="empty"><b>Belum ada hasil evaluasi</b>
        Atur profil peminjam di panel kiri, lalu jalankan analisis model.</div>"""), unsafe_allow_html=True)
else:
    col_l, col_r = st.columns([1.1, 1], gap="medium")
    with col_l:
        st.markdown(h(f"""<div class="card" style="text-align:center;padding:34px 26px">
            <div class="card-label">Evaluasi keputusan komite kredit</div>
            {gauge_svg(p, t['color'])}
            <div><span class="badge {t['cls']}">{t['status']}</span></div>
            <div class="verdict" style="color:{t['color']}">{t['label']}</div>
            <div class="muted" style="max-width:380px;margin:0 auto">{t['desc']}</div>
            <div class="muted" style="margin-top:16px;font-size:12.5px">Tingkat keyakinan model
            <b style="color:var(--text-primary)">{max(p, 1 - p)*100:.1f}%</b></div></div>"""),
                    unsafe_allow_html=True)
    with col_r:
        recs = "".join(f'<div class="rec"><b>{k}</b><span>{v}</span></div>' for k, v in t["recs"])
        st.markdown(h(f"""<div class="card" style="margin-bottom:16px">
            <div class="card-label">Pedoman mitigasi komite</div>{recs}</div>"""), unsafe_allow_html=True)
        a, b = st.columns(2)
        a.markdown(h(f'<div class="metric"><div class="l">Rasio beban cicilan</div><div class="v">{dti*100:.1f}%</div></div>'),
                   unsafe_allow_html=True)
        b.markdown(h(f'<div class="metric"><div class="l">Skor kredit SLIK</div><div class="v">{score}</div></div>'),
                   unsafe_allow_html=True)

# ───────────────────────── Feature verification ─────────────────────────
st.markdown('<div class="section">Parameter verifikasi fitur</div>', unsafe_allow_html=True)
dti_ok, score_ok = dti <= 0.35, score >= 650
items = [
    ("Pendapatan", rp(income), "Per bulan", "var(--text-tertiary)"),
    ("Plafon pengajuan", rp(loan), f"Tenor {tenor} bulan", "var(--text-tertiary)"),
    ("Rasio beban cicilan", f"{dti*100:.1f}%",
     "● Di bawah batas 35%" if dti_ok else "● Melebihi batas 35%", "var(--accent)" if dti_ok else "var(--danger)"),
    ("Skor kredit SLIK", str(score),
     "● Kategori baik" if score_ok else "● Di bawah 650", "var(--accent)" if score_ok else "var(--warn)"),
]
for col, (label, value, hint, color) in zip(st.columns(4), items):
    col.markdown(h(f'<div class="metric"><div class="l">{label}</div><div class="v">{value}</div>'
                   f'<div class="s" style="color:{color}">{hint}</div></div>'), unsafe_allow_html=True)

# ───────────────────────── Academic note ─────────────────────────
st.markdown(h("""<div class="note"><b>Catatan akademik:</b> Sistem inferensi ini merupakan luaran Praktikum Bab 14
    Mata Kuliah Machine Learning, Universitas Indraprasta PGRI (Unindra). Model dibangun menggunakan ensemble
    Random Forest dengan pembobotan <i>multi-criteria decision</i> antara metrik F1-Score dan kemampuan
    interpretasi model.</div>"""), unsafe_allow_html=True)
