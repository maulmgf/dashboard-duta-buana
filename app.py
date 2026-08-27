"""
Dashboard Analisis Penjualan PT Duta Buana Perkasa
====================================================
Aplikasi web interaktif berbasis Streamlit.

CATATAN METODOLOGIS:
Klaster hasil K-Means Clustering diberi label deskriptif ("Kelompok A/B/C/D")
yang SENGAJA BERBEDA dari istilah RFM Analysis ("Champion", "At Risk", dst),
mengikuti pendekatan Agus Tri (2021) yang memberi label unik seperti
"Top Class" dan "Big Consumers" pada klaster K-Means, bukan istilah RFM
standar, guna menghindari ambiguitas antara hasil clustering (unsupervised)
dan RFM Analysis (rule-based).
"""

import os
import base64

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules


# ════════════════════════════════════════════════════════════════
# BAGIAN 1 — STYLING
# ════════════════════════════════════════════════════════════════

MERAH_UTAMA = "#C62828"
MERAH_GELAP = "#A81E1E"
BIRU_UTAMA  = "#2563EB"
ABU_TEKS    = "#6B7280"
ABU_TERANG  = "#F2F3F5"

PALET_GRADASI = ["#C62828", "#D84A4A", "#E06B6B", "#EB9090", "#F3C0C0"]
PALET_NETRAL  = ["#1F2937", "#4B5563", "#6B7280", "#9CA3AF"]


def apply_custom_theme():
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {ABU_TERANG}; }}
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}

        [data-testid="stMetricValue"] {{ color: {MERAH_UTAMA}; font-weight: 700; font-size: 24px; }}
        [data-testid="stMetricLabel"] {{
            color: {ABU_TEKS}; font-weight: 600; text-transform: uppercase;
            letter-spacing: 0.5px; font-size: 11px;
        }}
        [data-testid="stMetric"] {{
            background-color: white; border: 1.5px solid {MERAH_UTAMA};
            border-radius: 14px; padding: 16px 16px;
            box-shadow: 0 4px 14px rgba(198, 40, 40, 0.12);
        }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {MERAH_UTAMA} 0%, {MERAH_GELAP} 100%);
        }}
        section[data-testid="stSidebar"] * {{ color: white !important; }}
        section[data-testid="stSidebar"] .stButton>button {{
            background-color: rgba(255,255,255,0.08) !important; color: white !important;
            border: none !important; border-radius: 8px !important; text-align: left !important;
            justify-content: flex-start !important; font-weight: 500 !important;
            padding: 10px 14px !important; margin: 2px 0 !important; width: 100%;
        }}
        section[data-testid="stSidebar"] .stButton>button:hover {{
            background-color: rgba(255,255,255,0.22) !important;
        }}
        section[data-testid="stSidebar"] .nav-aktif .stButton>button {{
            background-color: white !important; color: {MERAH_UTAMA} !important; font-weight: 700 !important;
        }}

        .stButton>button {{
            background-color: {MERAH_UTAMA}; color: white; border-radius: 8px;
            border: none; font-weight: 600; padding: 8px 20px;
        }}
        .stButton>button:hover {{ background-color: {MERAH_GELAP}; }}

        div[data-baseweb="tab-list"] button[aria-selected="true"] {{
            color: {MERAH_UTAMA}; border-bottom-color: {MERAH_UTAMA};
        }}
        [data-testid="stDataFrame"] {{ border: 1px solid #E5E7EB; border-radius: 10px; overflow: hidden; }}

        [data-baseweb="select"] > div {{
            background-color: white !important; color: #1F2937 !important;
            border: 1.5px solid {MERAH_UTAMA} !important; border-radius: 8px !important;
        }}
        [data-baseweb="select"] * {{ color: #1F2937 !important; }}
        [data-baseweb="select"] svg {{ fill: {MERAH_UTAMA} !important; }}
        [data-baseweb="select"] {{ background-color: white !important; }}
        [data-baseweb="select"] div {{ background-color: white !important; }}
        [data-testid="stSelectbox"] label {{ color: #1F2937 !important; }}
        [data-testid="stMultiSelect"] label {{ color: #1F2937 !important; }}
        [data-baseweb="popover"] [data-baseweb="menu"] {{ background-color: white !important; }}
        [data-baseweb="popover"] [data-baseweb="menu"] li {{ color: #1F2937 !important; background-color: white !important; }}
        [data-baseweb="popover"] [data-baseweb="menu"] li:hover {{ background-color: {ABU_TERANG} !important; color: {MERAH_UTAMA} !important; }}
        [data-baseweb="tag"] {{ background-color: {MERAH_UTAMA} !important; color: white !important; }}
        [data-baseweb="tag"] span {{ color: white !important; }}
        [data-baseweb="tag"] svg {{ fill: white !important; }}
        [data-testid="stSlider"] [role="slider"] {{ background-color: {MERAH_UTAMA} !important; }}
        .stSlider [data-baseweb="slider"] > div > div {{ background-color: {MERAH_UTAMA} !important; }}

        .hero-banner {{
            position: relative; border-radius: 18px; overflow: hidden;
            box-shadow: 0 10px 28px rgba(0,0,0,0.28); margin-bottom: 24px; height: 340px;
        }}
        .hero-banner img {{ width: 100%; height: 100%; object-fit: cover; object-position: center 90%; display: block; }}
        .hero-overlay {{
            position: absolute; bottom: 0; left: 0; right: 0;
            background: linear-gradient(0deg, rgba(0,0,0,0.88) 0%, rgba(0,0,0,0.35) 60%, transparent 100%);
            padding: 30px 28px 22px 28px;
        }}
        .hero-overlay h1 {{ color: white; font-size: 28px; font-weight: 800; margin: 0 0 4px 0; }}
        .hero-overlay p {{ color: #E8E8E8; font-size: 13px; margin: 0; }}
        .hero-badge {{
            display: inline-block; background-color: {MERAH_UTAMA}; color: white;
            font-size: 10px; font-weight: 700; padding: 4px 10px; border-radius: 20px;
            margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.8px;
        }}

        .brand-header {{ display: flex; align-items: center; gap: 12px; padding: 4px 0 14px 0; }}
        .brand-header .logo-box {{
            width: 40px; height: 40px; border-radius: 10px;
            background: linear-gradient(135deg, {MERAH_UTAMA}, {MERAH_GELAP});
            display: flex; align-items: center; justify-content: center;
            color: white; font-weight: 800; font-size: 16px;
        }}
        .brand-header .brand-text h2 {{ margin: 0; color: {MERAH_UTAMA}; font-size: 18px; font-weight: 800; }}
        .brand-header .brand-text p {{ margin: 0; color: {ABU_TEKS}; font-size: 11px; }}

        .header-garis {{ border: none; border-top: 2px solid #F1D4D4; margin: 18px 0; }}

        .kartu-insight {{
            background: white; border-left: 4px solid {MERAH_UTAMA}; border-radius: 10px;
            padding: 12px 14px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            display: flex; align-items: center; gap: 10px; font-size: 13px; color: #1F2937;
        }}
        .kartu-insight .ikon {{ font-size: 18px; flex-shrink: 0; }}

        .badge-ok {{ background-color: #DCFCE7; color: #15803D; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 700; text-transform: uppercase; }}
        .badge-warn {{ background-color: #FEF3C7; color: #B45309; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 700; text-transform: uppercase; }}
        .badge-info {{ background-color: #DBEAFE; color: #1D4ED8; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 700; text-transform: uppercase; }}

        .metric-pill {{
            display: inline-flex; align-items: center; gap: 6px;
            background: linear-gradient(135deg, {MERAH_UTAMA}12, {MERAH_UTAMA}05);
            border: 1px solid {MERAH_UTAMA}30; border-radius: 20px;
            padding: 6px 14px; margin: 3px; font-size: 12px; font-weight: 600; color: {MERAH_UTAMA};
        }}
        .metric-pill-netral {{
            display: inline-flex; align-items: center; gap: 6px;
            background: #F3F4F6; border: 1px solid #D1D5DB; border-radius: 20px;
            padding: 6px 14px; margin: 3px; font-size: 12px; font-weight: 600; color: #374151;
        }}

        .info-box {{
            background: #EFF6FF; border-left: 4px solid {BIRU_UTAMA}; border-radius: 8px;
            padding: 12px 14px; margin: 10px 0; font-size: 13px; color: #1E3A8A;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def render_brand_header():
    st.markdown(
        f"""
        <div class="brand-header">
            <div class="logo-box">D</div>
            <div class="brand-text">
                <h2>PT DUTA BUANA PERKASA</h2>
                <p>Dashboard Analisis Penjualan 2025</p>
            </div>
        </div>
        """, unsafe_allow_html=True
    )


def render_hero_banner():
    banner_path = os.path.join(os.path.dirname(__file__), "banner.jpg")
    if not os.path.exists(banner_path):
        st.warning("Berkas banner.jpg tidak ditemukan.")
        return
    with open(banner_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <div class="hero-banner">
            <img src="data:image/jpg;base64,{img_base64}">
            <div class="hero-overlay">
                <span class="hero-badge">Distributor Resmi</span>
                <h1>PT DUTA BUANA PERKASA</h1>
                <p>Distributor Pelumas & Sparepart — Dashboard Analisis Penjualan 2025</p>
            </div>
        </div>
        """, unsafe_allow_html=True
    )


def kartu_insight(teks_html: str, ikon: str = "info"):
    peta_ikon = {"up": "📈", "down": "📉", "person": "👤", "flag": "🚩", "info": "💡"}
    simbol = peta_ikon.get(ikon, "💡")
    st.markdown(f'<div class="kartu-insight"><span class="ikon">{simbol}</span><span>{teks_html}</span></div>', unsafe_allow_html=True)


def info_box(teks_html: str):
    st.markdown(f'<div class="info-box">ℹ️ {teks_html}</div>', unsafe_allow_html=True)
# ════════════════════════════════════════════════════════════════
# BAGIAN 2 — PIPELINE DATA
# ════════════════════════════════════════════════════════════════

DATA_PATH = os.path.join(os.path.dirname(__file__), "Dashboard_streamlit.xlsx")
FEATS = ['Recency', 'Frequency', 'Monetary', 'Total_QTY', 'Avg_Disc']
MIN_SUPPORT, MIN_CONFIDENCE, MIN_LIFT = 0.02, 0.30, 1.2
KOLOM_MENTAH_ASLI = ['NO_INVOICE','TANGGAL','NO_CUSTOMER','NAMA_CUSTOMER',
                     'ALAMAT','SALESMAN','ITEM','QTY','UNIT',
                     'HARGA_LIST','DISKON_PCT','DISKON_ITEM','HARGA_JUAL']
KOLOM_WAJIB = ['NO_INVOICE', 'TANGGAL', 'NO_CUSTOMER', 'NAMA_CUSTOMER',
               'ITEM', 'QTY', 'HARGA_JUAL', 'DISKON_PCT']

# Label klaster K-Means: SENGAJA BERBEDA dari istilah RFM (Champion/At Risk/dst)
LABEL_KLASTER = {
    1: 'Kelompok A (Frekuensi & Nilai Tertinggi)',
    2: 'Kelompok B (Frekuensi & Nilai Menengah)',
    3: 'Kelompok C (Frekuensi Rendah, Nilai Menengah)',
    4: 'Kelompok D (Frekuensi & Nilai Terendah)',
}


def rfm_segment(s):
    if s >= 13: return 'Champion'
    if s >= 10: return 'High Performer'
    if s >= 7:  return 'Growing'
    if s >= 4:  return 'At Risk'
    return 'Dormant'


def hitung_skor_rfm(rfm_df):
    """Skor RFM 1-5 via kuantil. Frequency & Monetary pakai ascending=True (bug lama diperbaiki)."""
    hasil = rfm_df.copy()
    for col, asc, lbl in [('Recency', True, [5,4,3,2,1]),
                           ('Frequency', True, [1,2,3,4,5]),
                           ('Monetary', True, [1,2,3,4,5])]:
        hasil[f'{col[0]}_Score'] = pd.qcut(
            hasil[col].rank(method='first', ascending=asc), q=5,
            labels=lbl, duplicates='drop').astype(int)
    hasil['RFM_Score'] = hasil['R_Score'] + hasil['F_Score'] + hasil['M_Score']
    return hasil


@st.cache_data
def load_data():
    xls = pd.ExcelFile(DATA_PATH)
    df_transaksi = pd.read_excel(xls, "Transaksi_Bersih")
    rfm          = pd.read_excel(xls, "RFM_Produk")
    seg_summary  = pd.read_excel(xls, "Segment_Summary")
    cluster_prof = pd.read_excel(xls, "Cluster_Profile")
    rules        = pd.read_excel(xls, "Association_Rules")
    tren         = pd.read_excel(xls, "Tren_Bulanan")
    kpi_raw      = pd.read_excel(xls, "KPI_Dashboard")

    try:
        rfm_customer = pd.read_excel(xls, "RFM_Customer")
    except Exception:
        rfm_customer = pd.DataFrame()

    try:
        crosstab_rfm_km = pd.read_excel(xls, "Crosstab_RFM_KMeans", index_col=0)
    except Exception:
        crosstab_rfm_km = pd.DataFrame()

    kpi = dict(zip(kpi_raw["Metric"], kpi_raw["Value"]))

    return {
        "transaksi": df_transaksi, "rfm": rfm, "rfm_customer": rfm_customer,
        "crosstab_rfm_km": crosstab_rfm_km,
        "segment_summary": seg_summary, "cluster_profile": cluster_prof,
        "rules": rules, "tren": tren,
        "kpi": {
            "produk": int(kpi.get("Jumlah Produk", 0)),
            "invoice": int(kpi.get("Jumlah Invoice", 0)),
            "customer": int(kpi.get("Jumlah Customer", 0)),
            "revenue": float(kpi.get("Revenue Total", 0)),
            "k_optimal": int(kpi.get("K Optimal", 0)),
            "silhouette": float(kpi.get("Silhouette Score", 0)),
            "total_rules": int(kpi.get("Total Rules", 0)),
            "max_lift": float(kpi.get("Max Lift", 0)),
        }
    }


def rekomendasi_produk(produk_terpilih, data, top_n=5):
    rfm = data["rfm"]; rules = data["rules"]
    info_produk = rfm[rfm["ITEM"] == produk_terpilih]
    if info_produk.empty:
        return None
    info_produk = info_produk.iloc[0]
    rules_terkait = rules[
        rules["antecedents"].str.contains(produk_terpilih, regex=False) |
        rules["consequents"].str.contains(produk_terpilih, regex=False)
    ].sort_values("lift", ascending=False).head(top_n)
    return {
        "segment_rfm": info_produk.get("Segment_RFM", "-"),
        "segment_cluster": info_produk.get("Segment_Cluster", "-"),
        "recency": info_produk.get("Recency"), "frequency": info_produk.get("Frequency"),
        "monetary": info_produk.get("Monetary"), "rules_terkait": rules_terkait
    }


def hitung_co_occurrence(produk_a, produk_b, transaksi_df):
    inv_a = set(transaksi_df.loc[transaksi_df["ITEM"] == produk_a, "NO_INVOICE"])
    inv_b = set(transaksi_df.loc[transaksi_df["ITEM"] == produk_b, "NO_INVOICE"])
    total_invoice = transaksi_df["NO_INVOICE"].nunique()
    both = len(inv_a & inv_b)
    supp_a = len(inv_a) / total_invoice if total_invoice else 0
    supp_b = len(inv_b) / total_invoice if total_invoice else 0
    support = both / total_invoice if total_invoice else 0
    conf_a_to_b = both / len(inv_a) if inv_a else 0
    conf_b_to_a = both / len(inv_b) if inv_b else 0
    lift = support / (supp_a * supp_b) if (supp_a > 0 and supp_b > 0) else 0
    return {"both": both, "support": support, "conf_a_to_b": conf_a_to_b, "conf_b_to_a": conf_b_to_a,
            "lift": lift, "count_a": len(inv_a), "count_b": len(inv_b), "total_invoice": total_invoice}


def simulasi_bundling(produk_a, produk_b, data):
    rules = data["rules"]
    match = rules[
        (rules["antecedents"].str.contains(produk_a, regex=False) & rules["consequents"].str.contains(produk_b, regex=False))
        | (rules["antecedents"].str.contains(produk_b, regex=False) & rules["consequents"].str.contains(produk_a, regex=False))
    ]
    if not match.empty:
        best = match.sort_values("lift", ascending=False).iloc[0]
        return {"status": "ditemukan_resmi", "support": best["support"], "confidence": best["confidence"], "lift": best["lift"]}
    co = hitung_co_occurrence(produk_a, produk_b, data["transaksi"])
    if co["both"] == 0:
        return {"status": "tidak_pernah_bareng", **co}
    return {"status": "dihitung_langsung", **co}


def hitung_matriks_korelasi(transaksi_df, top_n=15):
    top_items = transaksi_df.groupby("ITEM")["QTY"].sum().nlargest(top_n).index.tolist()
    subset = transaksi_df[transaksi_df["ITEM"].isin(top_items)]
    basket = (subset.groupby(["NO_INVOICE", "ITEM"])["QTY"].sum().unstack(fill_value=0).astype(bool).astype(int))
    basket = basket.reindex(columns=top_items, fill_value=0)
    return basket.corr(), top_items


def baca_berkas_upload(berkas):
    try:
        df_coba1 = pd.read_excel(berkas)
        if not [k for k in KOLOM_WAJIB if k not in df_coba1.columns]:
            return df_coba1, "rapi"
    except Exception:
        pass
    try:
        berkas.seek(0)
        df_coba2 = pd.read_excel(berkas, header=4)
        if df_coba2.shape[1] == len(KOLOM_MENTAH_ASLI):
            df_coba2.columns = KOLOM_MENTAH_ASLI
            if not [k for k in KOLOM_WAJIB if k not in df_coba2.columns]:
                return df_coba2, "mentah_asli"
    except Exception:
        pass
    return None, "tidak_dikenali"


def _bersihkan_data_mentah(df_upload):
    df = df_upload.copy()
    n_awal = len(df)
    kolom_hilang = [k for k in KOLOM_WAJIB if k not in df.columns]
    if kolom_hilang:
        return None, {"error": f"Kolom wajib tidak ditemukan: {', '.join(kolom_hilang)}"}

    n_missing_dropped = df[['NO_INVOICE','ITEM','HARGA_JUAL']].isna().any(axis=1).sum()
    df = df.dropna(subset=['NO_INVOICE', 'ITEM', 'HARGA_JUAL'])
    n_setelah_missing = len(df)
    n_sebelum_dup = len(df)
    df = df.drop_duplicates(subset=['NO_INVOICE', 'ITEM', 'QTY', 'HARGA_JUAL'], keep='first')
    n_dup_dropped = n_sebelum_dup - len(df)
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], errors='coerce')
    for c in ['QTY', 'HARGA_JUAL', 'DISKON_PCT']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    n_sebelum_filter = len(df)
    df = df[(df['HARGA_JUAL'] > 0) & (df['QTY'] > 0)]
    n_filter_dropped = n_sebelum_filter - len(df)
    df['ITEM'] = df['ITEM'].astype(str).str.strip().str.upper()
    if 'NAMA_CUSTOMER' in df.columns:
        df['NAMA_CUSTOMER'] = df['NAMA_CUSTOMER'].astype(str).str.strip().str.upper()
    df['BULAN'] = df['TANGGAL'].dt.to_period('M').astype(str)
    df['HARI'] = df['TANGGAL'].dt.day

    if len(df) == 0:
        return None, {"error": "Setelah pembersihan, tidak ada baris data yang tersisa."}

    funnel = pd.DataFrame({
        'Tahap': ['Data mentah', 'Setelah hapus missing values', 'Setelah hapus duplikat', 'Setelah filter QTY & harga tidak valid'],
        'Jumlah Baris': [n_awal, n_setelah_missing, n_setelah_missing - n_dup_dropped, len(df)],
        'Baris Terhapus': [0, n_missing_dropped, n_dup_dropped, n_filter_dropped]
    })
    return df, {"error": None, "funnel": funnel}


def jalankan_pipeline_baru(df_upload):
    """Pipeline SEGMENTASI PRODUK dari data yang diunggah pengguna."""
    df, info = _bersihkan_data_mentah(df_upload)
    if df is None:
        return info
    funnel = info["funnel"]

    snapshot = df['TANGGAL'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('ITEM').agg(
        Recency=('TANGGAL', lambda x: (snapshot - x.max()).days),
        Frequency=('NO_INVOICE', 'nunique'), Monetary=('HARGA_JUAL', 'sum'),
        Total_QTY=('QTY', 'sum'), Avg_Disc=('DISKON_PCT', 'mean')
    ).reset_index()
    rfm['Avg_Disc'] = rfm['Avg_Disc'].fillna(0)
    if len(rfm) < 4:
        return {"error": "Jumlah produk unik terlalu sedikit untuk clustering (minimal 4)."}

    qt = QuantileTransformer(output_distribution='normal', random_state=42, n_quantiles=min(100, len(rfm)))
    X_scaled = qt.fit_transform(rfm[FEATS])

    rfm = hitung_skor_rfm(rfm)
    rfm['Segment_RFM'] = rfm['RFM_Score'].apply(rfm_segment)

    k_opt = min(4, len(rfm) - 1)
    kmeans = KMeans(n_clusters=k_opt, init='k-means++', random_state=42, n_init=10, max_iter=1000)
    rfm['Cluster'] = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, rfm['Cluster']) if len(rfm) > k_opt else None

    prof = rfm.groupby('Cluster')[FEATS].mean()
    prof['score'] = prof['Monetary'].rank() + prof['Frequency'].rank() + prof['Recency'].rank(ascending=False)
    ranks = prof['score'].rank(ascending=False).astype(int).to_dict()
    rfm['Segment_Cluster'] = rfm['Cluster'].map({c: LABEL_KLASTER.get(r, f'Kelompok {r}') for c, r in ranks.items()})
    rfm['Peringkat_Cluster'] = rfm['Cluster'].map(ranks)

    rfm['Zscore_Monetary'] = rfm.groupby('Cluster')['Monetary'].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else 0)
    rfm['Is_Outlier'] = rfm['Zscore_Monetary'].abs() > 2

    basket = (df.groupby(['NO_INVOICE', 'ITEM'])['QTY'].sum().unstack(fill_value=0).astype(bool))
    frequent_itemsets = apriori(basket, min_support=MIN_SUPPORT, use_colnames=True, max_len=3)
    rules = pd.DataFrame()
    if not frequent_itemsets.empty:
        rules_all = association_rules(frequent_itemsets, metric='lift', min_threshold=MIN_LIFT)
        rules = rules_all[rules_all['confidence'] >= MIN_CONFIDENCE].sort_values('lift', ascending=False)

    tren_bulanan = df.groupby('BULAN')['HARGA_JUAL'].sum().reset_index().rename(columns={'BULAN': 'Bulan', 'HARGA_JUAL': 'Revenue'})
    rapi = lambda s: ', '.join(sorted(s))
    rules_rapi = rules.copy()
    if not rules_rapi.empty:
        rules_rapi['antecedents'] = rules_rapi['antecedents'].apply(rapi)
        rules_rapi['consequents'] = rules_rapi['consequents'].apply(rapi)

    bulanan = df.groupby('BULAN')['HARGA_JUAL'].sum()
    insight = {
        "bulan_terbaik": bulanan.idxmax() if not bulanan.empty else "-",
        "bulan_terendah": bulanan.idxmin() if not bulanan.empty else "-",
        "champion_count": int((rfm['Segment_RFM'] == 'Champion').sum()),
        "outlier_count": int(rfm['Is_Outlier'].sum()),
        "top_produk": rfm.sort_values('Monetary', ascending=False).iloc[0]['ITEM'] if len(rfm) else "-",
    }

    crosstab_baru = pd.crosstab(rfm['Segment_RFM'], rfm['Segment_Cluster'])

    return {
        "error": None, "funnel": funnel, "jumlah_baris": len(df),
        "jumlah_produk": rfm['ITEM'].nunique(), "jumlah_invoice": df['NO_INVOICE'].nunique(),
        "rfm": rfm, "k_optimal": k_opt, "silhouette": sil,
        "jumlah_rules": len(rules), "rules": rules, "rules_rapi": rules_rapi,
        "tren_bulanan": tren_bulanan, "insight": insight,
        "crosstab": crosstab_baru, "df_bersih": df,
    }


def jalankan_pipeline_customer_baru(df_upload):
    """Pipeline SEGMENTASI CUSTOMER dari data yang diunggah pengguna."""
    df, info = _bersihkan_data_mentah(df_upload)
    if df is None:
        return info
    snapshot = df['TANGGAL'].max() + pd.Timedelta(days=1)
    rfm_cust = df.groupby('NAMA_CUSTOMER').agg(
        Recency=('TANGGAL', lambda x: (snapshot - x.max()).days),
        Frequency=('NO_INVOICE', 'nunique'), Monetary=('HARGA_JUAL', 'sum'),
    ).reset_index()
    if len(rfm_cust) < 4:
        return {"error": "Jumlah customer unik terlalu sedikit untuk segmentasi (minimal 4)."}
    rfm_cust = hitung_skor_rfm(rfm_cust)
    rfm_cust['Segment_Customer'] = rfm_cust['RFM_Score'].apply(rfm_segment)
    return {"error": None, "jumlah_customer": rfm_cust['NAMA_CUSTOMER'].nunique(), "rfm_customer": rfm_cust}
# ════════════════════════════════════════════════════════════════
# BAGIAN 3 — KONFIGURASI HALAMAN & NAVIGASI SIDEBAR
# ════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Dashboard PT Duta Buana Perkasa", page_icon="🔴",
                    layout="wide", initial_sidebar_state="expanded")
apply_custom_theme()

MENU = [
    "Ringkasan",
    "Segmentasi Produk",
    "Segmentasi Customer",
    "Pola Beli",
    "Product Recommender",
    "Simulasi Bundling",
    "Peta Korelasi",
    "Kalkulator Keranjang",
    "Upload Data",
]

if "halaman_aktif" not in st.session_state:
    st.session_state.halaman_aktif = MENU[0]

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 10px 0 20px 0;">
            <div style="font-size:15px; font-weight:800; letter-spacing:0.5px;">PT DUTA BUANA PERKASA</div>
            <div style="font-size:11px; opacity:0.85;">Dashboard Analisis Penjualan</div>
        </div>
        """, unsafe_allow_html=True
    )
    for item in MENU:
        aktif = (item == st.session_state.halaman_aktif)
        wrapper_class = "nav-aktif" if aktif else "nav-nonaktif"
        st.markdown(f'<div class="{wrapper_class}">', unsafe_allow_html=True)
        if st.button(item, key=f"nav_{item}", use_container_width=True):
            st.session_state.halaman_aktif = item
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

halaman = st.session_state.halaman_aktif
# ════════════════════════════════════════════════════════════════
# BAGIAN 4 — HALAMAN: RINGKASAN
# ════════════════════════════════════════════════════════════════

def halaman_ringkasan():
    render_brand_header()
    render_hero_banner()
    data = load_data()
    rfm = data["rfm"]; transaksi = data["transaksi"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"Rp {data['kpi']['revenue']/1e9:.2f} M")
    col2.metric("Total Invoice", f"{data['kpi']['invoice']:,}".replace(",", "."))
    col3.metric("Produk Aktif", data['kpi']['produk'])
    col4.metric("Total Customer", data['kpi']['customer'])

    st.markdown('<hr class="header-garis">', unsafe_allow_html=True)
    st.markdown("### Insight Otomatis")

    bulanan = transaksi.groupby("BULAN")["HARGA_JUAL"].sum()
    champion_count = int((rfm["Segment_RFM"] == "Champion").sum())
    kelompok_a_count = int(rfm["Segment_Cluster"].str.contains("Kelompok A", na=False).sum())

    c1, c2 = st.columns(2)
    with c1:
        kartu_insight(f"Bulan penjualan tertinggi: <b>{bulanan.idxmax()}</b> (Rp {bulanan.max()/1e6:,.0f} Jt)", ikon="up")
        kartu_insight(f"<b>{champion_count} produk</b> masuk segmen <b>Champion</b> (RFM Analysis)", ikon="flag")
    with c2:
        kartu_insight(f"Bulan penjualan terendah: <b>{bulanan.idxmin()}</b> (Rp {bulanan.min()/1e6:,.0f} Jt)", ikon="down")
        kartu_insight(f"<b>{kelompok_a_count} produk</b> masuk <b>Kelompok A</b> (K-Means, performa tertinggi)", ikon="flag")

    st.markdown('<hr class="header-garis">', unsafe_allow_html=True)
    st.title("Ringkasan Penjualan")

    col_left, col_right = st.columns([1.4, 1])
    with col_left:
        st.subheader("Tren Revenue Bulanan")
        st.plotly_chart(px.line(data["tren"], x="Bulan", y="Revenue", markers=True, color_discrete_sequence=["#B91C1C"]), use_container_width=True)
    with col_right:
        st.subheader("Distribusi Segmen RFM Produk")
        st.plotly_chart(px.pie(data["segment_summary"], names="Segment_RFM", values="Jumlah_Produk", hole=0.5, color_discrete_sequence=PALET_GRADASI), use_container_width=True)

    col_left2, col_right2 = st.columns(2)
    with col_left2:
        st.subheader("Top 5 Produk berdasarkan Revenue")
        top5 = transaksi.groupby("ITEM")["HARGA_JUAL"].sum().nlargest(5).reset_index()
        fig_bar = px.bar(top5, x="HARGA_JUAL", y="ITEM", orientation="h", color_discrete_sequence=["#B91C1C"])
        fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)
    with col_right2:
        st.subheader("Distribusi Klaster K-Means")
        cp = data["cluster_profile"]
        if "Peringkat_Cluster" in cp.columns:
            cp = cp.sort_values("Peringkat_Cluster")
        fig_donat2 = px.pie(cp, names="Segment_Cluster", values="Jumlah_Produk", hole=0.5, color_discrete_sequence=PALET_NETRAL)
        st.plotly_chart(fig_donat2, use_container_width=True)

    info_box(
        "Segmen RFM (Champion/At Risk/dst) dan Klaster K-Means (Kelompok A/B/C/D) merupakan "
        "<b>dua hasil analisis independen</b> pada fitur dasar yang sama. Detail metodologi dan "
        "validasi silang keduanya dapat dilihat pada halaman <b>Segmentasi Produk</b>."
    )

    # ── Tren Penjualan per Tanggal ──
    st.markdown('<hr class="header-garis">', unsafe_allow_html=True)
    st.subheader("Tren Penjualan per Tanggal")

    transaksi_periode = transaksi.copy()
    transaksi_periode['TANGGAL'] = pd.to_datetime(transaksi_periode['TANGGAL'])
    transaksi_periode['HARI'] = transaksi_periode['TANGGAL'].dt.day
    transaksi_periode['BULAN_NAMA'] = transaksi_periode['TANGGAL'].dt.strftime('%B %Y')

    daftar_bulan = ["Semua Bulan (Gabungan)"] + sorted(
        transaksi_periode['BULAN_NAMA'].unique().tolist(), key=lambda x: pd.to_datetime(x, format='%B %Y'))
    bulan_terpilih = st.selectbox("Pilih tampilan", daftar_bulan)

    if bulan_terpilih == "Semua Bulan (Gabungan)":
        data_tanggal = transaksi_periode
        keterangan = "gabungan seluruh bulan (rata-rata per tanggal)"
    else:
        data_tanggal = transaksi_periode[transaksi_periode['BULAN_NAMA'] == bulan_terpilih]
        keterangan = f"bulan {bulan_terpilih} saja"

    tren_tanggal = data_tanggal.groupby('HARI').agg(Total_Revenue=('HARGA_JUAL', 'sum')).reset_index()
    if bulan_terpilih == "Semua Bulan (Gabungan)":
        jumlah_kemunculan = data_tanggal.groupby('HARI')['TANGGAL'].apply(lambda x: x.dt.to_period('M').nunique()).reset_index()
        jumlah_kemunculan.columns = ['HARI', 'Jumlah_Bulan_Muncul']
        tren_tanggal = tren_tanggal.merge(jumlah_kemunculan, on='HARI')
        tren_tanggal['Nilai_Tampil'] = tren_tanggal['Total_Revenue'] / tren_tanggal['Jumlah_Bulan_Muncul']
        label_sumbu_y = "Rata-rata Revenue"
    else:
        tren_tanggal['Nilai_Tampil'] = tren_tanggal['Total_Revenue']
        label_sumbu_y = "Revenue"

    fig_tanggal = px.line(tren_tanggal, x='HARI', y='Nilai_Tampil', markers=True, color_discrete_sequence=["#B91C1C"],
                           labels={'HARI': 'Tanggal', 'Nilai_Tampil': label_sumbu_y})
    fig_tanggal.update_layout(xaxis=dict(tickmode='linear', dtick=1))
    st.plotly_chart(fig_tanggal, use_container_width=True)

    st.markdown("##### Produk dengan Penjualan Tertinggi Berdasarkan Tanggal")
    daftar_tanggal = sorted(data_tanggal['HARI'].unique().tolist())
    tanggal_terpilih = st.selectbox("Pilih tanggal", daftar_tanggal, key="dropdown_tanggal_produk")
    data_tanggal_terpilih = data_tanggal[data_tanggal['HARI'] == tanggal_terpilih]
    top_produk_tanggal = data_tanggal_terpilih.groupby('ITEM')['HARGA_JUAL'].sum().nlargest(5).reset_index()

    if not top_produk_tanggal.empty:
        fig_top_tanggal = px.bar(top_produk_tanggal, x='HARGA_JUAL', y='ITEM', orientation='h', color_discrete_sequence=["#B91C1C"])
        fig_top_tanggal.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_top_tanggal, use_container_width=True)
        produk_teratas = top_produk_tanggal.iloc[0]
        st.success(f"Pada tanggal **{tanggal_terpilih}** ({keterangan}), produk tertinggi: **{produk_teratas['ITEM']}** (Rp {produk_teratas['HARGA_JUAL']/1e6:,.1f} Jt).")

    if st.button("Deteksi Tanggal Menonjol", type="primary"):
        rata2 = tren_tanggal['Nilai_Tampil'].mean(); std2 = tren_tanggal['Nilai_Tampil'].std()
        puncak = tren_tanggal[tren_tanggal['Nilai_Tampil'] > rata2 + std2].sort_values('Nilai_Tampil', ascending=False)
        lembah = tren_tanggal[tren_tanggal['Nilai_Tampil'] < rata2 - std2].sort_values('Nilai_Tampil')
        col_p, col_l = st.columns(2)
        with col_p:
            st.markdown("**Tanggal menonjol tinggi**")
            for _, row in puncak.head(5).iterrows():
                st.markdown(f"- Tanggal **{int(row['HARI'])}** — Rp {row['Nilai_Tampil']/1e6:,.1f} Jt")
        with col_l:
            st.markdown("**Tanggal menonjol rendah**")
            for _, row in lembah.head(5).iterrows():
                st.markdown(f"- Tanggal **{int(row['HARI'])}** — Rp {row['Nilai_Tampil']/1e6:,.1f} Jt")

    st.markdown('<hr class="header-garis">', unsafe_allow_html=True)
    if st.button("Tampilkan Perbandingan Periode Gajian vs Normal", type="primary"):
        def kategori_periode(hari):
            return "Periode Gajian (25-5)" if (hari >= 25 or hari <= 5) else "Periode Normal (6-24)"
        transaksi_periode['PERIODE_GAJIAN'] = transaksi_periode['HARI'].apply(kategori_periode)
        jumlah_hari = {"Periode Gajian (25-5)": 11, "Periode Normal (6-24)": 19}
        tg = transaksi_periode.groupby('PERIODE_GAJIAN')['HARGA_JUAL'].sum().reset_index()
        tg['Rata2'] = tg.apply(lambda r: r['HARGA_JUAL'] / jumlah_hari[r['PERIODE_GAJIAN']], axis=1)
        gajian_row = tg[tg['PERIODE_GAJIAN'].str.contains('Gajian')]
        normal_row = tg[tg['PERIODE_GAJIAN'].str.contains('Normal')]
        c1, c2 = st.columns(2)
        if not gajian_row.empty: c1.metric("Revenue/hari — Gajian", f"Rp {gajian_row['Rata2'].values[0]/1e6:,.1f} Jt")
        if not normal_row.empty: c2.metric("Revenue/hari — Normal", f"Rp {normal_row['Rata2'].values[0]/1e6:,.1f} Jt")
# ════════════════════════════════════════════════════════════════
# BAGIAN 5 — HALAMAN: SEGMENTASI PRODUK (RFM + K-Means + Validasi + Outlier)
# ════════════════════════════════════════════════════════════════

def halaman_segmentasi_produk():
    render_brand_header()
    st.title("Segmentasi Produk")
    info_box(
        "Halaman ini menampilkan <b>dua hasil analisis independen</b>: RFM Analysis "
        "(rule-based, 5 segmen tetap: Champion/High Performer/Growing/At Risk/Dormant) dan "
        "K-Means Clustering (unsupervised, diberi label deskriptif Kelompok A-D yang "
        "<b>sengaja berbeda</b> dari istilah RFM untuk menghindari ambiguitas)."
    )
    data = load_data()
    rfm = data["rfm"]

    tab_rfm, tab_kmeans, tab_validasi, tab_outlier = st.tabs([
        "🏷️ RFM Analysis", "🔷 K-Means Clustering", "🔄 Validasi Silang", "⚠️ Deteksi Outlier"
    ])

    # ── TAB RFM ──
    with tab_rfm:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Produk", len(rfm))
        c2.metric("Segmen Champion", int((rfm['Segment_RFM'] == 'Champion').sum()))
        c3.metric("Segmen At Risk", int((rfm['Segment_RFM'] == 'At Risk').sum()))

        colA, colB = st.columns([1, 1.3])
        with colA:
            st.markdown("##### Distribusi Segmen")
            st.plotly_chart(px.pie(data["segment_summary"], names="Segment_RFM", values="Jumlah_Produk", hole=0.5,
                                    color_discrete_sequence=PALET_GRADASI), use_container_width=True)
        with colB:
            st.markdown("##### Karakteristik Rata-Rata Tiap Segmen")
            urutan_segmen = ['Champion', 'High Performer', 'Growing', 'At Risk', 'Dormant']
            seg_urut = data["segment_summary"].set_index('Segment_RFM').reindex(
                [s for s in urutan_segmen if s in data["segment_summary"]['Segment_RFM'].values])
            st.dataframe(seg_urut[['Jumlah_Produk','Recency','Frequency','Monetary']], use_container_width=True)

        st.markdown("##### Perbandingan Karakteristik (Chart)")
        colx1, colx2, colx3 = st.columns(3)
        with colx1:
            st.plotly_chart(px.bar(seg_urut, y='Recency', color_discrete_sequence=["#B91C1C"],
                                    title="Recency (rendah = baik)"), use_container_width=True)
        with colx2:
            st.plotly_chart(px.bar(seg_urut, y='Frequency', color_discrete_sequence=[BIRU_UTAMA],
                                    title="Frequency (tinggi = baik)"), use_container_width=True)
        with colx3:
            st.plotly_chart(px.bar(seg_urut, y='Monetary', color_discrete_sequence=["#16A34A"],
                                    title="Monetary (tinggi = baik)"), use_container_width=True)

        st.caption(
            "Segmen **Champion** konsisten memiliki Recency terendah serta Frequency dan Monetary "
            "tertinggi, sesuai definisi Hughes (1994): produk yang terjual baru-baru ini, sering, "
            "dan bernilai tinggi. Kriteria skor: Champion (skor ≥13), High Performer (10-12), "
            "Growing (7-9), At Risk (4-6), Dormant (<4), dari total skor R+F+M (masing-masing 1-5)."
        )

        with st.expander("Lihat detail data produk per segmen"):
            filter_segmen = st.multiselect("Filter segmen", urutan_segmen, default=urutan_segmen, key="filter_rfm_produk")
            tampil = rfm[rfm['Segment_RFM'].isin(filter_segmen)].sort_values('RFM_Score', ascending=False)
            st.dataframe(tampil[['ITEM','Recency','Frequency','Monetary','RFM_Score','Segment_RFM']],
                         use_container_width=True, hide_index=True)

    # ── TAB K-MEANS ──
    with tab_kmeans:
        st.markdown("##### Sebaran Produk per Klaster")
        fig = px.scatter(rfm, x="Frequency", y="Monetary", color="Segment_Cluster", size="Total_QTY",
                          hover_name="ITEM", color_discrete_sequence=PALET_NETRAL)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Profil dan Peringkat Tiap Klaster")
        st.dataframe(data["cluster_profile"], use_container_width=True, hide_index=True)

        cp = data["cluster_profile"]
        urutan_klaster = cp.sort_values("Peringkat_Cluster")["Segment_Cluster"].tolist() if "Peringkat_Cluster" in cp.columns else sorted(rfm["Segment_Cluster"].unique())
        cp_idx = cp.set_index("Segment_Cluster").reindex(urutan_klaster)

        colc1, colc2, colc3 = st.columns(3)
        with colc1:
            st.plotly_chart(px.bar(cp_idx, y='Recency', color_discrete_sequence=["#1F2937"], title="Recency per Klaster"), use_container_width=True)
        with colc2:
            st.plotly_chart(px.bar(cp_idx, y='Frequency', color_discrete_sequence=["#4B5563"], title="Frequency per Klaster"), use_container_width=True)
        with colc3:
            st.plotly_chart(px.bar(cp_idx, y='Monetary', color_discrete_sequence=["#6B7280"], title="Monetary per Klaster"), use_container_width=True)

        with st.expander("Kenapa klaster diberi nama 'Kelompok A/B/C/D', bukan 'Champion' seperti RFM?"):
            st.markdown("""
            K-Means Clustering murni **unsupervised** — algoritma mengelompokkan data berdasarkan
            kemiripan pola pada 5 fitur ternormalisasi (Recency, Frequency, Monetary, Total_QTY,
            Avg_Disc), **tanpa** menggunakan Segment_RFM sebagai input.

            Proses pemberian nama dilakukan **setelah** klaster terbentuk: (1) hitung rata-rata
            tiap fitur per klaster, (2) hitung skor peringkat performa (ranking Monetary + Frequency
            + Recency terbalik), (3) beri label deskriptif sesuai peringkat.

            Label sengaja **berbeda** dari istilah RFM (bukan "Champion"/"At Risk") untuk mencegah
            kesan bahwa K-Means "meniru" kategori RFM — mengikuti pendekatan Agus Tri (2021) yang
            memberi label unik seperti "Top Class" dan "Big Consumers" pada klaster K-Means,
            bukan istilah RFM standar.
            """)

        with st.expander("Lihat detail data produk per klaster"):
            filter_klaster = st.multiselect("Filter klaster", urutan_klaster, default=urutan_klaster, key="filter_kmeans_produk")
            tampil_k = rfm[rfm['Segment_Cluster'].isin(filter_klaster)].sort_values('Peringkat_Cluster' if 'Peringkat_Cluster' in rfm.columns else 'Monetary')
            st.dataframe(tampil_k[['ITEM','Recency','Frequency','Monetary','Segment_Cluster']],
                         use_container_width=True, hide_index=True)

    # ── TAB VALIDASI SILANG ──
    with tab_validasi:
        info_box(
            "Tabel ini membuktikan independensi kedua metode — K-Means <b>tidak</b> menggunakan "
            "Segment_RFM sebagai input. Kesesuaian arah hasil menjadi bukti validasi silang, "
            "bukan bukti ketergantungan antar-metode."
        )
        crosstab = data.get("crosstab_rfm_km")
        if crosstab is None or crosstab.empty:
            crosstab = pd.crosstab(rfm['Segment_RFM'], rfm['Segment_Cluster'])

        fig_heat = px.imshow(crosstab, text_auto=True, color_continuous_scale="Reds",
                              labels=dict(x="Klaster K-Means", y="Segmen RFM", color="Jumlah Produk"))
        fig_heat.update_layout(height=400)
        st.plotly_chart(fig_heat, use_container_width=True)

        if st.button("Jalankan Analisis Kesesuaian", type="primary"):
            if "Peringkat_Cluster" in rfm.columns:
                klaster_terbaik = rfm.loc[rfm["Peringkat_Cluster"] == 1, "Segment_Cluster"].iloc[0]
            else:
                klaster_terbaik = crosstab.sum(axis=0).idxmax()

            if 'Champion' in crosstab.index and klaster_terbaik in crosstab.columns:
                champion_total = crosstab.loc['Champion'].sum()
                champion_selaras = crosstab.loc['Champion', klaster_terbaik]
                persen = (champion_selaras / champion_total * 100) if champion_total > 0 else 0
                colr1, colr2, colr3 = st.columns(3)
                colr1.metric("Produk Champion (RFM)", int(champion_total))
                colr2.metric(f"Juga masuk {klaster_terbaik}", int(champion_selaras))
                colr3.metric("Kesesuaian", f"{persen:.1f}%")
                st.success(
                    f"Dari {int(champion_total)} produk segmen 'Champion' (RFM, rule-based), "
                    f"{int(champion_selaras)} produk ({persen:.1f}%) juga masuk **{klaster_terbaik}** "
                    f"— klaster berperingkat tertinggi hasil K-Means (unsupervised). Kesesuaian ini "
                    f"membuktikan validitas silang tanpa ketergantungan antar-metode."
                )

    # ── TAB OUTLIER ──
    with tab_outlier:
        info_box(
            "Produk dengan nilai Monetary lebih dari dua kali lipat median segmen RFM-nya. "
            "Outlier ini memengaruhi rata-rata (mean) pada grafik segmen, sehingga perlu dibaca "
            "bersamaan dengan median untuk menghindari kesalahan interpretasi."
        )
        outlier_list = []
        for segmen in rfm['Segment_RFM'].unique():
            subset = rfm[rfm['Segment_RFM'] == segmen]
            if len(subset) > 2:
                median_segmen = subset['Monetary'].median()
                outliers = subset[subset['Monetary'] > median_segmen * 2]
                for _, row in outliers.iterrows():
                    outlier_list.append({
                        'Produk': row['ITEM'], 'Segmen RFM': segmen,
                        'Monetary Produk (Rp)': f"{row['Monetary']:,.0f}",
                        'Median Segmen (Rp)': f"{median_segmen:,.0f}",
                        'Selisih dari Median': f"{row['Monetary'] / median_segmen:.1f}x"
                    })
        if outlier_list:
            st.dataframe(pd.DataFrame(outlier_list), use_container_width=True, hide_index=True)
        else:
            st.success("Tidak ditemukan produk dengan nilai ekstrem yang signifikan.")
# ════════════════════════════════════════════════════════════════
# BAGIAN 6 — HALAMAN: SEGMENTASI CUSTOMER
# ════════════════════════════════════════════════════════════════

def halaman_segmentasi_customer():
    render_brand_header()
    st.title("Segmentasi Customer")
    info_box(
        "Analisis RFM diterapkan langsung pada konteks aslinya (pelanggan), sebagai pelengkap "
        "segmentasi produk. Mengacu pada kerangka Hughes (1994) tanpa penyesuaian konseptual."
    )
    data = load_data()
    rfm_customer = data["rfm_customer"]
    rfm_produk = data["rfm"]

    if rfm_customer.empty:
        st.warning("Data segmentasi customer belum tersedia pada berkas Excel saat ini.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Customer", f"{rfm_customer['NAMA_CUSTOMER'].nunique():,}".replace(",", "."))
    c2.metric("Customer Champion", int((rfm_customer['Segment_Customer'] == 'Champion').sum()))
    c3.metric("Customer At Risk", int((rfm_customer['Segment_Customer'] == 'At Risk').sum()))

    st.markdown('<hr class="header-garis">', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.2])
    with col_left:
        st.subheader("Distribusi Segmen Customer")
        seg_count = rfm_customer['Segment_Customer'].value_counts().rename_axis('Segmen').reset_index(name='Jumlah Customer')
        st.plotly_chart(px.pie(seg_count, names='Segmen', values='Jumlah Customer', hole=0.5,
                                color_discrete_sequence=PALET_GRADASI), use_container_width=True)
    with col_right:
        st.subheader("Sebaran Customer: Frequency vs Monetary")
        st.plotly_chart(px.scatter(rfm_customer, x='Frequency', y='Monetary', color='Segment_Customer',
                                    hover_name='NAMA_CUSTOMER', color_discrete_sequence=PALET_GRADASI),
                         use_container_width=True)

    st.subheader("Detail Data Customer")
    urutan_segmen = ['Champion', 'High Performer', 'Growing', 'At Risk', 'Dormant']
    filter_segmen = st.multiselect("Filter berdasarkan segmen", urutan_segmen, default=urutan_segmen, key="filter_cust")
    tampil = rfm_customer[rfm_customer['Segment_Customer'].isin(filter_segmen)].sort_values('RFM_Score', ascending=False)
    st.dataframe(tampil[['NAMA_CUSTOMER', 'Recency', 'Frequency', 'Monetary', 'RFM_Score', 'Segment_Customer']],
                 use_container_width=True, hide_index=True)

    # ── Perbandingan ringkas dengan segmentasi produk (menggantikan halaman terpisah) ──
    with st.expander("📊 Bandingkan dengan Segmentasi Produk"):
        st.caption(
            "Segmentasi customer dan produk menggunakan kerangka RFM yang sama, namun diterapkan "
            "pada unit analisis berbeda — keduanya bersifat komplementer, bukan saling menggantikan."
        )
        seg_p = rfm_produk['Segment_RFM'].value_counts()
        seg_c = rfm_customer['Segment_Customer'].value_counts()

        pills_p = "".join(f'<span class="metric-pill">📦 {s}: {j} produk</span>' for s, j in seg_p.items())
        pills_c = "".join(f'<span class="metric-pill-netral">👤 {s}: {j} customer</span>' for s, j in seg_c.items())
        st.markdown("**Segmentasi Produk:**<br>" + pills_p, unsafe_allow_html=True)
        st.markdown("<br>**Segmentasi Customer:**<br>" + pills_c, unsafe_allow_html=True)
# ════════════════════════════════════════════════════════════════
# BAGIAN 7 — HALAMAN: POLA BELI
# ════════════════════════════════════════════════════════════════

def halaman_pola_beli():
    render_brand_header()
    st.title("Pola Pembelian - Association Rules")
    info_box(
        "Apriori Algorithm menemukan produk yang sering dibeli bersamaan. <b>Support</b> = "
        "seberapa sering kombinasi muncul dari semua invoice. <b>Confidence</b> = dari invoice "
        "yang punya produk A, berapa persen juga punya produk B. <b>Lift</b> > 1 berarti hubungan "
        "nyata (bukan kebetulan) — semakin tinggi, semakin kuat."
    )
    data = load_data()
    rules = data["rules"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rules", data['kpi']['total_rules'])
    c2.metric("Lift Maksimum", f"{data['kpi']['max_lift']:.2f}")
    c3.metric("Rata-rata Lift", f"{rules['lift'].mean():.2f}")

    min_lift = st.slider("Filter minimum Lift", 1.0, float(rules["lift"].max()), 1.2, 0.1)
    min_conf = st.slider("Filter minimum Confidence", 0.0, 1.0, 0.3, 0.05)
    filtered = rules[(rules["lift"] >= min_lift) & (rules["confidence"] >= min_conf)]

    st.subheader(f"10 Rules dengan Lift Tertinggi ({len(filtered)} rules sesuai filter)")
    top10 = filtered.sort_values("lift", ascending=False).head(10)
    st.plotly_chart(px.bar(top10, x="lift", y="antecedents", orientation="h", color_discrete_sequence=["#B91C1C"]), use_container_width=True)

    st.subheader("Daftar Association Rules")
    st.dataframe(filtered[["antecedents", "consequents", "support", "confidence", "lift"]].sort_values("lift", ascending=False), use_container_width=True)


# ════════════════════════════════════════════════════════════════
# BAGIAN 8 — HALAMAN: PRODUCT RECOMMENDER
# ════════════════════════════════════════════════════════════════

def halaman_product_recommender():
    render_brand_header()
    st.title("Product Recommender")
    st.caption("Pilih satu produk untuk melihat segmen RFM, klaster K-Means, tren penjualannya, serta rekomendasi bundling.")
    data = load_data()
    daftar_produk = sorted(data["rfm"]["ITEM"].unique())
    produk_terpilih = st.selectbox("Pilih produk", daftar_produk)

    if produk_terpilih:
        hasil = rekomendasi_produk(produk_terpilih, data)
        if hasil is None:
            st.warning("Produk tidak ditemukan.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Segmen RFM", hasil["segment_rfm"])
            col2.metric("Klaster K-Means", hasil["segment_cluster"])
            col3.metric("Recency (hari)", f"{hasil['recency']:.0f}")
            col4.metric("Frequency (kali)", f"{hasil['frequency']:.0f}")

            st.markdown('<hr class="header-garis">', unsafe_allow_html=True)
            tren_produk = (data["transaksi"][data["transaksi"]["ITEM"] == produk_terpilih].groupby("BULAN")["HARGA_JUAL"].sum().reset_index())
            if not tren_produk.empty:
                st.subheader(f"Tren Penjualan - {produk_terpilih}")
                st.plotly_chart(px.line(tren_produk, x="BULAN", y="HARGA_JUAL", markers=True, color_discrete_sequence=["#B91C1C"]), use_container_width=True)

            st.subheader("Rekomendasi Bundling")
            rt = hasil["rules_terkait"]
            rt_valid = rt[~((rt["antecedents"].str.contains(produk_terpilih, regex=False)) & (rt["consequents"].str.contains(produk_terpilih, regex=False)))]
            if rt_valid.empty:
                st.info("Belum ditemukan pola pembelian bersama yang cukup kuat.")
            else:
                st.dataframe(rt_valid[["antecedents", "consequents", "support", "confidence", "lift"]], use_container_width=True, hide_index=True)
                top_rule = rt_valid.iloc[0]
                produk_lain = top_rule['consequents'] if produk_terpilih in top_rule['antecedents'] else top_rule['antecedents']
                st.success(f"Rekomendasi utama: sering dibeli bersama {produk_lain} (confidence {top_rule['confidence']*100:.1f}%, lift {top_rule['lift']:.2f}).")


# ════════════════════════════════════════════════════════════════
# BAGIAN 9 — HALAMAN: SIMULASI BUNDLING
# ════════════════════════════════════════════════════════════════

def halaman_simulasi_bundling():
    render_brand_header()
    st.title("Simulasi What-If Bundling")
    data = load_data()
    daftar_produk = sorted(data["rfm"]["ITEM"].unique())
    col1, col2 = st.columns(2)
    produk_a = col1.selectbox("Produk pertama", daftar_produk, key="produk_a")
    produk_b = col2.selectbox("Produk kedua", daftar_produk, key="produk_b", index=1 if len(daftar_produk) > 1 else 0)

    if st.button("Simulasikan Bundling", type="primary"):
        if produk_a == produk_b:
            st.warning("Pilih dua produk yang berbeda.")
        else:
            hasil = simulasi_bundling(produk_a, produk_b, data)
            if hasil["status"] == "ditemukan_resmi":
                st.markdown('<span class="badge-ok">Rule resmi Apriori</span>', unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Support", f"{hasil['support']*100:.2f}%")
                c2.metric("Confidence", f"{hasil['confidence']*100:.2f}%")
                c3.metric("Lift", f"{hasil['lift']:.2f}")
            elif hasil["status"] == "dihitung_langsung":
                st.markdown('<span class="badge-warn">Dihitung langsung dari data</span>', unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Support", f"{hasil['support']*100:.3f}%")
                c2.metric("Confidence", f"{hasil['conf_a_to_b']*100:.2f}%")
                c3.metric("Lift", f"{hasil['lift']:.2f}")
            else:
                st.markdown('<span class="badge-warn">Tidak pernah dibeli bersama</span>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# BAGIAN 10 — HALAMAN: PETA KORELASI
# ════════════════════════════════════════════════════════════════

def halaman_peta_korelasi():
    render_brand_header()
    st.title("Peta Korelasi Antar Produk")
    data = load_data()
    top_n = st.slider("Jumlah produk terlaris", 5, 30, 15)
    corr, top_items = hitung_matriks_korelasi(data["transaksi"], top_n=top_n)
    st.plotly_chart(px.imshow(corr, x=top_items, y=top_items, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto"), use_container_width=True)

    st.subheader("Pasangan dengan korelasi tertinggi")
    corr_clean = corr.copy(); corr_clean.index.name = None; corr_clean.columns.name = None
    pairs = corr_clean.where(~corr_clean.isna()).stack().reset_index()
    pairs.columns = ["Produk A", "Produk B", "Korelasi"]
    pairs = pairs[pairs["Produk A"] != pairs["Produk B"]]
    pairs["pasangan_unik"] = pairs.apply(lambda r: tuple(sorted([r["Produk A"], r["Produk B"]])), axis=1)
    pairs = pairs.drop_duplicates("pasangan_unik").drop(columns="pasangan_unik").sort_values("Korelasi", ascending=False).head(10)
    st.dataframe(pairs, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════
# BAGIAN 11 — HALAMAN: KALKULATOR KERANJANG
# ════════════════════════════════════════════════════════════════

def halaman_kalkulator_keranjang():
    render_brand_header()
    st.title("Kalkulator Keranjang Belanja")
    data = load_data()
    rules = data["rules"]
    daftar_produk = sorted(data["rfm"]["ITEM"].unique())
    keranjang = st.multiselect("Isi keranjang", daftar_produk)

    if keranjang:
        saran = rules[rules["antecedents"].apply(lambda a: any(p in a for p in keranjang)) & ~rules["consequents"].apply(lambda c: any(p in c for p in keranjang))].sort_values("lift", ascending=False)
        if saran.empty:
            st.warning("Belum ada saran produk tambahan.")
        else:
            st.subheader(f"{len(saran)} saran produk tambahan")
            sb = saran[["antecedents", "consequents", "confidence", "lift"]].rename(columns={"antecedents": "Karena cocok dengan", "consequents": "Disarankan tambah"}).head(15)
            st.dataframe(sb, use_container_width=True, hide_index=True)
    else:
        st.info("Pilih minimal satu produk untuk mulai simulasi.")


# ════════════════════════════════════════════════════════════════
# BAGIAN 12 — HALAMAN: UPLOAD DATA
# ════════════════════════════════════════════════════════════════

def halaman_upload_data():
    render_brand_header()
    st.title("Upload Data Transaksi Baru")
    st.caption("Unggah berkas Excel untuk menjalankan ulang analisis segmentasi produk dan customer secara otomatis.")

    berkas = st.file_uploader("Pilih berkas Excel (.xlsx)", type=["xlsx"])
    if berkas is None:
        st.info("Silakan unggah berkas Excel untuk memulai.")
        return

    try:
        df_upload, format_terdeteksi = baca_berkas_upload(berkas)
        if df_upload is None:
            st.error("Format berkas tidak dikenali.")
            return
        st.success(f"Format terdeteksi: {'data mentah asli' if format_terdeteksi=='mentah_asli' else 'data sudah rapi'}.")
        st.dataframe(df_upload.head(10), use_container_width=True)

        if st.button("Jalankan Analisis Lengkap", type="primary"):
            with st.spinner("Menjalankan pipeline segmentasi produk dan customer..."):
                st.session_state["hasil_upload_produk"] = jalankan_pipeline_baru(df_upload)
                st.session_state["hasil_upload_customer"] = jalankan_pipeline_customer_baru(df_upload)
    except Exception as e:
        st.error(f"Gagal membaca berkas: {e}")
        return

    hasil = st.session_state.get("hasil_upload_produk")
    hasil_cust = st.session_state.get("hasil_upload_customer")
    if hasil is None or hasil.get("error"):
        if hasil: st.error(hasil["error"])
        return

    st.success("Analisis berhasil dijalankan.")
    st.markdown("#### Funnel Pembersihan Data")
    st.dataframe(hasil["funnel"], use_container_width=True, hide_index=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baris Bersih", f"{hasil['jumlah_baris']:,}".replace(",", "."))
    c2.metric("Produk Unik", hasil["jumlah_produk"])
    c3.metric("Invoice Unik", f"{hasil['jumlah_invoice']:,}".replace(",", "."))
    c4.metric("K Optimal", hasil["k_optimal"])

    rfm_baru = hasil["rfm"]; rules_baru = hasil["rules"]; df_bersih = hasil["df_bersih"]

    tabs = st.tabs(["📈 Ringkasan", "🏷️ RFM & K-Means Produk", "👤 Segmentasi Customer",
                     "🔄 Validasi Silang", "🔗 Association Rules", "📦 Data Bersih"])

    with tabs[0]:
        tren_bulanan = hasil["tren_bulanan"]
        if not tren_bulanan.empty:
            st.plotly_chart(px.line(tren_bulanan, x="Bulan", y="Revenue", markers=True, color_discrete_sequence=["#B91C1C"]), use_container_width=True)
        ins = hasil["insight"]
        ic1, ic2 = st.columns(2)
        with ic1:
            st.info(f"📈 Bulan tertinggi: **{ins['bulan_terbaik']}**")
            st.info(f"🏆 Produk omzet tertinggi: **{ins['top_produk']}**")
        with ic2:
            st.info(f"📉 Bulan terendah: **{ins['bulan_terendah']}**")
            st.warning(f"🚩 **{ins['champion_count']}** produk Champion, **{ins['outlier_count']}** outlier")

    with tabs[1]:
        st.plotly_chart(px.scatter(rfm_baru, x="Frequency", y="Monetary", color="Segment_Cluster",
                                    size="Total_QTY", hover_name="ITEM", color_discrete_sequence=PALET_NETRAL), use_container_width=True)
        st.dataframe(rfm_baru, use_container_width=True)
        st.download_button("Unduh Hasil RFM Produk (CSV)", rfm_baru.to_csv(index=False).encode("utf-8"), "hasil_rfm_produk_baru.csv", "text/csv")

    with tabs[2]:
        if hasil_cust and not hasil_cust.get("error"):
            rfm_cust_baru = hasil_cust["rfm_customer"]
            st.metric("Jumlah Customer", hasil_cust["jumlah_customer"])
            seg_count = rfm_cust_baru['Segment_Customer'].value_counts().rename_axis('Segmen').reset_index(name='Jumlah')
            st.plotly_chart(px.pie(seg_count, names='Segmen', values='Jumlah', hole=0.5, color_discrete_sequence=PALET_GRADASI), use_container_width=True)
            st.dataframe(rfm_cust_baru, use_container_width=True)
            st.download_button("Unduh Hasil RFM Customer (CSV)", rfm_cust_baru.to_csv(index=False).encode("utf-8"), "hasil_rfm_customer_baru.csv", "text/csv")
        else:
            st.info("Segmentasi customer tidak dapat dijalankan (data tidak cukup).")

    with tabs[3]:
        crosstab_baru = hasil.get("crosstab")
        if crosstab_baru is not None and not crosstab_baru.empty:
            st.plotly_chart(px.imshow(crosstab_baru, text_auto=True, color_continuous_scale="Reds"), use_container_width=True)

    with tabs[4]:
        st.metric("Jumlah Rules", hasil["jumlah_rules"])
        if not rules_baru.empty:
            st.dataframe(hasil["rules_rapi"][["antecedents", "consequents", "support", "confidence", "lift"]], use_container_width=True)

    with tabs[5]:
        st.dataframe(df_bersih, use_container_width=True)
        st.download_button("Unduh Data Bersih (CSV)", df_bersih.to_csv(index=False).encode("utf-8"), "data_bersih.csv", "text/csv")


# ════════════════════════════════════════════════════════════════
# BAGIAN 13 — ROUTER
# ════════════════════════════════════════════════════════════════

ROUTER = {
    "Ringkasan": halaman_ringkasan,
    "Segmentasi Produk": halaman_segmentasi_produk,
    "Segmentasi Customer": halaman_segmentasi_customer,
    "Pola Beli": halaman_pola_beli,
    "Product Recommender": halaman_product_recommender,
    "Simulasi Bundling": halaman_simulasi_bundling,
    "Peta Korelasi": halaman_peta_korelasi,
    "Kalkulator Keranjang": halaman_kalkulator_keranjang,
    "Upload Data": halaman_upload_data,
}

ROUTER[halaman]()
