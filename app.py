"""
Dashboard Analisis Penjualan PT Duta Buana Perkasa
====================================================
Aplikasi web interaktif berbasis Streamlit.

CATATAN METODOLOGIS: Klaster hasil K-Means diberi label deskriptif "Kelompok A/B/C/D"
yang sengaja berbeda dari istilah RFM Analysis ("Champion", "At Risk", dst) agar
hasil clustering (unsupervised) tidak tertukar dengan RFM (rule-based).
K-Means memakai tiga fitur: Recency, Frequency, Monetary.
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

URUTAN_SEGMEN = ['Champion', 'High Performer', 'Growing', 'At Risk', 'Dormant']
# Warna tegas dan berbeda agar kelompok mudah dibedakan (revisi dosen).
# Segmen yang sama memakai warna yang sama di halaman Produk maupun Customer.
WARNA_SEGMEN = {'Champion': '#C62828', 'High Performer': '#F59E0B', 'Growing': '#16A34A',
                'At Risk': '#2563EB', 'Dormant': '#94A3B8'}
WARNA_KLASTER = {'Kelompok A': '#2563EB', 'Kelompok B': '#F59E0B', 'Kelompok C': '#16A34A',
                 'Kelompok D': '#C62828', 'Kelompok E': '#7C3AED', 'Kelompok F': '#0891B2'}

NAMA_BULAN = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli',
              'Agustus', 'September', 'Oktober', 'November', 'Desember']

# Kolom pada Cluster_Profile yang berisi nilai Rupiah (diformat penuh di tabel)
KOLOM_RUPIAH_KLASTER = ['Monetary', 'Monetary_Median', 'Monetary_Min', 'Monetary_Max']


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
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {{
            gap: 0.25rem !important;
        }}
        section[data-testid="stSidebar"] .element-container {{
            margin-bottom: 0 !important;
        }}
        section[data-testid="stSidebar"] .stButton {{
            margin: 0 !important;
        }}
        section[data-testid="stSidebar"] .stButton>button {{
            background-color: rgba(255,255,255,0.08) !important; color: white !important;
            border: none !important; border-radius: 8px !important; text-align: left !important;
            justify-content: flex-start !important; font-weight: 500 !important;
            padding: 10px 14px !important; margin: 0 !important; width: 100%;
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

        /* Dropdown putih, berlaku global di semua halaman */
        [data-baseweb="select"] > div {{
            background-color: #FFFFFF !important; color: #1F2937 !important;
            border: 2px solid {MERAH_UTAMA} !important; border-radius: 8px !important;
        }}
        [data-baseweb="select"] span {{ color: #1F2937 !important; }}
        [data-baseweb="select"] svg {{ fill: {MERAH_UTAMA} !important; }}
        div[data-baseweb="popover"] {{ background-color: #FFFFFF !important; }}
        div[data-baseweb="popover"] {{ box-shadow: 0 6px 20px rgba(0,0,0,0.18) !important; border: 1px solid #D1D5DB !important; border-radius: 8px !important; }}
        ul[data-baseweb="menu"] {{ background-color: #FFFFFF !important; border: 1px solid #E5E7EB !important; }}
        li[role="option"] {{ background-color: #FFFFFF !important; color: #1F2937 !important; }}
        li[role="option"]:hover {{ background-color: {ABU_TERANG} !important; color: {MERAH_UTAMA} !important; }}
        [data-testid="stSelectbox"] label {{ color: #1F2937 !important; }}
        [data-testid="stMultiSelect"] label {{ color: #1F2937 !important; }}
        [data-baseweb="popover"] [data-baseweb="menu"] {{ background-color: white !important; }}
        [data-baseweb="popover"] [data-baseweb="menu"] li {{ color: #1F2937 !important; background-color: white !important; }}
        [data-baseweb="popover"] [data-baseweb="menu"] li:hover {{ background-color: {ABU_TERANG} !important; color: {MERAH_UTAMA} !important; }}
        [data-baseweb="tag"] {{ background-color: {MERAH_UTAMA} !important; color: white !important; }}
        [data-baseweb="tag"] span {{ color: white !important; }}
        [data-baseweb="tag"] svg {{ fill: white !important; }}
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
            background-color: #FFFFFF !important;
        }}
        [data-testid="stMultiSelect"] div[data-baseweb="select"] > div {{
            background-color: #FFFFFF !important;
        }}
        [data-testid="stSelectbox"] input, [data-testid="stMultiSelect"] input {{
            background-color: #FFFFFF !important; color: #1F2937 !important;
        }}

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
        """
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
                <p>Distributor Pelumas &amp; Sparepart — Dashboard Analisis Penjualan 2025</p>
            </div>
        </div>
        """, unsafe_allow_html=True
    )


def _id(angka, desimal=0):
    """Format angka gaya Indonesia: titik ribuan, koma desimal."""
    teks = f"{angka:,.{desimal}f}"
    return teks.replace(",", "X").replace(".", ",").replace("X", ".")


def format_rupiah(nilai):
    """Ringkas untuk kartu/teks: Ribu, Juta, Miliar (2 desimal agar tidak menyesatkan)."""
    if nilai is None or pd.isna(nilai):
        return "Rp 0"
    nilai = float(nilai); a = abs(nilai); tanda = "-" if nilai < 0 else ""
    if a >= 1e9:
        return f"{tanda}Rp {_id(a/1e9, 2)} Miliar"
    elif a >= 1e6:
        return f"{tanda}Rp {_id(a/1e6, 2)} Juta"
    elif a >= 1e3:
        return f"{tanda}Rp {_id(a/1e3, 0)} Ribu"
    return f"{tanda}Rp {_id(a, 0)}"


def format_rupiah_penuh(nilai):
    """Angka lengkap untuk tabel, contoh: Rp 130.032.260 (tanpa pembulatan)."""
    if nilai is None or pd.isna(nilai):
        return "Rp 0"
    return f"Rp {_id(float(nilai), 0)}"


def format_tabel_rupiah(df_input, kolom_list):
    df_tampil = df_input.copy()
    for kolom in kolom_list:
        if kolom in df_tampil.columns:
            df_tampil[kolom] = df_tampil[kolom].apply(format_rupiah_penuh)
    return df_tampil


def nama_bulan_id(periode):
    try:
        th, bl = str(periode).split('-')[:2]
        return f"{NAMA_BULAN[int(bl) - 1]} {th}"
    except Exception:
        return str(periode)


def donat(df_in, nama, nilai, peta, urutan=None):
    """Donat dengan warna berbeda per kelompok dan label nama + jumlah + persen."""
    fig = px.pie(df_in, names=nama, values=nilai, hole=0.5, color=nama,
                 color_discrete_map=peta,
                 category_orders={nama: urutan} if urutan else None)
    fig.update_traces(textinfo="label+value+percent", textposition="outside", sort=False)
    fig.update_layout(showlegend=False, margin=dict(t=30, b=30, l=30, r=30))
    return fig


def bar_warna(df_idx, kolom, judul, peta):
    """Bar chart: tiap segmen/klaster berwarna berbeda. Monetary ditampilkan dalam Rp juta."""
    d = df_idx.reset_index()
    nama = d.columns[0]
    if kolom == 'Monetary':
        d[kolom] = d[kolom] / 1e6
        judul = judul + " (Rp juta)"
    fig = px.bar(d, x=nama, y=kolom, color=nama, color_discrete_map=peta, title=judul)
    fig.update_layout(showlegend=False, xaxis_title=None, separators=",.")
    return fig


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
FEATS_KMEANS = ['Recency', 'Frequency', 'Monetary']   # sama dengan notebook
MIN_SUPPORT, MIN_CONFIDENCE, MIN_LIFT = 0.02, 0.30, 1.2
KOLOM_MENTAH_ASLI = ['NO_INVOICE','TANGGAL','NO_CUSTOMER','NAMA_CUSTOMER',
                     'ALAMAT','SALESMAN','ITEM','QTY','UNIT',
                     'HARGA_LIST','DISKON_PCT','DISKON_ITEM','HARGA_JUAL']
KOLOM_WAJIB = ['NO_INVOICE', 'TANGGAL', 'NO_CUSTOMER', 'NAMA_CUSTOMER',
               'ITEM', 'QTY', 'HARGA_JUAL', 'DISKON_PCT']


def rfm_segment(s):
    if s >= 13: return 'Champion'
    if s >= 10: return 'High Performer'
    if s >= 7:  return 'Growing'
    if s >= 4:  return 'At Risk'
    return 'Dormant'


def _ambang(seri):
    s = seri.round(0)
    q = np.floor(s.quantile([0.2, 0.4, 0.6, 0.8])).tolist()
    return s, q


def beri_skor(seri, tinggi_lebih_baik):
    """Skor 1-5 dari persentil 20/40/60/80. Nilai yang sama selalu mendapat skor yang sama."""
    s, q = _ambang(seri)
    n = np.searchsorted(q, s.values, side='left')
    return (n + 1) if tinggi_lebih_baik else (5 - n)


def hitung_skor_rfm(rfm_df):
    hasil = rfm_df.copy()
    hasil['R_Score'] = beri_skor(hasil['Recency'], False)
    hasil['F_Score'] = beri_skor(hasil['Frequency'], True)
    hasil['M_Score'] = beri_skor(hasil['Monetary'], True)
    hasil['RFM_Score'] = hasil['R_Score'] + hasil['F_Score'] + hasil['M_Score']
    return hasil


def _baca_opsional(xls, nama, **kw):
    try:
        return pd.read_excel(xls, nama, **kw)
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_data():
    xls = pd.ExcelFile(DATA_PATH)
    kpi_raw = pd.read_excel(xls, "KPI_Dashboard")
    kpi = dict(zip(kpi_raw["Metric"], kpi_raw["Value"]))
    return {
        "transaksi": pd.read_excel(xls, "Transaksi_Bersih"),
        "rfm": pd.read_excel(xls, "RFM_Produk"),
        "segment_summary": pd.read_excel(xls, "Segment_Summary"),
        "cluster_profile": pd.read_excel(xls, "Cluster_Profile"),
        "rules": pd.read_excel(xls, "Association_Rules"),
        "tren": pd.read_excel(xls, "Tren_Bulanan"),
        "rfm_customer": _baca_opsional(xls, "RFM_Customer"),
        "crosstab_rfm_km": _baca_opsional(xls, "Crosstab_RFM_KMeans", index_col=0),
        "interval_rfm": _baca_opsional(xls, "Interval_RFM"),
        "interval_rfm_cust": _baca_opsional(xls, "Interval_RFM_Customer"),
        "aturan_klaster": _baca_opsional(xls, "Aturan_Klaster"),
        "pengaruh_fitur": _baca_opsional(xls, "Pengaruh_Fitur"),
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
        "alasan": info_produk.get("Alasan_Klaster", ""),
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
        return None, {"error": "Setelah pembersihan, tidak ada baris data yang tersisa. Periksa kembali format tanggal, harga, dan qty pada berkas yang diunggah."}

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
        return {"error": "Jumlah produk unik terlalu sedikit untuk clustering (minimal 4 produk berbeda dibutuhkan)."}

    qt = QuantileTransformer(output_distribution='normal', random_state=42, n_quantiles=min(100, len(rfm)))
    X_km = qt.fit_transform(rfm[FEATS_KMEANS])

    rfm = hitung_skor_rfm(rfm)
    rfm['Segment_RFM'] = rfm['RFM_Score'].apply(rfm_segment)

    # K dipilih dari Silhouette (K >= 3), bukan dipatok 4
    sil_k = {}
    for k in range(3, min(9, len(rfm) - 1) + 1):
        lb = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10, max_iter=1000).fit_predict(X_km)
        sil_k[k] = silhouette_score(X_km, lb)
    k_opt = max(sil_k, key=sil_k.get)
    kmeans = KMeans(n_clusters=k_opt, init='k-means++', random_state=42, n_init=10, max_iter=1000)
    rfm['Cluster'] = kmeans.fit_predict(X_km)
    sil = silhouette_score(X_km, rfm['Cluster'])

    prof = rfm.groupby('Cluster')[FEATS_KMEANS].mean()
    prof['score'] = sum(prof[f].rank(ascending=(f != 'Recency')) for f in FEATS_KMEANS)
    ranks = prof['score'].rank(ascending=False, method='first').astype(int).to_dict()
    rfm['Segment_Cluster'] = rfm['Cluster'].map({c: f"Kelompok {chr(64 + r)}" for c, r in ranks.items()})
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
        return {"error": "Jumlah customer unik terlalu sedikit untuk segmentasi (minimal 4 customer berbeda dibutuhkan)."}
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

    # ── KPI ──
    st.markdown("### Key Performance Indicator (KPI) Utama")
    st.caption(
        "*Key Performance Indicator* (KPI) atau Indikator Kinerja Utama adalah ukuran ringkas "
        "kondisi bisnis. Keempat KPI berikut dihitung dari seluruh transaksi Januari–Desember 2025 "
        "setelah pembersihan data."
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", format_rupiah(data['kpi']['revenue']),
                help="Jumlah seluruh nilai penjualan (HARGA_JUAL) pada data bersih.")
    col2.metric("Total Invoice", _id(data['kpi']['invoice']),
                help="Jumlah nomor invoice yang unik.")
    col3.metric("Total Produk", data['kpi']['produk'],
                help="Jumlah produk (ITEM) unik yang terjual selama 2025, bukan hanya yang baru-baru ini dibeli.")
    col4.metric("Total Customer", data['kpi']['customer'],
                help="Jumlah customer unik (berdasarkan nama) selama 2025.")

    # ── Insight otomatis ──
    st.markdown('<hr class="header-garis">', unsafe_allow_html=True)
    st.markdown("### Insight Otomatis")
    st.caption("Ringkasan otomatis untuk membantu manajemen memahami kondisi penjualan tanpa membaca data mentah. "
               "Angka pada kartu dihitung dari data yang sama dengan grafik di bawahnya.")

    bulanan = transaksi.groupby("BULAN")["HARGA_JUAL"].sum()
    champion_count = int((rfm["Segment_RFM"] == "Champion").sum())
    terbaik = rfm[rfm["Peringkat_Cluster"] == 1] if "Peringkat_Cluster" in rfm.columns else rfm.iloc[0:0]

    c1, c2 = st.columns(2)
    with c1:
        kartu_insight(f"Bulan penjualan tertinggi: <b>{nama_bulan_id(bulanan.idxmax())}</b> "
                      f"({format_rupiah(bulanan.max())}) — terlihat pada grafik <i>Tren Revenue Bulanan</i> di bawah", ikon="up")
        kartu_insight(f"<b>{champion_count} produk</b> masuk segmen <b>Champion</b> (RFM Analysis): skor RFM ≥ 13 dari "
                      f"maksimal 15, artinya baru terjual, sering terjual, dan bernilai tinggi — prioritaskan ketersediaan stok", ikon="flag")
    with c2:
        kartu_insight(f"Bulan penjualan terendah: <b>{nama_bulan_id(bulanan.idxmin())}</b> "
                      f"({format_rupiah(bulanan.min())})", ikon="down")
        if not terbaik.empty:
            ket_t = terbaik["Karakteristik_Cluster"].iloc[0] if "Karakteristik_Cluster" in terbaik.columns else ""
            kartu_insight(f"<b>{len(terbaik)} produk</b> masuk <b>{terbaik['Segment_Cluster'].iloc[0]}</b> "
                          f"(K-Means, klaster peringkat 1). Ciri rata-rata klaster: {ket_t}", ikon="flag")

    b1, b2, _ = st.columns([1.2, 1.2, 2.6])
    if b1.button("Rincian segmen RFM →", key="ke_rfm"):
        st.session_state.halaman_aktif = "Segmentasi Produk"; st.rerun()
    if b2.button("Rincian klaster K-Means →", key="ke_kmeans"):
        st.session_state.halaman_aktif = "Segmentasi Produk"; st.rerun()

    st.markdown('<hr class="header-garis">', unsafe_allow_html=True)
    st.title("Ringkasan Penjualan")

    col_left, col_right = st.columns([1.4, 1])
    with col_left:
        st.subheader("Tren Revenue Bulanan")
        tren_plot = data["tren"].copy()
        tren_plot["Revenue (Rp juta)"] = tren_plot["Revenue"] / 1e6
        fig_tren = px.line(tren_plot, x="Bulan", y="Revenue (Rp juta)", markers=True, color_discrete_sequence=["#B91C1C"])
        fig_tren.update_layout(separators=",.")
        st.plotly_chart(fig_tren, use_container_width=True)
    with col_right:
        st.subheader("Distribusi Segmen RFM Produk")
        st.caption("RFM Analysis (rule-based). Label: nama segmen, jumlah produk, persentase.")
        st.plotly_chart(donat(data["segment_summary"], "Segment_RFM", "Jumlah_Produk", WARNA_SEGMEN, URUTAN_SEGMEN),
                        use_container_width=True)

    # ── Perkiraan bulan berikutnya ──
    st.markdown('<hr class="header-garis">', unsafe_allow_html=True)
    st.subheader("Perkiraan Revenue Bulan Berikutnya")
    st.caption("Perkiraan dihitung dari rata-rata pertumbuhan revenue 3 bulan terakhir (moving average), "
               "bukan model machine learning kompleks. Hasilnya indikatif, bukan jaminan.")
    tren_data = data["tren"].copy()
    if len(tren_data) >= 3:
        tiga = tren_data.tail(3)['Revenue'].values
        pertumbuhan = [(tiga[i] - tiga[i-1]) / tiga[i-1] for i in range(1, len(tiga)) if tiga[i-1] > 0]
        rata_pertumbuhan = sum(pertumbuhan) / len(pertumbuhan) if pertumbuhan else 0
        revenue_terakhir = tren_data.iloc[-1]['Revenue']
        perkiraan = revenue_terakhir * (1 + rata_pertumbuhan)

        colf1, colf2, colf3 = st.columns(3)
        colf1.metric("Revenue Bulan Terakhir", format_rupiah(revenue_terakhir))
        colf2.metric("Rata-rata Pertumbuhan", f"{rata_pertumbuhan*100:+.1f}%".replace(".", ","))
        colf3.metric("Perkiraan Bulan Berikutnya", format_rupiah(perkiraan))

        if rata_pertumbuhan > 0.05:
            st.success(f"Tren 3 bulan terakhir tumbuh rata-rata **{rata_pertumbuhan*100:.1f}% per bulan**. "
                       f"Jika berlanjut, revenue bulan berikutnya sekitar **{format_rupiah(perkiraan)}**; siapkan stok sesuai proyeksi.")
        elif rata_pertumbuhan < -0.05:
            st.warning(f"Tren 3 bulan terakhir turun rata-rata **{abs(rata_pertumbuhan)*100:.1f}% per bulan**. "
                       f"Jika berlanjut, revenue bulan berikutnya sekitar **{format_rupiah(perkiraan)}**; tinjau strategi penjualan atau promosi.")
        else:
            st.info(f"Tren 3 bulan terakhir relatif stabil (**{rata_pertumbuhan*100:+.1f}% per bulan**). "
                    f"Perkiraan bulan berikutnya sekitar **{format_rupiah(perkiraan)}**.")
        with st.expander("Lihat metode perhitungan"):
            st.markdown("**Rumus:** Perkiraan = Revenue bulan terakhir × (1 + rata-rata pertumbuhan 3 bulan).\n\n"
                        "Metode sederhana dipilih karena data historis hanya satu tahun, sehingga mudah ditelusuri manajemen.")
    else:
        st.info("Data historis belum cukup (minimal 3 bulan) untuk menghitung perkiraan tren.")

    # ── Top 5 dan donat K-Means ──
    st.markdown('<hr class="header-garis">', unsafe_allow_html=True)
    col_left2, col_right2 = st.columns(2)
    with col_left2:
        st.subheader("Top 5 Produk berdasarkan Revenue")
        top5 = transaksi.groupby("ITEM")["HARGA_JUAL"].sum().nlargest(5).reset_index()
        top5["Revenue (Rp juta)"] = top5["HARGA_JUAL"] / 1e6
        fig_bar = px.bar(top5, x="Revenue (Rp juta)", y="ITEM", orientation="h", color_discrete_sequence=["#B91C1C"])
        fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'}, separators=",.")
        st.plotly_chart(fig_bar, use_container_width=True)
    with col_right2:
        st.subheader("Distribusi Klaster K-Means")
        st.caption("K-Means (unsupervised). Label: kelompok, jumlah produk, persentase.")
        cp = data["cluster_profile"]
        if "Peringkat_Cluster" in cp.columns:
            cp = cp.sort_values("Peringkat_Cluster")
        st.plotly_chart(donat(cp, "Segment_Cluster", "Jumlah_Produk", WARNA_KLASTER, cp["Segment_Cluster"].tolist()),
                        use_container_width=True)

    info_box(
        "Segmen RFM (Champion/At Risk/dst) dan Klaster K-Means (Kelompok A/B/C/…) adalah "
        "<b>dua hasil analisis independen</b>. Warna dibuat berbeda di setiap kelompok agar mudah dibedakan. "
        "Penjelasan tiap kelompok dan alasan sebuah produk masuk ke kelompoknya ada di halaman <b>Segmentasi Produk</b>."
    )

    # ── Tren Penjualan per Tanggal ──
    st.markdown('<hr class="header-garis">', unsafe_allow_html=True)
    st.subheader("Tren Penjualan per Tanggal")
    st.caption("Pola penjualan menurut tanggal (1–31) untuk membantu manajemen menyiapkan stok pada tanggal yang secara historis ramai.")

    tp = transaksi.copy()
    tp['TANGGAL'] = pd.to_datetime(tp['TANGGAL'])
    tp['HARI'] = tp['TANGGAL'].dt.day
    tp['BULAN_KEY'] = tp['TANGGAL'].dt.to_period('M').astype(str)

    # Kalender lengkap dari awal bulan pertama sampai akhir bulan terakhir data.
    # Dipakai sebagai pembagi rata-rata (berapa kali tiap tanggal muncul) dan untuk menghitung hari tanpa transaksi sebagai Rp 0.
    awal = tp['TANGGAL'].min().to_period('M').start_time
    akhir = tp['TANGGAL'].max().to_period('M').end_time.normalize()
    kal = pd.date_range(awal, akhir, freq='D')
    kemunculan = pd.Series(kal.day).value_counts()

    opsi = ["Semua Bulan (Gabungan)"] + sorted(tp['BULAN_KEY'].unique().tolist())
    bulan_terpilih = st.selectbox("Pilih tampilan", opsi,
                                  format_func=lambda x: x if x.startswith("Semua") else nama_bulan_id(x))
    gabungan = bulan_terpilih.startswith("Semua")

    if gabungan:
        data_tanggal = tp
        keterangan = "gabungan seluruh bulan (rata-rata per bulan)"
    else:
        data_tanggal = tp[tp['BULAN_KEY'] == bulan_terpilih]
        keterangan = f"bulan {nama_bulan_id(bulan_terpilih)} saja"

    tren_tanggal = data_tanggal.groupby('HARI')['HARGA_JUAL'].sum().reset_index(name='Total_Revenue')
    if gabungan:
        tren_tanggal['Nilai_Tampil'] = tren_tanggal['Total_Revenue'] / tren_tanggal['HARI'].map(kemunculan)
        label_y = "Rata-rata Revenue per Bulan (Rp juta)"
        info_box("<b>Cara membaca tampilan gabungan:</b> setiap titik adalah revenue rata-rata pada tanggal tersebut. "
                 "Revenue tanggal itu dari seluruh bulan dijumlahkan, lalu dibagi berapa kali tanggal itu muncul pada "
                 "kalender 2025. Contoh: titik tanggal 11 = total revenue semua tanggal 11 dibagi 12 bulan.")
    else:
        tren_tanggal['Nilai_Tampil'] = tren_tanggal['Total_Revenue']
        label_y = "Revenue (Rp juta)"
    tren_tanggal['Nilai_Juta'] = tren_tanggal['Nilai_Tampil'] / 1e6

    fig_tanggal = px.line(tren_tanggal, x='HARI', y='Nilai_Juta', markers=True, color_discrete_sequence=["#B91C1C"],
                          labels={'HARI': 'Tanggal', 'Nilai_Juta': label_y})
    fig_tanggal.update_layout(xaxis=dict(tickmode='linear', dtick=1), separators=",.")
    st.plotly_chart(fig_tanggal, use_container_width=True)

    t_max = tren_tanggal.loc[tren_tanggal['Nilai_Tampil'].idxmax()]
    t_min = tren_tanggal.loc[tren_tanggal['Nilai_Tampil'].idxmin()]

    def rata_rentang(a, b):
        sub = tren_tanggal[(tren_tanggal['HARI'] >= a) & (tren_tanggal['HARI'] <= b)]['Nilai_Tampil']
        return sub.mean() if not sub.empty else np.nan
    rentang = {"tanggal 1–10": rata_rentang(1, 10), "tanggal 11–20": rata_rentang(11, 20), "tanggal 21–31": rata_rentang(21, 31)}
    rentang = {k: v for k, v in rentang.items() if not pd.isna(v)}
    rentang_ramai = max(rentang, key=rentang.get)

    info_box(
        f"Pada tampilan <b>{keterangan}</b>, penjualan <b>tertinggi</b> terjadi pada tanggal "
        f"<b>{int(t_max['HARI'])}</b> ({format_rupiah(t_max['Nilai_Tampil'])}) dan <b>terendah</b> pada tanggal "
        f"<b>{int(t_min['HARI'])}</b> ({format_rupiah(t_min['Nilai_Tampil'])}). "
        f"<b>Tren per rentang:</b> " + "; ".join(f"{k} rata-rata {format_rupiah(v)}" for k, v in rentang.items()) +
        f". Rentang paling ramai: <b>{rentang_ramai}</b>."
    )

    st.markdown("##### Produk dengan Penjualan Tertinggi Berdasarkan Tanggal")
    daftar_tanggal = sorted(data_tanggal['HARI'].unique().tolist())
    tanggal_terpilih = st.selectbox("Pilih tanggal", daftar_tanggal, key="dropdown_tanggal_produk")
    dt = data_tanggal[data_tanggal['HARI'] == tanggal_terpilih]
    top_produk = dt.groupby('ITEM')['HARGA_JUAL'].sum().nlargest(5).reset_index()

    if not top_produk.empty:
        pembagi = kemunculan.get(tanggal_terpilih, 1) if gabungan else 1
        top_produk['Nilai'] = top_produk['HARGA_JUAL'] / pembagi
        top_produk['Nilai_Juta'] = top_produk['Nilai'] / 1e6
        sumbu = "Rata-rata Revenue per Bulan (Rp juta)" if gabungan else "Revenue (Rp juta)"
        fig_tt = px.bar(top_produk, x='Nilai_Juta', y='ITEM', orientation='h', color_discrete_sequence=["#B91C1C"],
                        labels={'Nilai_Juta': sumbu})
        fig_tt.update_layout(yaxis={'categoryorder': 'total ascending'}, separators=",.")
        st.plotly_chart(fig_tt, use_container_width=True)
        nilai_tgl = tren_tanggal.loc[tren_tanggal['HARI'] == tanggal_terpilih, 'Nilai_Tampil'].iloc[0]
        st.success(
            f"Pada tanggal **{tanggal_terpilih}** ({keterangan}), produk tertinggi adalah **{top_produk.iloc[0]['ITEM']}** "
            f"({format_rupiah(top_produk.iloc[0]['Nilai'])}), dari total {format_rupiah(nilai_tgl)} pada tanggal tersebut."
        )
    else:
        st.info(f"Tidak ada transaksi pada tanggal {tanggal_terpilih} untuk {keterangan}.")

    if st.button("Deteksi Tanggal Menonjol", type="primary"):
        rata2 = tren_tanggal['Nilai_Tampil'].mean(); std2 = tren_tanggal['Nilai_Tampil'].std()
        puncak = tren_tanggal[tren_tanggal['Nilai_Tampil'] > rata2 + std2].sort_values('Nilai_Tampil', ascending=False)
        lembah = tren_tanggal[tren_tanggal['Nilai_Tampil'] < rata2 - std2].sort_values('Nilai_Tampil')
        st.caption(f"Metode: tanggal disebut 'menonjol' bila nilainya berbeda lebih dari 1 standar deviasi dari "
                   f"rata-rata seluruh tanggal (rata-rata = {format_rupiah(rata2)}).")

        def produk_teratas_pada(h):
            sub = data_tanggal[data_tanggal['HARI'] == h]
            return sub.groupby('ITEM')['HARGA_JUAL'].sum().idxmax() if not sub.empty else "-"

        col_p, col_l = st.columns(2)
        with col_p:
            st.markdown("**📈 Tanggal menonjol tinggi**")
            if not puncak.empty:
                for _, r in puncak.head(5).iterrows():
                    st.markdown(f"- Tanggal **{int(r['HARI'])}** — {format_rupiah(r['Nilai_Tampil'])} "
                                f"({r['Nilai_Tampil']/rata2:.1f}× rata-rata). Produk penyumbang terbesar: {produk_teratas_pada(r['HARI'])}")
            else:
                st.caption("Tidak ada tanggal yang menonjol tinggi.")
        with col_l:
            st.markdown("**📉 Tanggal menonjol rendah**")
            if not lembah.empty:
                for _, r in lembah.head(5).iterrows():
                    st.markdown(f"- Tanggal **{int(r['HARI'])}** — {format_rupiah(r['Nilai_Tampil'])} "
                                f"({r['Nilai_Tampil']/rata2:.1f}× rata-rata)")
            else:
                st.caption("Tidak ada tanggal yang menonjol rendah.")
        if not puncak.empty:
            st.success("**Rekomendasi:** tanggal " + ", ".join(str(int(h)) for h in puncak['HARI'].head(5)) +
                       " secara historis lebih ramai dari rata-rata; pastikan stok produk terlaris tersedia menjelang tanggal tersebut.")

    # ── Periode Gajian vs Normal ──
    st.markdown('<hr class="header-garis">', unsafe_allow_html=True)
    st.subheader("Insight: Periode Gajian vs Normal")
    st.caption(
        "Membandingkan rata-rata revenue per hari pada **periode gajian** (tanggal 25 sampai akhir bulan dan tanggal 1–5) "
        "dengan **periode normal** (tanggal 6–24, selalu 19 hari). Jumlah hari periode gajian berbeda tiap bulan "
        "(9–12 hari, tergantung panjang bulan), sehingga dipakai rata-rata per hari yang dihitung dari kalender, bukan total. "
        "Hari tanpa transaksi (misalnya hari libur) tetap dihitung sebagai Rp 0."
    )
    if st.button("Tampilkan Perbandingan Periode Gajian vs Normal", type="primary"):
        harian = tp.groupby(tp['TANGGAL'].dt.normalize())['HARGA_JUAL'].sum().reindex(kal, fill_value=0)
        dfh = harian.rename('Revenue').to_frame()
        P_G, P_N = "Periode Gajian (25–5)", "Periode Normal (6–24)"
        dfh['PERIODE'] = np.where((dfh.index.day >= 25) | (dfh.index.day <= 5), P_G, P_N)
        g = dfh.loc[dfh['PERIODE'] == P_G, 'Revenue']; n = dfh.loc[dfh['PERIODE'] == P_N, 'Revenue']
        nilai_g, nilai_n = g.mean(), n.mean()
        selisih = ((nilai_g - nilai_n) / nilai_n * 100) if nilai_n else 0
        try:
            from scipy.stats import mannwhitneyu
            p_val = mannwhitneyu(g, n, alternative='two-sided').pvalue
        except Exception:
            p_val = None

        c1, c2, c3 = st.columns(3)
        c1.metric(f"Revenue/hari — Gajian ({len(g)} hari)", format_rupiah(nilai_g))
        c2.metric(f"Revenue/hari — Normal ({len(n)} hari)", format_rupiah(nilai_n))
        c3.metric("Selisih Gajian vs Normal", f"{selisih:+.1f}%".replace(".", ","))

        catatan = ""
        if p_val is not None:
            catatan = (" Uji statistik (Mann-Whitney) menunjukkan selisih ini "
                       + ("tidak signifikan" if p_val >= 0.05 else "signifikan") + f" (p = {p_val:.2f}).")
        if selisih > 5:
            st.success(f"**Mengapa berbeda?** Revenue harian pada periode gajian **{selisih:.1f}% lebih tinggi** dari periode normal, "
                       f"sehingga pembelian pelanggan cenderung meningkat di sekitar tanggal gajian.{catatan} "
                       f"**Rekomendasi:** tambah stok produk terlaris menjelang tanggal 25.")
        elif selisih < -5:
            st.info(f"**Mengapa berbeda?** Revenue harian pada periode gajian justru **{abs(selisih):.1f}% lebih rendah** dari periode normal. "
                    f"Artinya penjualan pelumas tidak terdorong oleh periode gajian; pembelian pelanggan lebih mengikuti "
                    f"kebutuhan operasional usahanya.{catatan}")
        else:
            st.info(f"**Mengapa tidak jauh berbeda?** Selisih revenue harian kedua periode di bawah 5%, sehingga tidak cukup "
                    f"kuat untuk disimpulkan sebagai pola berbeda.{catatan}")


# ════════════════════════════════════════════════════════════════
# BAGIAN 5 — HALAMAN: SEGMENTASI PRODUK
# ════════════════════════════════════════════════════════════════

def halaman_segmentasi_produk():
    render_brand_header()
    st.title("Segmentasi Produk")
    info_box(
        "Halaman ini menampilkan <b>dua hasil analisis independen</b>: RFM Analysis (rule-based, 5 segmen) "
        "dan K-Means Clustering (unsupervised, Kelompok A, B, C, …). K-Means memakai tiga fitur: "
        "<b>Recency</b> (hari sejak terakhir terjual, makin kecil makin baik), <b>Frequency</b> (jumlah invoice) "
        "dan <b>Monetary</b> (total nilai penjualan). Label klaster menggambarkan rata-rata, "
        "bukan sifat tiap produk."
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
            st.plotly_chart(donat(data["segment_summary"], "Segment_RFM", "Jumlah_Produk", WARNA_SEGMEN, URUTAN_SEGMEN),
                            use_container_width=True)
        with colB:
            st.markdown("##### Karakteristik Rata-Rata Tiap Segmen")
            seg_urut = data["segment_summary"].set_index('Segment_RFM').reindex(
                [s for s in URUTAN_SEGMEN if s in data["segment_summary"]['Segment_RFM'].values])
            st.dataframe(format_tabel_rupiah(seg_urut[['Jumlah_Produk', 'Recency', 'Frequency', 'Monetary']], ['Monetary']),
                         use_container_width=True)

        st.markdown("##### Perbandingan Karakteristik (Chart)")
        colx1, colx2, colx3 = st.columns(3)
        with colx1:
            st.plotly_chart(bar_warna(seg_urut, 'Recency', "Recency (hari, rendah = baik)", WARNA_SEGMEN), use_container_width=True)
        with colx2:
            st.plotly_chart(bar_warna(seg_urut, 'Frequency', "Frequency (tinggi = baik)", WARNA_SEGMEN), use_container_width=True)
        with colx3:
            st.plotly_chart(bar_warna(seg_urut, 'Monetary', "Monetary (tinggi = baik)", WARNA_SEGMEN), use_container_width=True)

        iv = data.get("interval_rfm")
        if iv is not None and not iv.empty:
            st.markdown("##### Batas Interval Skor R, F, M")
            st.caption("Setiap nilai hanya masuk ke satu skor. Batas ditentukan dari persentil 20/40/60/80 data. "
                       "Recency dibalik: hari lebih sedikit = skor lebih tinggi.")
            ci = st.columns(3)
            for kol, komp in zip(ci, ['Recency', 'Frequency', 'Monetary']):
                with kol:
                    st.markdown(f"**{komp}**")
                    st.dataframe(iv[iv['Komponen'] == komp][['Skor', 'Interval', 'Jumlah']]
                                 .rename(columns={'Jumlah': 'Jumlah Produk'}),
                                 hide_index=True, use_container_width=True)

        st.caption(
            "Skor total R+F+M berkisar 3–15. Pembagian segmen: Champion (≥13), High Performer (10–12), "
            "Growing (7–9), At Risk (4–6), Dormant (<4). Batas ini ditetapkan peneliti untuk kebutuhan interpretasi bisnis."
        )

        with st.expander("Lihat detail data produk per segmen"):
            filter_segmen = st.multiselect("Filter segmen", URUTAN_SEGMEN, default=URUTAN_SEGMEN, key="filter_rfm_produk")
            tampil = rfm[rfm['Segment_RFM'].isin(filter_segmen)].sort_values('RFM_Score', ascending=False)
            st.dataframe(format_tabel_rupiah(tampil[['ITEM', 'Recency', 'Frequency', 'Monetary', 'RFM_Score', 'Segment_RFM']], ['Monetary']),
                         use_container_width=True, hide_index=True)

    # ── TAB K-MEANS ──
    with tab_kmeans:
        info_box(
            "<b>Cara membaca klaster:</b> label seperti \"Recency rata-rata paling baru\" menggambarkan "
            "<b>rata-rata</b> klaster, bukan sifat setiap produk di dalamnya. Nilai tiap produk bisa "
            "berbeda dan saling tumpang tindih antar-klaster, terutama pada Monetary. Lihat kolom "
            "<i>median</i> dan <i>min/max</i> pada tabel profil di bawah."
        )
        urut_k = sorted(rfm["Segment_Cluster"].unique())
        st.markdown("##### Sebaran Produk per Klaster (3 Dimensi)")
        st.caption("Sumbu: Recency, Frequency, Monetary — sama dengan fitur yang dipakai K-Means. "
                   "Ukuran titik = total kuantitas terjual. Putar grafik untuk melihat dari sudut lain.")
        st.plotly_chart(px.scatter_3d(rfm, x="Recency", y="Frequency", z="Monetary", color="Segment_Cluster",
                                      size="Total_QTY", hover_name="ITEM", color_discrete_map=WARNA_KLASTER,
                                      category_orders={"Segment_Cluster": urut_k}),
                        use_container_width=True)
        with st.expander("Lihat proyeksi 2 dimensi"):
            for sx, sy in [("Recency", "Frequency"), ("Recency", "Monetary"), ("Frequency", "Monetary")]:
                st.plotly_chart(px.scatter(rfm, x=sx, y=sy, color="Segment_Cluster", size="Total_QTY",
                                           hover_name="ITEM", log_y=(sy == "Monetary"),
                                           color_discrete_map=WARNA_KLASTER,
                                           category_orders={"Segment_Cluster": urut_k}, title=f"{sx} vs {sy}"),
                                use_container_width=True)

        st.markdown("##### Profil dan Peringkat Tiap Klaster")
        cp = data["cluster_profile"]
        kolom_profil = [c for c in ['Segment_Cluster', 'Peringkat_Cluster', 'Karakteristik_Cluster', 'Jumlah_Produk',
                                     'Recency', 'Recency_Median', 'Frequency', 'Frequency_Median',
                                     'Monetary', 'Monetary_Median', 'Monetary_Min', 'Monetary_Max',
                                     'Total_QTY', 'Avg_Disc'] if c in cp.columns]
        st.dataframe(format_tabel_rupiah(cp.sort_values("Peringkat_Cluster")[kolom_profil], KOLOM_RUPIAH_KLASTER),
                     use_container_width=True, hide_index=True)
        st.caption("Kolom Karakteristik menjelaskan kedudukan rata-rata tiap klaster dibanding klaster lain "
                   "(peringkat antar-klaster). Peringkat klaster dihitung dari gabungan peringkat Recency (dibalik), "
                   "Frequency, dan Monetary. Gunakan median dan min/max untuk melihat sebaran nilai produk sebenarnya.")

        urutan_klaster = cp.sort_values("Peringkat_Cluster")["Segment_Cluster"].tolist()
        cp_idx = cp.set_index("Segment_Cluster").reindex(urutan_klaster)
        colc1, colc2, colc3 = st.columns(3)
        with colc1:
            st.plotly_chart(bar_warna(cp_idx, 'Recency', "Rata-rata Recency per Klaster (rendah = baik)", WARNA_KLASTER),
                            use_container_width=True)
        with colc2:
            st.plotly_chart(bar_warna(cp_idx, 'Frequency', "Rata-rata Frequency per Klaster", WARNA_KLASTER),
                            use_container_width=True)
        with colc3:
            st.plotly_chart(bar_warna(cp_idx, 'Monetary', "Rata-rata Monetary per Klaster", WARNA_KLASTER),
                            use_container_width=True)

        # Fitur yang paling menentukan klaster
        pf = data.get("pengaruh_fitur")
        if pf is not None and not pf.empty:
            st.markdown("##### Fitur yang Paling Menentukan Klaster")
            st.caption("Setiap fitur dibuang satu per satu lalu K-Means dijalankan ulang, dan hasilnya dibandingkan "
                       "dengan klaster asli memakai Adjusted Rand Index (ARI). ARI mendekati 1 berarti klaster hampir "
                       "tidak berubah, jadi fitur itu kurang berpengaruh. ARI yang rendah berarti fitur itu menentukan.")
            st.dataframe(pf, hide_index=True, use_container_width=True)

        st.markdown("##### Mengapa produk ini masuk klaster tersebut?")
        if "Alasan_Klaster" in rfm.columns:
            pilih = st.selectbox("Pilih produk", sorted(rfm["ITEM"].unique()), key="pilih_alasan_klaster")
            b = rfm[rfm["ITEM"] == pilih].iloc[0]
            st.markdown(f"**{pilih}** masuk **{b['Segment_Cluster']}**.")
            st.write(b["Alasan_Klaster"])
            if "Silhouette_Produk" in rfm.columns:
                st.caption(f"Skor silhouette produk = {b['Silhouette_Produk']:.2f} (mendekati 1 = sangat cocok di klasternya; "
                           f"di bawah 0,25 = produk batas yang dekat dengan klaster lain).")

        if "aturan_klaster" in data and not data["aturan_klaster"].empty:
            with st.expander("Ringkasan aturan interval klaster (decision tree)"):
                st.caption("K-Means mengelompokkan berdasarkan jarak, bukan ambang. Aturan berikut hanya ringkasan "
                           "yang meniru hasil K-Means, dengan tingkat kecocokan seperti pada judul kolom. "
                           "Perhatikan fitur mana yang muncul pada aturan: itulah yang paling membedakan klaster.")
                st.code("\n".join(data["aturan_klaster"].iloc[:, 0].fillna("").astype(str).tolist()))

        with st.expander("Kenapa klaster diberi nama 'Kelompok A/B/C…', bukan 'Champion' seperti RFM?"):
            st.markdown("""
            K-Means murni **unsupervised**: algoritma mengelompokkan produk berdasarkan kemiripan
            **Recency, Frequency, dan Monetary** (ternormalisasi), tanpa memakai Segment_RFM sebagai input.

            Nama diberikan **setelah** klaster terbentuk: (1) hitung rata-rata tiap fitur per klaster,
            (2) hitung peringkat performa dari gabungan peringkat ketiga fitur, (3) beri nama Kelompok A, B, C, …
            sesuai peringkat. Nama sengaja berbeda dari istilah RFM agar kedua hasil analisis tidak tertukar.

            **Catatan penting:** peringkat dan label menggambarkan *rata-rata* klaster. Karena itu sebuah produk
            di Kelompok A bisa saja bernilai Monetary rendah bila Recency dan Frequency-nya mirip anggota
            Kelompok A lainnya.
            """)

        with st.expander("Lihat detail data produk per klaster"):
            filter_klaster = st.multiselect("Filter klaster", urutan_klaster, default=urutan_klaster, key="filter_kmeans_produk")
            tampil_k = rfm[rfm['Segment_Cluster'].isin(filter_klaster)].sort_values('Peringkat_Cluster')
            kolom_k = [c for c in ['ITEM', 'Segment_Cluster', 'Recency', 'Frequency', 'Monetary', 'Silhouette_Produk', 'Alasan_Klaster']
                       if c in tampil_k.columns]
            st.dataframe(format_tabel_rupiah(tampil_k[kolom_k], ['Monetary']), use_container_width=True, hide_index=True)

    # ── TAB VALIDASI SILANG ──
    with tab_validasi:
        info_box(
            "Tabel ini membandingkan dua metode yang bekerja independen — K-Means <b>tidak</b> menggunakan "
            "Segment_RFM sebagai input. Kesesuaian hasil keduanya adalah gambaran konsistensi, "
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
                colr3.metric("Kesesuaian", f"{persen:.1f}%".replace(".", ","))
                teks = (f"Dari {int(champion_total)} produk segmen 'Champion' (RFM), {int(champion_selaras)} produk "
                        f"({persen:.1f}%) juga masuk **{klaster_terbaik}**, klaster berperingkat tertinggi hasil K-Means.")
                if persen >= 50:
                    st.success(teks + " Kedua metode relatif sejalan dalam mengenali produk berkinerja terbaik.")
                else:
                    st.info(teks + " Kesesuaiannya rendah: kedua metode memakai dasar yang berbeda "
                                   "(skor kuantil vs jarak antar-produk), sehingga hasilnya tidak harus sama.")

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
                        'Monetary Produk': format_rupiah_penuh(row['Monetary']),
                        'Median Segmen': format_rupiah_penuh(median_segmen),
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
        "Analisis RFM diterapkan pada pelanggan sebagai pelengkap segmentasi produk. "
        "Skor dan batas intervalnya dihitung dengan cara yang sama seperti pada produk."
    )
    data = load_data()
    rfm_customer = data["rfm_customer"]

    if rfm_customer.empty:
        st.warning("Data segmentasi customer belum tersedia pada berkas Excel saat ini.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Customer", _id(rfm_customer['NAMA_CUSTOMER'].nunique()))
    c2.metric("Customer Champion", int((rfm_customer['Segment_Customer'] == 'Champion').sum()))
    c3.metric("Customer At Risk", int((rfm_customer['Segment_Customer'] == 'At Risk').sum()))

    st.markdown('<hr class="header-garis">', unsafe_allow_html=True)
    col_left, col_right = st.columns([1, 1.2])
    with col_left:
        st.subheader("Distribusi Segmen Customer")
        seg_count = rfm_customer['Segment_Customer'].value_counts().rename_axis('Segmen').reset_index(name='Jumlah Customer')
        st.plotly_chart(donat(seg_count, 'Segmen', 'Jumlah Customer', WARNA_SEGMEN, URUTAN_SEGMEN), use_container_width=True)
    with col_right:
        st.subheader("Sebaran Customer: Frequency vs Monetary")
        st.plotly_chart(px.scatter(rfm_customer, x='Frequency', y='Monetary', color='Segment_Customer',
                                   hover_name='NAMA_CUSTOMER', log_y=True, color_discrete_map=WARNA_SEGMEN,
                                   category_orders={'Segment_Customer': URUTAN_SEGMEN}),
                        use_container_width=True)

    st.markdown("##### Karakteristik Rata-Rata Tiap Segmen Customer")
    kar = rfm_customer.groupby('Segment_Customer').agg(
        Jumlah_Customer=('NAMA_CUSTOMER', 'count'), Recency=('Recency', 'mean'),
        Frequency=('Frequency', 'mean'), Monetary=('Monetary', 'mean')).round(2)
    kar = kar.reindex([s for s in URUTAN_SEGMEN if s in kar.index])
    st.dataframe(format_tabel_rupiah(kar, ['Monetary']), use_container_width=True)

    colx1, colx2, colx3 = st.columns(3)
    with colx1:
        st.plotly_chart(bar_warna(kar, 'Recency', "Recency (hari, rendah = baik)", WARNA_SEGMEN), use_container_width=True)
    with colx2:
        st.plotly_chart(bar_warna(kar, 'Frequency', "Frequency (tinggi = baik)", WARNA_SEGMEN), use_container_width=True)
    with colx3:
        st.plotly_chart(bar_warna(kar, 'Monetary', "Monetary (tinggi = baik)", WARNA_SEGMEN), use_container_width=True)

    iv = data.get("interval_rfm_cust")
    if iv is not None and not iv.empty:
        st.markdown("##### Batas Interval Skor R, F, M (Customer)")
        ci = st.columns(3)
        for kol, komp in zip(ci, ['Recency', 'Frequency', 'Monetary']):
            with kol:
                st.markdown(f"**{komp}**")
                st.dataframe(iv[iv['Komponen'] == komp][['Skor', 'Interval', 'Jumlah']]
                             .rename(columns={'Jumlah': 'Jumlah Customer'}), hide_index=True, use_container_width=True)

    info_box("<b>Kriteria segmen:</b> Champion (skor RFM ≥13), High Performer (10–12), Growing (7–9), "
             "At Risk (4–6), Dormant (&lt;4), dari total skor Recency + Frequency + Monetary (masing-masing 1–5).")

    # ── Produk yang dibeli tiap segmen ──
    st.markdown('<hr class="header-garis">', unsafe_allow_html=True)
    st.subheader("Produk yang Dibeli per Segmen Customer")
    seg_tersedia = [s for s in URUTAN_SEGMEN if s in rfm_customer['Segment_Customer'].unique()]
    seg_pilih = st.selectbox("Pilih segmen customer", seg_tersedia, key="seg_cust_produk")
    nama_seg = set(rfm_customer.loc[rfm_customer['Segment_Customer'] == seg_pilih, 'NAMA_CUSTOMER'])
    trx_seg = data["transaksi"][data["transaksi"]['NAMA_CUSTOMER'].isin(nama_seg)]

    if trx_seg.empty:
        st.info("Tidak ada transaksi untuk segmen ini.")
    else:
        colp1, colp2 = st.columns(2)
        with colp1:
            st.markdown("**5 produk dengan revenue terbesar**")
            tp5 = trx_seg.groupby('ITEM')['HARGA_JUAL'].sum().nlargest(5).reset_index()
            tp5['Revenue (Rp juta)'] = tp5['HARGA_JUAL'] / 1e6
            fig = px.bar(tp5, x='Revenue (Rp juta)', y='ITEM', orientation='h',
                         color_discrete_sequence=[WARNA_SEGMEN[seg_pilih]])
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, separators=",.")
            st.plotly_chart(fig, use_container_width=True)
        with colp2:
            st.markdown("**5 customer terbesar di segmen ini**")
            tc = rfm_customer[rfm_customer['Segment_Customer'] == seg_pilih].nlargest(5, 'Monetary')
            st.dataframe(format_tabel_rupiah(tc[['NAMA_CUSTOMER', 'Frequency', 'Monetary']], ['Monetary']),
                         hide_index=True, use_container_width=True)
        st.success(f"Segmen **{seg_pilih}** beranggotakan **{len(nama_seg)} customer** dengan produk revenue terbesar "
                   f"**{tp5.iloc[0]['ITEM']}** ({format_rupiah(tp5.iloc[0]['HARGA_JUAL'])}).")

    st.markdown("##### Detail Data Customer")
    filter_segmen = st.multiselect("Filter berdasarkan segmen", URUTAN_SEGMEN, default=URUTAN_SEGMEN, key="filter_cust")
    tampil = rfm_customer[rfm_customer['Segment_Customer'].isin(filter_segmen)].sort_values('RFM_Score', ascending=False)
    st.dataframe(format_tabel_rupiah(tampil[['NAMA_CUSTOMER', 'Recency', 'Frequency', 'Monetary', 'RFM_Score', 'Segment_Customer']], ['Monetary']),
                 use_container_width=True, hide_index=True)


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
    c2.metric("Lift Maksimum", f"{data['kpi']['max_lift']:.2f}".replace(".", ","))
    c3.metric("Rata-rata Lift", f"{rules['lift'].mean():.2f}".replace(".", ","))

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
            if hasil.get("alasan"):
                with st.expander("Mengapa produk ini masuk klaster tersebut?"):
                    st.write(hasil["alasan"])

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
    info_box(
        "Pilih dua produk untuk melihat seberapa sering keduanya dibeli dalam invoice yang sama. "
        "<b>Support</b> = persentase invoice yang memuat keduanya; <b>Confidence</b> = dari invoice yang memuat produk pertama, "
        "persentase yang juga memuat produk kedua; <b>Lift</b> di atas 1 berarti keduanya lebih sering dibeli bersama "
        "daripada kebetulan. Angka ini bisa dipakai untuk menimbang paket bundling."
    )
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
                st.caption("Kombinasi ini tidak lolos ambang Apriori (support minimal 2%), sehingga angkanya dihitung langsung dari invoice.")
            else:
                st.markdown('<span class="badge-warn">Tidak pernah dibeli bersama</span>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# BAGIAN 10 — HALAMAN: PETA KORELASI
# ════════════════════════════════════════════════════════════════

def halaman_peta_korelasi():
    render_brand_header()
    st.title("Peta Korelasi Antar Produk")
    info_box(
        "Peta ini menunjukkan seberapa sering dua produk muncul bersamaan dalam invoice. "
        "Warna merah = cenderung dibeli bersama, biru = cenderung tidak dibeli bersama, putih = tidak ada hubungan. "
        "Nilainya antara -1 sampai 1."
    )
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
    info_box(
        "Pilih produk yang dipesan pelanggan, lalu sistem menyarankan produk tambahan yang sering dibeli bersamanya "
        "berdasarkan pola pembelian (Association Rules). Makin tinggi <b>lift</b> dan <b>confidence</b>, makin kuat sarannya."
    )
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

    # Tahap 1: membaca berkas — error di sini murni soal format file
    try:
        df_upload, format_terdeteksi = baca_berkas_upload(berkas)
    except Exception as e:
        st.error(f"Gagal membaca berkas: {e}")
        return

    if df_upload is None:
        st.error(
            "Format berkas tidak dikenali. Pastikan berkas memiliki kolom wajib "
            f"({', '.join(KOLOM_WAJIB)}) atau merupakan ekspor data mentah asli dengan 4 baris header."
        )
        return

    st.success(f"Format terdeteksi: {'data mentah asli' if format_terdeteksi=='mentah_asli' else 'data sudah rapi'}.")
    st.dataframe(df_upload.head(10), use_container_width=True)

    # Tahap 2: menjalankan analisis — error ditampilkan apa adanya
    if st.button("Jalankan Analisis Lengkap", type="primary"):
        with st.spinner("Menjalankan pipeline segmentasi produk dan customer..."):
            try:
                hasil_produk = jalankan_pipeline_baru(df_upload)
                hasil_cust = jalankan_pipeline_customer_baru(df_upload)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat menjalankan analisis: {e}")
                st.caption(
                    "Kemungkinan penyebab: format kolom TANGGAL tidak dikenali, kolom QTY/HARGA_JUAL "
                    "berisi teks bukan angka, atau jumlah produk/customer unik terlalu sedikit."
                )
                return
            st.session_state["hasil_upload_produk"] = hasil_produk
            st.session_state["hasil_upload_customer"] = hasil_cust

    hasil = st.session_state.get("hasil_upload_produk")
    hasil_cust = st.session_state.get("hasil_upload_customer")

    if hasil is None:
        return

    if hasil.get("error"):
        st.error(hasil["error"])
        return

    st.success("Analisis berhasil dijalankan.")
    st.markdown("#### Funnel Pembersihan Data")
    st.dataframe(hasil["funnel"], use_container_width=True, hide_index=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baris Bersih", _id(hasil['jumlah_baris']))
    c2.metric("Produk Unik", hasil["jumlah_produk"])
    c3.metric("Invoice Unik", _id(hasil['jumlah_invoice']))
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
        st.plotly_chart(px.scatter(rfm_baru, x="Frequency", y="Monetary", color="Segment_Cluster", size="Total_QTY",
                                   hover_name="ITEM", log_y=True, color_discrete_map=WARNA_KLASTER), use_container_width=True)
        st.dataframe(format_tabel_rupiah(rfm_baru, ['Monetary']), use_container_width=True)
        st.download_button("Unduh Hasil RFM Produk (CSV)", rfm_baru.to_csv(index=False).encode("utf-8"), "hasil_rfm_produk_baru.csv", "text/csv")

    with tabs[2]:
        if hasil_cust and not hasil_cust.get("error"):
            rfm_cust_baru = hasil_cust["rfm_customer"]
            st.metric("Jumlah Customer", hasil_cust["jumlah_customer"])
            seg_count = rfm_cust_baru['Segment_Customer'].value_counts().rename_axis('Segmen').reset_index(name='Jumlah')
            st.plotly_chart(donat(seg_count, 'Segmen', 'Jumlah', WARNA_SEGMEN, URUTAN_SEGMEN), use_container_width=True)
            st.dataframe(format_tabel_rupiah(rfm_cust_baru, ['Monetary']), use_container_width=True)
            st.download_button("Unduh Hasil RFM Customer (CSV)", rfm_cust_baru.to_csv(index=False).encode("utf-8"), "hasil_rfm_customer_baru.csv", "text/csv")
        else:
            pesan_error_cust = hasil_cust.get("error") if hasil_cust else "Segmentasi customer tidak dapat dijalankan."
            st.info(pesan_error_cust)

    with tabs[3]:
        crosstab_baru = hasil.get("crosstab")
        if crosstab_baru is not None and not crosstab_baru.empty:
            st.plotly_chart(px.imshow(crosstab_baru, text_auto=True, color_continuous_scale="Reds"), use_container_width=True)

    with tabs[4]:
        st.metric("Jumlah Rules", hasil["jumlah_rules"])
        if not rules_baru.empty:
            st.dataframe(hasil["rules_rapi"][["antecedents", "consequents", "support", "confidence", "lift"]], use_container_width=True)
        else:
            st.info("Tidak ditemukan association rules dari data yang diunggah (kombinasi produk terlalu jarang muncul bersamaan).")

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
