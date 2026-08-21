"""
Dashboard Analisis Penjualan PT Duta Buana Perkasa
====================================================
Aplikasi web interaktif berbasis Streamlit untuk menyajikan hasil
analisis RFM, K-Means Clustering, dan Apriori Association Rules.

VERSI SATU FILE — semua styling, pipeline data, dan halaman digabung
di sini. Navigasi antar halaman memakai sidebar radio (bukan sistem
pages/ bawaan Streamlit), supaya tidak bergantung pada struktur folder.
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
# BAGIAN 1 — STYLING (dulunya styling.py)
# ════════════════════════════════════════════════════════════════

MERAH_UTAMA = "#C62828"
MERAH_GELAP = "#A81E1E"
MERAH_MUDA  = "#E57373"
ABU_TEKS    = "#6B7280"
ABU_TERANG  = "#F2F3F5"

PALET_GRADASI = ["#C62828", "#D84A4A", "#E06B6B", "#EB9090", "#F3C0C0"]


def apply_custom_theme():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {ABU_TERANG};
        }}

        /* ── Sembunyikan elemen bawaan yang tidak perlu ── */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}

        /* ── Metric / Card KPI ── */
        [data-testid="stMetricValue"] {{
            color: {MERAH_UTAMA};
            font-weight: 700;
            font-size: 26px;
        }}
        [data-testid="stMetricLabel"] {{
            color: {ABU_TEKS};
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 12px;
        }}
        [data-testid="stMetric"] {{
            background-color: white;
            border: 1.5px solid {MERAH_UTAMA};
            border-radius: 14px;
            padding: 20px 18px;
            box-shadow: 0 4px 14px rgba(198, 40, 40, 0.12);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}
        [data-testid="stMetric"]:hover {{
            transform: translateY(-4px);
            box-shadow: 0 12px 26px rgba(198, 40, 40, 0.24);
            border-color: {MERAH_GELAP};
        }}

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {MERAH_UTAMA} 0%, {MERAH_GELAP} 100%);
        }}
        section[data-testid="stSidebar"] * {{
            color: white !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stPageLink"] {{
            border-radius: 8px;
            margin: 2px 0;
            transition: background-color 0.15s ease;
        }}
        section[data-testid="stSidebar"] [data-testid="stPageLink"]:hover {{
            background-color: rgba(255,255,255,0.15);
        }}

        /* ── Tombol navigasi di sidebar (pengganti pages/) ── */
        section[data-testid="stSidebar"] .stButton>button {{
            background-color: rgba(255,255,255,0.08) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            text-align: left !important;
            justify-content: flex-start !important;
            font-weight: 500 !important;
            padding: 10px 14px !important;
            margin-bottom: 4px !important;
            box-shadow: none !important;
            width: 100%;
        }}
        section[data-testid="stSidebar"] .stButton>button:hover {{
            background-color: rgba(255,255,255,0.22) !important;
            transform: none !important;
            box-shadow: none !important;
        }}
        section[data-testid="stSidebar"] .stButton>button p {{
            font-weight: 500 !important;
        }}
        /* Menu yang sedang aktif */
        section[data-testid="stSidebar"] .nav-aktif .stButton>button {{
            background-color: white !important;
            color: {MERAH_UTAMA} !important;
            font-weight: 700 !important;
        }}
        section[data-testid="stSidebar"] .nav-aktif .stButton>button p {{
            color: {MERAH_UTAMA} !important;
            font-weight: 700 !important;
        }}

        /* ── Tombol ── */
        .stButton>button {{
            background-color: {MERAH_UTAMA};
            color: white;
            border-radius: 8px;
            border: none;
            font-weight: 600;
            padding: 8px 20px;
            transition: all 0.15s ease;
        }}
        .stButton>button:hover {{
            background-color: {MERAH_GELAP};
            transform: translateY(-2px);
            box-shadow: 0 6px 14px rgba(198, 40, 40, 0.35);
        }}

        /* ── Tabs ── */
        div[data-baseweb="tab-list"] button[aria-selected="true"] {{
            color: {MERAH_UTAMA};
            border-bottom-color: {MERAH_UTAMA};
        }}

        /* ── Dataframe ── */
        [data-testid="stDataFrame"] {{
            border: 1px solid #E5E7EB;
            border-radius: 10px;
            overflow: hidden;
        }}

        /* ══════════════════════════════════════════════ */
        /* ── DROPDOWN: selectbox & multiselect (kotak tertutup) ── */
        /* ══════════════════════════════════════════════ */
        [data-baseweb="select"] > div {{
            background-color: white !important;
            color: #1F2937 !important;
            border: 1.5px solid {MERAH_UTAMA} !important;
            border-radius: 8px !important;
        }}
        [data-baseweb="select"] * {{
            color: #1F2937 !important;
        }}
        [data-baseweb="select"] input {{
            color: #1F2937 !important;
        }}
        [data-baseweb="select"] svg {{
            fill: {MERAH_UTAMA} !important;
        }}

        /* Background putih penuh untuk seluruh bagian dalam dropdown */
        [data-baseweb="select"] {{
            background-color: white !important;
        }}
        [data-baseweb="select"] div {{
            background-color: white !important;
        }}
        [data-testid="stSelectbox"] > div > div {{
            background-color: white !important;
        }}
        [data-testid="stMultiSelect"] > div > div {{
            background-color: white !important;
        }}
        [data-testid="stSelectbox"] label {{
            color: #1F2937 !important;
        }}
        [data-testid="stMultiSelect"] label {{
            color: #1F2937 !important;
        }}

        /* ── Dropdown: panel menu yang muncul saat diklik ── */
        [data-baseweb="popover"] [data-baseweb="menu"] {{
            background-color: white !important;
        }}
        [data-baseweb="popover"] [data-baseweb="menu"] li {{
            color: #1F2937 !important;
            background-color: white !important;
        }}
        [data-baseweb="popover"] [data-baseweb="menu"] li:hover {{
            background-color: {ABU_TERANG} !important;
            color: {MERAH_UTAMA} !important;
        }}

        /* ── Dropdown: item terpilih di multiselect (tag/chip) ── */
        [data-baseweb="tag"] {{
            background-color: {MERAH_UTAMA} !important;
            color: white !important;
        }}
        [data-baseweb="tag"] span {{
            color: white !important;
        }}
        [data-baseweb="tag"] svg {{
            fill: white !important;
        }}

        /* ── Slider ── */
        [data-testid="stSlider"] [role="slider"] {{
            background-color: {MERAH_UTAMA} !important;
        }}
        .stSlider [data-baseweb="slider"] > div > div {{
            background-color: {MERAH_UTAMA} !important;
        }}

        /* ── Hero banner ── */
        .hero-banner {{
            position: relative;
            border-radius: 18px;
            overflow: hidden;
            box-shadow: 0 10px 28px rgba(0,0,0,0.28);
            margin-bottom: 26px;
            height: 380px;
        }}
        .hero-banner img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            object-position: center 90%;
            display: block;
        }}
        .hero-overlay {{
            position: absolute;
            bottom: 0; left: 0; right: 0;
            background: linear-gradient(0deg, rgba(0,0,0,0.88) 0%, rgba(0,0,0,0.35) 60%, transparent 100%);
            padding: 36px 32px 26px 32px;
        }}
        .hero-overlay h1 {{
            color: white;
            font-size: 32px;
            font-weight: 800;
            margin: 0 0 6px 0;
            letter-spacing: 0.5px;
        }}
        .hero-overlay p {{
            color: #E8E8E8;
            font-size: 14px;
            margin: 0;
        }}
        .hero-badge {{
            display: inline-block;
            background-color: {MERAH_UTAMA};
            color: white;
            font-size: 11px;
            font-weight: 700;
            padding: 5px 12px;
            border-radius: 20px;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }}

        /* ── Brand header (logo + judul kecil di atas tiap halaman) ── */
        .brand-header {{
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 6px 0 18px 0;
        }}
        .brand-header .logo-box {{
            width: 46px;
            height: 46px;
            border-radius: 10px;
            background: linear-gradient(135deg, {MERAH_UTAMA}, {MERAH_GELAP});
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 800;
            font-size: 18px;
            box-shadow: 0 4px 10px rgba(198,40,40,0.3);
        }}
        .brand-header .brand-text h2 {{
            margin: 0;
            color: {MERAH_UTAMA};
            font-size: 20px;
            font-weight: 800;
        }}
        .brand-header .brand-text p {{
            margin: 0;
            color: {ABU_TEKS};
            font-size: 12px;
        }}

        /* ── Divider custom ── */
        .header-garis {{
            border: none;
            border-top: 2px solid #F1D4D4;
            margin: 22px 0;
        }}

        /* ── Kartu insight ── */
        .kartu-insight {{
            background: white;
            border-left: 4px solid {MERAH_UTAMA};
            border-radius: 10px;
            padding: 14px 16px;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 14px;
            color: #1F2937;
        }}
        .kartu-insight .ikon {{
            font-size: 20px;
            flex-shrink: 0;
        }}

        /* ── Grid menu di halaman utama ── */
        .menu-item {{
            background: white;
            border: 1px solid #EDEDED;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            transition: all 0.15s ease;
            height: 90px;
        }}
        .menu-item:hover {{
            border-color: {MERAH_UTAMA};
            box-shadow: 0 6px 16px rgba(198,40,40,0.15);
            transform: translateY(-3px);
        }}
        .menu-item .judul {{
            font-weight: 700;
            color: {MERAH_UTAMA};
            font-size: 14px;
            margin-bottom: 4px;
        }}
        .menu-item .desk {{
            font-size: 12px;
            color: {ABU_TEKS};
            line-height: 1.4;
        }}

        /* ── Badge status ── */
        .badge-ok {{
            background-color: #DCFCE7;
            color: #15803D;
            padding: 5px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .badge-warn {{
            background-color: #FEF3C7;
            color: #B45309;
            padding: 5px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        /* ── Section header ── */
        .section-header {{
            border-left: 5px solid {MERAH_UTAMA};
            padding-left: 12px;
            margin: 28px 0 16px 0;
        }}
        .section-header h3 {{
            margin: 0;
            color: #1F2937;
        }}
        .section-header p {{
            margin: 2px 0 0 0;
            color: {ABU_TEKS};
            font-size: 13px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def render_brand_header():
    """Header kecil (logo + nama brand) yang tampil di bagian atas
    setiap halaman, di luar hero banner."""
    st.markdown(
        f"""
        <div class="brand-header">
            <div class="logo-box">D</div>
            <div class="brand-text">
                <h2>PT DUTA BUANA PERKASA</h2>
                <p>Dashboard Analisis Penjualan 2025</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_hero_banner():
    """Hero banner besar dengan foto produk dan overlay teks,
    dipakai khusus di halaman utama."""
    # PERBAIKAN: banner.jpg ada di root repo (sejajar app.py), bukan di folder data/
    banner_path = os.path.join(os.path.dirname(__file__), "banner.jpg")
    if not os.path.exists(banner_path):
        st.warning("Berkas banner.jpg tidak ditemukan — hero banner dilewati.")
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
        """,
        unsafe_allow_html=True
    )


def kartu_insight(teks_html: str, ikon: str = "info"):
    """Kartu insight bergaya card dengan aksen border merah dan ikon.
    ikon: 'up', 'down', 'person', 'flag', atau 'info'."""
    peta_ikon = {
        "up": "📈", "down": "📉", "person": "👤",
        "flag": "🚩", "info": "💡"
    }
    simbol = peta_ikon.get(ikon, "💡")
    st.markdown(
        f"""
        <div class="kartu-insight">
            <span class="ikon">{simbol}</span>
            <span>{teks_html}</span>
        </div>
        """,
        unsafe_allow_html=True
    )


def page_header(judul: str, subjudul: str = ""):
    st.markdown(
        f"""
        <div class="section-header">
            <h3>{judul}</h3>
            <p>{subjudul}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ════════════════════════════════════════════════════════════════
# BAGIAN 2 — PIPELINE DATA (dulunya pipeline.py)
# ════════════════════════════════════════════════════════════════

# PERBAIKAN: DATA_PATH dulunya "/content/Dashboard_streamlit.xlsx" (path
# khusus Google Colab). Diubah jadi path relatif terhadap lokasi app.py
# supaya jalan di Streamlit Cloud maupun lokal.
DATA_PATH = os.path.join(os.path.dirname(__file__), "Dashboard_streamlit.xlsx")
FEATS = ['Recency', 'Frequency', 'Monetary', 'Total_QTY', 'Avg_Disc']
MIN_SUPPORT, MIN_CONFIDENCE, MIN_LIFT = 0.02, 0.30, 1.2


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

    kpi = dict(zip(kpi_raw["Metric"], kpi_raw["Value"]))

    return {
        "transaksi": df_transaksi,
        "rfm": rfm,
        "segment_summary": seg_summary,
        "cluster_profile": cluster_prof,
        "rules": rules,
        "tren": tren,
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
    rfm = data["rfm"]
    rules = data["rules"]
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
        "recency": info_produk.get("Recency"),
        "frequency": info_produk.get("Frequency"),
        "monetary": info_produk.get("Monetary"),
        "rules_terkait": rules_terkait
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

    return {
        "both": both, "support": support,
        "conf_a_to_b": conf_a_to_b, "conf_b_to_a": conf_b_to_a,
        "lift": lift, "count_a": len(inv_a), "count_b": len(inv_b),
        "total_invoice": total_invoice,
    }


def simulasi_bundling(produk_a, produk_b, data):
    rules = data["rules"]
    match = rules[
        (rules["antecedents"].str.contains(produk_a, regex=False) &
         rules["consequents"].str.contains(produk_b, regex=False))
        |
        (rules["antecedents"].str.contains(produk_b, regex=False) &
         rules["consequents"].str.contains(produk_a, regex=False))
    ]
    if not match.empty:
        best = match.sort_values("lift", ascending=False).iloc[0]
        return {"status": "ditemukan_resmi", "support": best["support"],
                "confidence": best["confidence"], "lift": best["lift"]}

    co = hitung_co_occurrence(produk_a, produk_b, data["transaksi"])
    if co["both"] == 0:
        return {"status": "tidak_pernah_bareng", **co}
    return {"status": "dihitung_langsung", **co}


def hitung_matriks_korelasi(transaksi_df, top_n=15):
    top_items = transaksi_df.groupby("ITEM")["QTY"].sum().nlargest(top_n).index.tolist()
    subset = transaksi_df[transaksi_df["ITEM"].isin(top_items)]
    basket = (subset.groupby(["NO_INVOICE", "ITEM"])["QTY"]
              .sum().unstack(fill_value=0).astype(bool).astype(int))
    basket = basket.reindex(columns=top_items, fill_value=0)
    corr = basket.corr()
    return corr, top_items


def jalankan_pipeline_baru(df_upload):
    df = df_upload.copy()
    kolom_wajib = ['NO_INVOICE', 'TANGGAL', 'NO_CUSTOMER', 'NAMA_CUSTOMER',
                   'ITEM', 'QTY', 'HARGA_JUAL', 'DISKON_PCT']
    kolom_hilang = [k for k in kolom_wajib if k not in df.columns]
    if kolom_hilang:
        return {"error": f"Kolom wajib tidak ditemukan: {', '.join(kolom_hilang)}"}

    df = df.dropna(subset=['NO_INVOICE', 'ITEM', 'HARGA_JUAL'])
    df = df.drop_duplicates(subset=['NO_INVOICE', 'ITEM', 'QTY', 'HARGA_JUAL'], keep='first')
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], errors='coerce')
    for c in ['QTY', 'HARGA_JUAL', 'DISKON_PCT']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    df = df[(df['HARGA_JUAL'] > 0) & (df['QTY'] > 0)]
    df['ITEM'] = df['ITEM'].astype(str).str.strip().str.upper()

    if len(df) == 0:
        return {"error": "Setelah pembersihan, tidak ada baris data yang tersisa."}

    snapshot = df['TANGGAL'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('ITEM').agg(
        Recency=('TANGGAL', lambda x: (snapshot - x.max()).days),
        Frequency=('NO_INVOICE', 'nunique'),
        Monetary=('HARGA_JUAL', 'sum'),
        Total_QTY=('QTY', 'sum'),
        Avg_Disc=('DISKON_PCT', 'mean')
    ).reset_index()
    rfm['Avg_Disc'] = rfm['Avg_Disc'].fillna(0)

    if len(rfm) < 4:
        return {"error": "Jumlah produk unik terlalu sedikit untuk clustering (minimal 4)."}

    qt = QuantileTransformer(output_distribution='normal', random_state=42,
                              n_quantiles=min(100, len(rfm)))
    X_scaled = qt.fit_transform(rfm[FEATS])

    k_opt = min(4, len(rfm) - 1)
    kmeans = KMeans(n_clusters=k_opt, init='k-means++', random_state=42,
                     n_init=10, max_iter=1000)
    rfm['Cluster'] = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, rfm['Cluster']) if len(rfm) > k_opt else None

    basket = (df.groupby(['NO_INVOICE', 'ITEM'])['QTY']
                .sum().unstack(fill_value=0).astype(bool))
    frequent_itemsets = apriori(basket, min_support=MIN_SUPPORT,
                                 use_colnames=True, max_len=3)
    rules = pd.DataFrame()
    if not frequent_itemsets.empty:
        rules_all = association_rules(frequent_itemsets, metric='lift',
                                       min_threshold=MIN_LIFT)
        rules = rules_all[rules_all['confidence'] >= MIN_CONFIDENCE]
        rules = rules.sort_values('lift', ascending=False)

    return {
        "error": None, "jumlah_baris": len(df),
        "jumlah_produk": rfm['ITEM'].nunique(),
        "jumlah_invoice": df['NO_INVOICE'].nunique(),
        "rfm": rfm, "k_optimal": k_opt, "silhouette": sil,
        "jumlah_rules": len(rules), "rules": rules
    }


# ════════════════════════════════════════════════════════════════
# BAGIAN 3 — KONFIGURASI HALAMAN & NAVIGASI SIDEBAR
# ════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Dashboard PT Duta Buana Perkasa",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_theme()

MENU = [
    "Ringkasan",
    "Segmentasi",
    "Pola Beli",
    "Product Recommender",
    "Simulasi Bundling",
    "Peta Korelasi",
    "Upload Data",
]

if "halaman_aktif" not in st.session_state:
    st.session_state.halaman_aktif = MENU[0]

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 10px 0 20px 0;">
            <div style="font-size:15px; font-weight:800; letter-spacing:0.5px;">
                PT DUTA BUANA PERKASA
            </div>
            <div style="font-size:11px; opacity:0.85;">Dashboard Analisis Penjualan</div>
        </div>
        """,
        unsafe_allow_html=True
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
# BAGIAN 4 — HALAMAN: RINGKASAN (dulunya app.py + 1_Ringkasan.py)
# ════════════════════════════════════════════════════════════════

def halaman_ringkasan():
    render_brand_header()
    render_hero_banner()

    data = load_data()
    rfm = data["rfm"]
    transaksi = data["transaksi"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"Rp {data['kpi']['revenue']/1e9:.2f} M")
    col2.metric("Total Invoice", f"{data['kpi']['invoice']:,}".replace(",", "."))
    col3.metric("Produk Aktif", data['kpi']['produk'])
    col4.metric("Total Customer", data['kpi']['customer'])

    st.markdown('<hr class="header-garis">', unsafe_allow_html=True)

    st.markdown("### Insight Otomatis")
    st.caption("Ringkasan otomatis dari data historis, murni deskriptif bukan prediksi.")

    bulanan = transaksi.groupby("BULAN")["HARGA_JUAL"].sum()
    bulan_terbaik = bulanan.idxmax()
    bulan_terendah = bulanan.idxmin()
    top_salesman = transaksi.groupby("SALESMAN")["HARGA_JUAL"].sum().idxmax() if "SALESMAN" in transaksi.columns else "-"
    champion_count = int((rfm["Segment_RFM"] == "Champion").sum())
    tidur_count = int((rfm["Segment_Cluster"] == "Produk Tidur").sum())

    c1, c2 = st.columns(2)
    with c1:
        kartu_insight(f"Bulan penjualan tertinggi: <b>{bulan_terbaik}</b> (Rp {bulanan.max()/1e6:,.0f} Jt)", ikon="up")
        kartu_insight(f"Salesman dengan revenue tertinggi: <b>{top_salesman}</b>", ikon="person")
    with c2:
        kartu_insight(f"Bulan penjualan terendah: <b>{bulan_terendah}</b> (Rp {bulanan.min()/1e6:,.0f} Jt)", ikon="down")
        kartu_insight(f"<b>{champion_count} produk Champion</b> (RFM) dan <b>{tidur_count} produk Tidur</b> (K-Means) perlu perhatian khusus", ikon="flag")

    st.markdown('<hr class="header-garis">', unsafe_allow_html=True)

    st.title("Ringkasan Penjualan")

    col_left, col_right = st.columns([1.4, 1])
    with col_left:
        st.subheader("Tren Revenue Bulanan")
        fig_tren = px.line(data["tren"], x="Bulan", y="Revenue", markers=True, color_discrete_sequence=["#B91C1C"])
        st.plotly_chart(fig_tren, use_container_width=True)
    with col_right:
        st.subheader("Distribusi Segmen RFM")
        fig_donat = px.pie(data["segment_summary"], names="Segment_RFM", values="Jumlah_Produk", hole=0.5, color_discrete_sequence=PALET_GRADASI)
        st.plotly_chart(fig_donat, use_container_width=True)

    col_left2, col_right2 = st.columns(2)
    with col_left2:
        st.subheader("Top 5 Produk berdasarkan Revenue")
        top5 = transaksi.groupby("ITEM")["HARGA_JUAL"].sum().nlargest(5).reset_index()
        fig_bar = px.bar(top5, x="HARGA_JUAL", y="ITEM", orientation="h", color_discrete_sequence=["#B91C1C"])
        fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)
    with col_right2:
        st.subheader("Profil Klaster")
        st.dataframe(data["cluster_profile"][["Segment_Cluster", "Jumlah_Produk", "Monetary"]], use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════
# BAGIAN 5 — HALAMAN: SEGMENTASI (dulunya 2_Segmentasi.py)
# ════════════════════════════════════════════════════════════════

def halaman_segmentasi():
    render_brand_header()
    st.title("Segmentasi Produk - RFM dan K-Means")
    data = load_data()

    st.subheader("Sebaran Produk per Klaster")
    fig = px.scatter(
        data["rfm"], x="Frequency", y="Monetary", color="Segment_Cluster",
        size="Total_QTY", hover_name="ITEM",
        color_discrete_sequence=["#B91C1C", "#EF4444", "#F87171", "#94A3B8"]
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rata-rata Monetary per Segmen")
    fig2 = px.bar(data["segment_summary"], x="Monetary", y="Segment_RFM", orientation="h", color_discrete_sequence=["#B91C1C"])
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Detail Profil Klaster")
    st.dataframe(data["cluster_profile"], use_container_width=True)


# ════════════════════════════════════════════════════════════════
# BAGIAN 6 — HALAMAN: POLA BELI (dulunya 3_Pola_Beli.py)
# ════════════════════════════════════════════════════════════════

def halaman_pola_beli():
    render_brand_header()
    st.title("Pola Pembelian - Association Rules")
    data = load_data()
    rules = data["rules"]

    min_lift = st.slider("Filter minimum Lift", 1.0, float(rules["lift"].max()), 1.2, 0.1)
    min_conf = st.slider("Filter minimum Confidence", 0.0, 1.0, 0.3, 0.05)

    filtered = rules[(rules["lift"] >= min_lift) & (rules["confidence"] >= min_conf)]

    st.subheader(f"10 Rules dengan Lift Tertinggi ({len(filtered)} rules sesuai filter)")
    top10 = filtered.sort_values("lift", ascending=False).head(10)
    fig = px.bar(top10, x="lift", y="antecedents", orientation="h", color_discrete_sequence=["#B91C1C"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daftar Association Rules")
    st.dataframe(
        filtered[["antecedents", "consequents", "support", "confidence", "lift"]].sort_values("lift", ascending=False),
        use_container_width=True
    )


# ════════════════════════════════════════════════════════════════
# BAGIAN 7 — HALAMAN: PRODUCT RECOMMENDER (dulunya 4_Product_Recommender.py)
# ════════════════════════════════════════════════════════════════

def halaman_product_recommender():
    render_brand_header()
    st.title("Product Recommender")
    st.caption("Pilih satu produk untuk melihat segmen RFM, klaster K-Means, tren penjualannya, serta rekomendasi bundling berdasarkan hasil Apriori.")

    data = load_data()
    daftar_produk = sorted(data["rfm"]["ITEM"].unique())
    produk_terpilih = st.selectbox("Pilih produk", daftar_produk)

    if produk_terpilih:
        hasil = rekomendasi_produk(produk_terpilih, data)
        if hasil is None:
            st.warning("Produk tidak ditemukan pada data RFM.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Segmen RFM", hasil["segment_rfm"])
            col2.metric("Klaster K-Means", hasil["segment_cluster"])
            col3.metric("Recency (hari)", f"{hasil['recency']:.0f}")
            col4.metric("Frequency (kali)", f"{hasil['frequency']:.0f}")

            st.markdown('<hr class="header-garis">', unsafe_allow_html=True)
            tren_produk = (data["transaksi"][data["transaksi"]["ITEM"] == produk_terpilih]
                           .groupby("BULAN")["HARGA_JUAL"].sum().reset_index())
            if not tren_produk.empty:
                st.subheader(f"Tren Penjualan - {produk_terpilih}")
                fig = px.line(tren_produk, x="BULAN", y="HARGA_JUAL", markers=True, color_discrete_sequence=["#B91C1C"])
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Rekomendasi Bundling")
            if hasil["rules_terkait"].empty:
                st.info("Belum ditemukan pola pembelian bersama yang cukup kuat untuk produk ini.")
            else:
                st.dataframe(
                    hasil["rules_terkait"][["antecedents", "consequents", "support", "confidence", "lift"]],
                    use_container_width=True, hide_index=True
                )
                top_rule = hasil["rules_terkait"].iloc[0]
                st.success(
                    f"Rekomendasi utama: produk ini sering dibeli bersama {top_rule['consequents']} "
                    f"(confidence {top_rule['confidence']*100:.1f} persen, lift {top_rule['lift']:.2f})."
                )


# ════════════════════════════════════════════════════════════════
# BAGIAN 8 — HALAMAN: SIMULASI BUNDLING (dulunya 5_Simulasi_Bundling.py)
# ════════════════════════════════════════════════════════════════

def halaman_simulasi_bundling():
    render_brand_header()
    st.title("Simulasi What-If Bundling")
    st.caption("Pilih dua produk untuk melihat kekuatan asosiasinya. Sistem cek rules resmi dulu, kalau tidak ada baru dihitung langsung dari data transaksi.")

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
                st.write("")
                c1, c2, c3 = st.columns(3)
                c1.metric("Support", f"{hasil['support']*100:.2f}%")
                c2.metric("Confidence", f"{hasil['confidence']*100:.2f}%")
                c3.metric("Lift", f"{hasil['lift']:.2f}")
                if hasil["lift"] > 3:
                    st.caption("Nilai lift tergolong kuat, layak dipertimbangkan sebagai bundling.")

            elif hasil["status"] == "dihitung_langsung":
                st.markdown('<span class="badge-warn">Dihitung langsung dari data (di luar 210 rules resmi)</span>', unsafe_allow_html=True)
                st.write("")
                st.caption("Pasangan ini di bawah ambang batas support/confidence yang dipakai saat mining Apriori, tapi tetap pernah dibeli bersama.")
                c1, c2, c3 = st.columns(3)
                c1.metric("Support", f"{hasil['support']*100:.3f}%", help=f"{hasil['both']} dari {hasil['total_invoice']} invoice mengandung keduanya")
                c2.metric("Confidence (A ke B)", f"{hasil['conf_a_to_b']*100:.2f}%", help=f"Dari {hasil['count_a']} invoice yang beli {produk_a}, {hasil['both']} juga beli {produk_b}")
                c3.metric("Lift", f"{hasil['lift']:.2f}")
                if hasil["lift"] > 1.2:
                    st.caption("Lift lebih dari 1 artinya pasangan ini lebih sering dibeli bersama dibanding acak.")
                else:
                    st.caption("Lift kurang dari atau sama dengan 1 artinya kedua produk cenderung dibeli sendiri-sendiri.")

            else:
                st.markdown('<span class="badge-warn">Tidak pernah dibeli bersama</span>', unsafe_allow_html=True)
                st.write("")
                st.caption(f"{produk_a} dan {produk_b} tidak pernah muncul dalam invoice yang sama sepanjang tahun 2025.")


# ════════════════════════════════════════════════════════════════
# BAGIAN 9 — HALAMAN: PETA KORELASI (dulunya 7_Peta_Korelasi.py)
# ════════════════════════════════════════════════════════════════

def halaman_peta_korelasi():
    render_brand_header()
    st.title("Peta Korelasi Antar Produk")
    st.caption("Heatmap ini menunjukkan seberapa sering pasangan produk dibeli bersama dalam satu invoice.")

    data = load_data()
    top_n = st.slider("Jumlah produk terlaris yang ditampilkan", 5, 30, 15)

    corr, top_items = hitung_matriks_korelasi(data["transaksi"], top_n=top_n)

    fig = px.imshow(
        corr, x=top_items, y=top_items,
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        aspect="auto", labels=dict(color="Korelasi")
    )
    fig.update_layout(height=600, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pasangan dengan korelasi tertinggi")
    corr_clean = corr.copy()
    corr_clean.index.name = None
    corr_clean.columns.name = None
    pairs = corr_clean.where(~corr_clean.isna()).stack().reset_index()
    pairs.columns = ["Produk A", "Produk B", "Korelasi"]
    pairs = pairs[pairs["Produk A"] != pairs["Produk B"]]
    pairs["pasangan_unik"] = pairs.apply(lambda r: tuple(sorted([r["Produk A"], r["Produk B"]])), axis=1)
    pairs = pairs.drop_duplicates("pasangan_unik").drop(columns="pasangan_unik")
    pairs = pairs.sort_values("Korelasi", ascending=False).head(10)
    st.dataframe(pairs, use_container_width=True, hide_index=True)

    st.caption("Berbeda dengan Association Rules yang butuh threshold support/confidence, korelasi di sini murni statistik, cocok untuk eksplorasi awal sebelum melihat rules resmi Apriori.")


# ════════════════════════════════════════════════════════════════
# BAGIAN 10 — HALAMAN: UPLOAD DATA (dulunya 6_Upload_Data.py)
# ════════════════════════════════════════════════════════════════

def halaman_upload_data():
    render_brand_header()
    st.title("Upload Data Transaksi Baru")
    st.caption("Unggah berkas Excel transaksi terbaru untuk menjalankan ulang analisis RFM, K-Means, dan Apriori. Kolom wajib: NO_INVOICE, TANGGAL, NO_CUSTOMER, NAMA_CUSTOMER, ITEM, QTY, HARGA_JUAL, DISKON_PCT.")

    berkas = st.file_uploader("Pilih berkas Excel (.xlsx)", type=["xlsx"])

    if berkas is not None:
        try:
            df_upload = pd.read_excel(berkas)
            st.write("Pratinjau data:")
            st.dataframe(df_upload.head(10), use_container_width=True)

            if st.button("Jalankan Analisis"):
                with st.spinner("Menjalankan pipeline analisis..."):
                    hasil = jalankan_pipeline_baru(df_upload)

                if hasil.get("error"):
                    st.error(hasil['error'])
                else:
                    st.success("Analisis berhasil dijalankan.")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Baris Bersih", f"{hasil['jumlah_baris']:,}".replace(",", "."))
                    c2.metric("Produk Unik", hasil["jumlah_produk"])
                    c3.metric("Invoice Unik", f"{hasil['jumlah_invoice']:,}".replace(",", "."))
                    c4.metric("K Optimal", hasil["k_optimal"])

                    tab1, tab2 = st.tabs(["Hasil RFM dan Clustering", "Association Rules"])
                    with tab1:
                        st.dataframe(hasil["rfm"], use_container_width=True)
                        csv_rfm = hasil["rfm"].to_csv(index=False).encode("utf-8")
                        st.download_button("Unduh Hasil RFM (CSV)", csv_rfm, "hasil_rfm_baru.csv", "text/csv")
                    with tab2:
                        st.metric("Jumlah Rules", hasil["jumlah_rules"])
                        if not hasil["rules"].empty:
                            rd = hasil["rules"].copy()
                            rd["antecedents"] = rd["antecedents"].apply(lambda s: ", ".join(sorted(s)))
                            rd["consequents"] = rd["consequents"].apply(lambda s: ", ".join(sorted(s)))
                            st.dataframe(rd[["antecedents", "consequents", "support", "confidence", "lift"]], use_container_width=True)
        except Exception as e:
            st.error(f"Gagal membaca berkas: {e}")


# ════════════════════════════════════════════════════════════════
# BAGIAN 11 — ROUTER: tampilkan halaman sesuai pilihan sidebar
# ════════════════════════════════════════════════════════════════

ROUTER = {
    "Ringkasan": halaman_ringkasan,
    "Segmentasi": halaman_segmentasi,
    "Pola Beli": halaman_pola_beli,
    "Product Recommender": halaman_product_recommender,
    "Simulasi Bundling": halaman_simulasi_bundling,
    "Peta Korelasi": halaman_peta_korelasi,
    "Upload Data": halaman_upload_data,
}

ROUTER[halaman]()
