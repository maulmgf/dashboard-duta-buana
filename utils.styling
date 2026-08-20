%%writefile utils/styling.py
"""Modul styling — tema visual brand PT Duta Buana Perkasa (merah-putih)
untuk seluruh komponen Streamlit."""

import streamlit as st
import base64
import os

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
    dipakai khusus di halaman utama (app.py)."""
    banner_path = "data/banner.jpg"
    if not os.path.exists(banner_path):
        st.warning("Berkas data/banner.jpg tidak ditemukan — hero banner dilewati.")
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
