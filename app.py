"""
Dashboard Analisis Penjualan PT Duta Buana Perkasa
====================================================
Aplikasi web interaktif berbasis Streamlit untuk menyajikan hasil
analisis RFM, K-Means Clustering, dan Apriori Association Rules.
"""

import streamlit as st
from utils.styling import apply_custom_theme, render_brand_header, render_hero_banner, kartu_insight
from utils.pipeline import load_data

st.set_page_config(
    page_title="Dashboard PT Duta Buana Perkasa",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_theme()
render_brand_header()
render_hero_banner()

data = load_data()
rfm = data["rfm"]
transaksi = data["transaksi"]

# ── KPI UTAMA ────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"Rp {data['kpi']['revenue']/1e9:.2f} M")
col2.metric("Total Invoice", f"{data['kpi']['invoice']:,}".replace(",", "."))
col3.metric("Produk Aktif", data['kpi']['produk'])
col4.metric("Total Customer", data['kpi']['customer'])

st.markdown('<hr class="header-garis">', unsafe_allow_html=True)

# ── INSIGHT OTOMATIS ─────────────────────────────────
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

# ── GRID MENU HALAMAN ────────────────────────────────
st.markdown("### Menu yang tersedia")

menu_info = [
    ("Ringkasan", "KPI utama, tren bulanan, top produk dan customer"),
    ("Segmentasi", "Sebaran produk per klaster RFM dan K-Means"),
    ("Pola Beli", "Seluruh association rules hasil Apriori"),
    ("Product Recommender", "Profil lengkap satu produk dan rekomendasi bundling"),
    ("Simulasi Bundling", "Cek kekuatan asosiasi dua produk mana pun"),
    ("Peta Korelasi", "Heatmap co-purchase antar produk terlaris"),
    ("Kalkulator Keranjang", "Simulasi keranjang belanja multi produk"),
    ("Upload Data", "Jalankan ulang analisis dengan data baru"),
]

menu_cols = st.columns(4)
for i, (judul, desk) in enumerate(menu_info):
    with menu_cols[i % 4]:
        st.markdown(
            f'<div class="menu-item"><div class="judul">{judul}</div><div class="desk">{desk}</div></div>',
            unsafe_allow_html=True
        )

st.caption("Gunakan menu di sidebar kiri untuk membuka masing-masing halaman.")
