import streamlit as st
import plotly.express as px
from utils.pipeline import load_data
from utils.styling import apply_custom_theme, render_brand_header, PALET_GRADASI

st.set_page_config(page_title="Ringkasan", layout="wide")
apply_custom_theme()
render_brand_header()

st.title("Ringkasan Penjualan")
data = load_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"Rp {data['kpi']['revenue']/1e9:.2f} M")
col2.metric("Total Invoice", f"{data['kpi']['invoice']:,}".replace(",", "."))
col3.metric("Produk Aktif", data['kpi']['produk'])
col4.metric("Total Customer", data['kpi']['customer'])

st.markdown('<hr class="header-garis">', unsafe_allow_html=True)
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
    top5 = data["transaksi"].groupby("ITEM")["HARGA_JUAL"].sum().nlargest(5).reset_index()
    fig_bar = px.bar(top5, x="HARGA_JUAL", y="ITEM", orientation="h", color_discrete_sequence=["#B91C1C"])
    fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right2:
    st.subheader("Profil Klaster")
    st.dataframe(data["cluster_profile"][["Segment_Cluster", "Jumlah_Produk", "Monetary"]], use_container_width=True, hide_index=True)
