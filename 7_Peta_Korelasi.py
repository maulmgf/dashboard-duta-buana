import streamlit as st
import plotly.express as px
from utils.pipeline import load_data, hitung_matriks_korelasi
from utils.styling import apply_custom_theme, render_brand_header

st.set_page_config(page_title="Peta Korelasi Produk", layout="wide")
apply_custom_theme()
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
