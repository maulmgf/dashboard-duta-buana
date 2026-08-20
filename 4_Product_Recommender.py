import streamlit as st
import plotly.express as px
from utils.pipeline import load_data, rekomendasi_produk
from utils.styling import apply_custom_theme, render_brand_header

st.set_page_config(page_title="Product Recommender", layout="wide")
apply_custom_theme()
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
