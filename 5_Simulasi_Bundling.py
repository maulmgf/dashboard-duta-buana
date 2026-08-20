import streamlit as st
from utils.pipeline import load_data, simulasi_bundling
from utils.styling import apply_custom_theme, render_brand_header

st.set_page_config(page_title="Simulasi Bundling", layout="wide")
apply_custom_theme()
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
            st.caption(f"Pasangan ini di bawah ambang batas support/confidence yang dipakai saat mining Apriori, tapi tetap pernah dibeli bersama.")
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
