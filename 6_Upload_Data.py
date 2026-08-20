import streamlit as st
import pandas as pd
from utils.pipeline import jalankan_pipeline_baru
from utils.styling import apply_custom_theme, render_brand_header

st.set_page_config(page_title="Upload Data Baru", layout="wide")
apply_custom_theme()
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
