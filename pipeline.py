import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules

DATA_PATH = "/content/Dashboard_streamlit.xlsx"
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
