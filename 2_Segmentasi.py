import streamlit as st
import plotly.express as px
from utils.pipeline import load_data
from utils.styling import apply_custom_theme, render_brand_header

st.set_page_config(page_title="Segmentasi", layout="wide")
apply_custom_theme()
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
