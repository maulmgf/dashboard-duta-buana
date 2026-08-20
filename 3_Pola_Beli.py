import streamlit as st
import plotly.express as px
from utils.pipeline import load_data
from utils.styling import apply_custom_theme, render_brand_header

st.set_page_config(page_title="Pola Beli", layout="wide")
apply_custom_theme()
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
