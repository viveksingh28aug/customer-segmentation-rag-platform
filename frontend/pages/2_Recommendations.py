"""Product recommendations page by segment."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.path_setup import ensure_project_root

ensure_project_root()

from frontend.api_client import get_recommendations, get_segments
from frontend.ui import apply_dashboard_style, page_header

st.set_page_config(page_title="Recommendations", page_icon="🎯", layout="wide")
apply_dashboard_style()
page_header(
    "Product affinity",
    "Segment Product Recommendations",
    "Prioritize products using observed buyer penetration and lift relative to the full customer base.",
)

try:
    segments = get_segments()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

if not segments:
    st.warning("No segments are available. Run the data pipeline and retry.")
    st.stop()

segment_names = {int(s["id"]): str(s["name"]) for s in segments}
segment_meta = {int(s["id"]): s for s in segments}
selected_id = st.selectbox(
    "Select segment",
    options=[int(s["id"]) for s in segments],
    format_func=lambda sid: f"{sid}: {segment_names[sid]}",
)

selected_segment = segment_meta[selected_id]
headline_cols = st.columns(3)
headline_cols[0].metric("Selected segment", str(selected_id))
headline_cols[1].metric("Households", f'{selected_segment["size"]:,}')
headline_cols[2].metric(
    "Confidence",
    "Low" if selected_segment.get("low_confidence") else "Standard",
)
if selected_segment.get("low_confidence"):
    st.warning(
        f"Segment {selected_id} has only {selected_segment['size']} households — "
        "product lift estimates may be unstable."
    )

try:
    payload = get_recommendations(selected_id)
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

recs = payload.get("recommendations", [])
if not recs:
    st.warning(f"No recommendations available for segment {selected_id}.")
    st.stop()

df = pd.DataFrame(recs)
filter_cols = st.columns([1, 1, 2])
sort_label = filter_cols[0].selectbox(
    "Sort recommendations",
    options=["Lift", "Segment purchase rate", "Buyer count"],
)
minimum_buyers = filter_cols[1].number_input(
    "Minimum segment buyers",
    min_value=0,
    max_value=int(df["segment_buyers"].max()),
    value=0,
    step=1,
)
sort_map = {
    "Lift": "lift",
    "Segment purchase rate": "segment_purchase_rate",
    "Buyer count": "segment_buyers",
}
df = df[df["segment_buyers"] >= minimum_buyers].sort_values(
    sort_map[sort_label], ascending=False
)
if df.empty:
    st.info("No recommendations match the current buyer-count filter.")
    st.stop()

display_cols = [
    "product_id",
    "department",
    "brand",
    "commodity_desc",
    "lift",
    "segment_purchase_rate",
    "population_purchase_rate",
    "segment_buyers",
]
df = df[display_cols]
df.columns = [
    "Product ID",
    "Department",
    "Brand",
    "Commodity",
    "Lift",
    "Segment Rate",
    "Population Rate",
    "Segment Buyers",
]

st.subheader(f"Top products for {payload['segment_name']}")
st.dataframe(
    df,
    width="stretch",
    hide_index=True,
    column_config={
        "Product ID": st.column_config.NumberColumn(format="%d"),
        "Lift": st.column_config.NumberColumn(format="%.2fx"),
        "Segment Rate": st.column_config.NumberColumn(format="percent"),
        "Population Rate": st.column_config.NumberColumn(format="percent"),
        "Segment Buyers": st.column_config.NumberColumn(format="%d"),
    },
)

st.caption(
    "Lift = segment purchase rate ÷ population purchase rate. A lift of 2.00× means "
    "the product is observed among segment households at twice the population rate. "
    "Lift describes association, not causation; pipeline support thresholds still apply."
)
