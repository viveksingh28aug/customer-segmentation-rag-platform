"""Segment overview page with PCA scatter and radar charts."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.path_setup import ensure_project_root

ensure_project_root()

from frontend.api_client import get_segment, get_segments

# SEGMENT_COLORS lives in charts; import locally to keep the UI module lightweight.
from frontend.charts import (
    SEGMENT_COLORS,
    build_pca_scatter,
    build_radar_chart,
    load_population_feature_means,
    load_segment_scatter_data,
)
from frontend.ui import apply_dashboard_style, page_header

st.set_page_config(page_title="Segment Overview", page_icon="📊", layout="wide")
apply_dashboard_style()
page_header(
    "Segment intelligence",
    "Segment Overview",
    "Understand the size, behavioral profile, and analytical confidence of each customer group.",
)

try:
    segments = get_segments()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

if not segments:
    st.warning("No segment profiles are available. Run `bash scripts/run_pipeline.sh` and retry.")
    st.stop()

segment_names = {int(s["id"]): str(s["name"]) for s in segments}
low_confidence_ids = {int(s["id"]) for s in segments if s.get("low_confidence")}
try:
    population_means = load_population_feature_means()
except (FileNotFoundError, KeyError, ValueError) as exc:
    st.error(f"Segment profile metrics could not be loaded: {exc}")
    st.stop()

total_households = sum(int(segment["size"]) for segment in segments)
summary_metrics = st.columns(3)
summary_metrics[0].metric("Segments", f"{len(segments):,}")
summary_metrics[1].metric("Households", f"{total_households:,}")
summary_metrics[2].metric("Low-confidence groups", f"{len(low_confidence_ids):,}")

col1, col2 = st.columns([2, 1])

with col1:
    try:
        scatter_df = load_segment_scatter_data()
    except FileNotFoundError:
        st.info("**PCA projection not generated**")
        st.write(
            "The bundled sample contains verified aggregate profiles but no household-level "
            "coordinates. The segment distribution and profile comparison remain available. "
            "To create PCA artifacts, place all required source CSVs in `DATA_DIR` and run "
            "`bash scripts/run_pipeline.sh`."
        )
    else:
        st.plotly_chart(build_pca_scatter(scatter_df, segment_names), width="stretch")

with col2:
    summary_df = pd.DataFrame(segments)[["id", "name", "size"]]
    summary_df.columns = ["Segment ID", "Name", "Households"]
    fig_bar = px.bar(
        summary_df,
        x="Households",
        y="Name",
        color="Name",
        orientation="h",
        title="Household distribution",
        text="Households",
        color_discrete_sequence=SEGMENT_COLORS,
    )
    fig_bar.update_layout(
        showlegend=False,
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 55, "b": 20},
        yaxis={"title": None, "categoryorder": "total ascending"},
    )
    fig_bar.update_traces(textposition="outside")
    st.plotly_chart(fig_bar, width="stretch")

if low_confidence_ids:
    st.warning(
        "Low-confidence segments (fewer than 30 households): "
        + ", ".join(f"Segment {sid}" for sid in sorted(low_confidence_ids))
    )

st.divider()
st.subheader("Behavioral profile")
selected_id = st.selectbox(
    "Select segment for radar comparison vs population",
    options=[int(s["id"]) for s in segments],
    format_func=lambda sid: f"{sid}: {segment_names[sid]}",
)

try:
    detail = get_segment(selected_id)
    if detail.get("low_confidence"):
        st.warning(
            f"Segment {selected_id} has only {detail['size']} households — "
            "interpret profiles and comparisons with caution."
        )
    metric_cols = st.columns(4)
    feature_means = detail["feature_means"]
    metric_cols[0].metric("Households", f'{detail["size"]:,}')
    metric_cols[1].metric("Spend / trip", f'${feature_means.get("monetary", 0):,.2f}')
    metric_cols[2].metric("Trip frequency", f'{feature_means.get("frequency", 0):.2f}')
    metric_cols[3].metric("Promo response", f'{feature_means.get("promo_responsiveness", 0):.1%}')
    radar_col, narrative_col = st.columns([2, 1])
    with radar_col:
        st.plotly_chart(
            build_radar_chart(
                segment_name=str(detail["name"]),
                segment_means=feature_means,
                population_means=population_means,
            ),
            width="stretch",
        )
    with narrative_col:
        st.markdown("#### Profile interpretation")
        st.write(detail["narrative"])
        st.caption(
            "Radar values are normalized against the population mean for visual comparison; "
            "they are not probabilities or model confidence scores."
        )
except RuntimeError as exc:
    st.error(str(exc))
