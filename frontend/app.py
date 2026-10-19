"""Streamlit dashboard entry point for shopper segmentation."""

from __future__ import annotations

import os
import sys

import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from path_setup import ensure_project_root

ensure_project_root()

from frontend.api_client import check_health, get_api_base_url, get_status
from frontend.ui import apply_dashboard_style, data_mode_label, hero_header

st.set_page_config(
    page_title="Shopper Intelligence",
    page_icon="🛒",
    layout="wide",
)
apply_dashboard_style()

hero_header(
    "Shopper Segmentation & Personalization",
    "Explore behavioral customer groups, inspect evidence-based product affinities, "
    "and ask grounded questions about the available analytical outputs.",
)

api_url = get_api_base_url()
if not check_health():
    st.error(
        f"API not reachable at `{api_url}`. "
        "Start the backend: `uvicorn --app-dir src shopper_segmentation.api.app:app "
        "--host 127.0.0.1 --port 8000`"
    )
    st.stop()

try:
    status = get_status()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

st.markdown(
    f'<span class="status-pill">● API connected · {data_mode_label(status["data_mode"])}</span>',
    unsafe_allow_html=True,
)
st.write("")

metric_cols = st.columns(4)
metric_cols[0].metric("Customer segments", f'{status["segment_count"]:,}')
metric_cols[1].metric("Households", f'{status["household_count"]:,}')
metric_cols[2].metric("PCA projection", "Available" if status["pca_available"] else "Not generated")
metric_cols[3].metric("AI analyst", "Configured" if status["chat_configured"] else "Needs API key")

if status["data_mode"] == "sample":
    st.info(
        "You are viewing bundled sample artifacts. Segment metrics and recommendations are "
        "available; household-level PCA requires the licensed source CSVs and a pipeline run."
    )

st.subheader("Analytics workspace")
nav_cols = st.columns(3)
with nav_cols[0]:
    with st.container(border=True):
        st.markdown("### ◉ Segment Overview")
        st.write(
            "Compare segment size, behavioral profiles, confidence, and PCA position when available."
        )
        st.page_link("pages/1_Segment_Overview.py", label="Open segment analysis →")
with nav_cols[1]:
    with st.container(border=True):
        st.markdown("### ↗ Product Recommendations")
        st.write(
            "Review product affinity using segment and population purchase rates with lift evidence."
        )
        st.page_link("pages/2_Recommendations.py", label="Open recommendations →")
with nav_cols[2]:
    with st.container(border=True):
        st.markdown("### ✦ Analyst Chat")
        st.write("Ask grounded questions backed by retrieved segment cards and numeric validation.")
        st.page_link("pages/3_Analyst_Chat.py", label="Open analyst chat →")

st.divider()
st.caption(
    "Pipeline status reflects the artifacts currently served by the local API. "
    "No model is retrained during dashboard use."
)
