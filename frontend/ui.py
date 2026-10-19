"""Shared presentation helpers for the Streamlit analytics dashboard."""

from __future__ import annotations

import streamlit as st

ACCENT = "#2563EB"
TEAL = "#0F766E"
AMBER = "#B45309"
SLATE = "#0F172A"


def apply_dashboard_style() -> None:
    """Apply a small, accessible visual system using only native Streamlit CSS."""
    st.markdown(
        """
        <style>
        .stApp { background: #F8FAFC; color: #0F172A; }
        [data-testid="stHeader"] { background: rgba(248,250,252,.92); }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #FFFFFF 0%, #F1F5F9 100%);
            border-right: 1px solid #E2E8F0;
        }
        .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
        h1, h2, h3 { color: #0F172A; letter-spacing: -0.02em; }
        p, label, [data-testid="stCaptionContainer"] { color: #475569; }
        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1rem 1.1rem;
            box-shadow: 0 1px 2px rgba(15,23,42,.04);
            transition: transform .16s ease, box-shadow .16s ease;
        }
        [data-testid="stMetric"]:hover { transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(15,23,42,.08); }
        [data-testid="stMetricLabel"] { color: #64748B; }
        [data-testid="stMetricValue"] { color: #0F172A; }
        .status-pill {
            display: inline-block; padding: .3rem .65rem; border-radius: 999px;
            background: #DBEAFE; color: #1E40AF; font-size: .82rem; font-weight: 650;
        }
        .eyebrow { color: #2563EB; font-size: .78rem; font-weight: 750;
            letter-spacing: .09em; text-transform: uppercase; margin-bottom: .25rem; }
        .page-subtitle { color: #475569; max-width: 820px; margin-top: -.45rem; }
        .hero-panel {
            position: relative; overflow: hidden; border-radius: 20px;
            padding: 2.5rem 2.75rem; margin-bottom: 1.3rem;
            color: #FFFFFF;
            background: linear-gradient(125deg, #071A3D 0%, #174A8B 58%, #0F766E 100%);
            box-shadow: 0 16px 42px rgba(15,23,42,.16);
        }
        .hero-panel:after { content: ""; position: absolute; width: 320px; height: 320px;
            right: -90px; top: -145px; border-radius: 50%;
            background: radial-gradient(circle, rgba(147,197,253,.48), transparent 67%); }
        .hero-kicker { color: #BFDBFE; font-size: .75rem; font-weight: 800;
            letter-spacing: .12em; text-transform: uppercase; }
        .hero-panel h1 { color: #FFFFFF; margin: .55rem 0 .55rem; font-size: 2.7rem; }
        .hero-panel p { color: #DBEAFE; max-width: 780px; font-size: 1.05rem; margin: 0; }
        .hero-tags { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1.35rem; }
        .hero-tag { border: 1px solid rgba(255,255,255,.22); border-radius: 999px;
            background: rgba(255,255,255,.1); padding: .35rem .7rem;
            color: #F8FAFC; font-size: .78rem; font-weight: 650; }
        div[data-testid="stDataFrame"] { border: 1px solid #E2E8F0; border-radius: 10px; }
        .stButton > button { border-radius: 8px; font-weight: 650; }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #FFFFFF; border-color: #E2E8F0; box-shadow: 0 4px 14px rgba(15,23,42,.045);
        }
        @media (max-width: 700px) {
            .block-container { padding: 1.25rem .8rem 2rem; }
            h1 { font-size: 1.85rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    """Render a consistent page heading and supporting description."""
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="page-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def hero_header(title: str, subtitle: str) -> None:
    """Render the dashboard's branded landing hero."""
    st.markdown(
        f"""
        <section class="hero-panel">
          <div class="hero-kicker">Customer analytics platform</div>
          <h1>{title}</h1>
          <p>{subtitle}</p>
          <div class="hero-tags">
            <span class="hero-tag">Customer segments</span>
            <span class="hero-tag">Product affinity</span>
            <span class="hero-tag">Campaign evidence</span>
            <span class="hero-tag">Grounded AI analyst</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def data_mode_label(data_mode: str) -> str:
    """Return a concise, human-readable data mode label."""
    return "Bundled sample data" if data_mode == "sample" else "Pipeline output"

