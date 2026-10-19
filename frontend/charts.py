"""Shared chart helpers for the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from shopper_segmentation.artifacts import load_profiles
from shopper_segmentation.etl import OUTPUT_DIR

RADAR_FEATURES = [
    "monetary",
    "frequency",
    "recency",
    "price_sensitivity",
    "promo_responsiveness",
    "avg_basket_size",
    "display_exposure_rate",
    "mailer_exposure_rate",
]

FEATURE_LABELS = {
    "monetary": "Spend / trip",
    "frequency": "Trip frequency",
    "recency": "Recency",
    "price_sensitivity": "Price sensitivity",
    "promo_responsiveness": "Promo response",
    "avg_basket_size": "Basket size",
    "display_exposure_rate": "Display exposure",
    "mailer_exposure_rate": "Mailer exposure",
}

SEGMENT_COLORS = [
    "#2563EB",
    "#0F766E",
    "#7C3AED",
    "#C2410C",
    "#0891B2",
    "#BE123C",
    "#4D7C0F",
    "#475569",
]

SEGMENTS_PATH = OUTPUT_DIR / "household_segments.parquet"
FEATURES_PATH = OUTPUT_DIR / "household_features.parquet"


def load_segment_scatter_data() -> pd.DataFrame:
    """Load PCA coordinates joined with segment assignments.

    Returns:
        DataFrame with pca_1, pca_2, segment_id, household_key.
    """
    segments = pd.read_parquet(SEGMENTS_PATH)
    return segments


def load_population_feature_means() -> pd.Series:
    """Compute population-average feature means for radar normalization.

    Returns:
        Series of population means for radar features.
    """
    if FEATURES_PATH.exists():
        features = pd.read_parquet(FEATURES_PATH)
        return features[RADAR_FEATURES].mean()

    # Bundled sample mode contains profile JSON but intentionally omits the much
    # larger household-level parquet files. Reconstruct the population mean as a
    # segment-size-weighted average so the radar chart remains useful.
    segments = load_profiles()["segments"]
    total_size = sum(int(segment["size"]) for segment in segments)
    if total_size <= 0:
        raise ValueError("Segment profiles contain no households")

    return pd.Series(
        {
            feature: sum(
                float(segment["feature_means"].get(feature, 0.0))
                * int(segment["size"])
                for segment in segments
            )
            / total_size
            for feature in RADAR_FEATURES
        }
    )


def normalize_radar_values(
    segment_means: dict[str, float],
    population_means: pd.Series,
) -> tuple[list[str], list[float], list[float]]:
    """Normalize segment and population values to 0-1 for radar charts.

    Args:
        segment_means: Segment-level feature averages.
        population_means: Population feature averages.

    Returns:
        Tuple of feature labels, segment normalized values, population normalized values.
    """
    labels = [FEATURE_LABELS[feature] for feature in RADAR_FEATURES]
    segment_vals: list[float] = []
    population_vals: list[float] = []

    for feature in RADAR_FEATURES:
        seg_val = float(segment_means.get(feature, 0.0))
        pop_val = float(population_means.get(feature, 0.0))
        max_val = max(seg_val, pop_val, 1e-9)
        segment_vals.append(seg_val / max_val)
        population_vals.append(pop_val / max_val)

    return labels, segment_vals, population_vals


def build_pca_scatter(df: pd.DataFrame, segment_names: dict[int, str]) -> go.Figure:
    """Build a PCA scatter plot colored by segment.

    Args:
        df: Household segment dataframe with PCA columns.
        segment_names: Mapping of segment id to display name.

    Returns:
        Plotly figure.
    """
    plot_df = df.copy()
    plot_df["segment_label"] = plot_df["segment_id"].map(
        lambda sid: f"{sid}: {segment_names.get(int(sid), 'Unknown')}"
    )
    fig = px.scatter(
        plot_df,
        x="pca_1",
        y="pca_2",
        color="segment_label",
        opacity=0.65,
        title="Household Segments — PCA Projection",
        labels={"pca_1": "PCA 1", "pca_2": "PCA 2", "segment_label": "Segment"},
        hover_data=["household_key"],
        color_discrete_sequence=SEGMENT_COLORS,
    )
    fig.update_traces(marker={"size": 7, "line": {"width": 0}})
    fig.update_layout(
        legend_title_text="Segment",
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        hovermode="closest",
    )
    return fig


def build_radar_chart(
    segment_name: str,
    segment_means: dict[str, float],
    population_means: pd.Series,
) -> go.Figure:
    """Build a radar chart comparing segment vs population feature means.

    Args:
        segment_name: Display name for the segment.
        segment_means: Segment feature averages.
        population_means: Population feature averages.

    Returns:
        Plotly radar figure.
    """
    labels, segment_vals, population_vals = normalize_radar_values(
        segment_means, population_means
    )
    labels_closed = labels + [labels[0]]
    segment_closed = segment_vals + [segment_vals[0]]
    population_closed = population_vals + [population_vals[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=population_closed,
            theta=labels_closed,
            fill="toself",
            name="Population",
            opacity=0.45,
            line={"color": "#94A3B8"},
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=segment_closed,
            theta=labels_closed,
            fill="toself",
            name=segment_name,
            opacity=0.65,
            line={"color": "#2563EB", "width": 3},
        )
    )
    fig.update_layout(
        title=f"Segment Profile — {segment_name}",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1.05])),
        showlegend=True,
        template="plotly_white",
        margin={"l": 55, "r": 55, "t": 70, "b": 45},
    )
    return fig
