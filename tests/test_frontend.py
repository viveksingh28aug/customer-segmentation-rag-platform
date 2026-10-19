"""Tests for dashboard bootstrapping and sample-data chart support."""

from __future__ import annotations

import sys

from frontend import path_setup


def test_path_setup_adds_src_directory(monkeypatch) -> None:
    """Streamlit can import the src-layout package without an editable install."""
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != str(path_setup.SRC_DIR)])

    path_setup.ensure_project_root()

    assert str(path_setup.SRC_DIR) in sys.path


def test_population_means_fall_back_to_segment_profiles(monkeypatch, tmp_path) -> None:
    """The overview radar remains available when sample parquet files are absent."""
    from frontend import charts

    monkeypatch.setattr(charts, "FEATURES_PATH", tmp_path / "missing.parquet")
    monkeypatch.setattr(
        charts,
        "load_profiles",
        lambda: {
            "segments": [
                {"size": 1, "feature_means": {"monetary": 2.0}},
                {"size": 3, "feature_means": {"monetary": 6.0}},
            ]
        },
    )

    means = charts.load_population_feature_means()

    assert means["monetary"] == 5.0
