"""Validation tests for segmentation inputs and small datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shopper_segmentation.segmentation import evaluate_kmeans_k, load_feature_matrix


def test_load_feature_matrix_rejects_empty_dataset(tmp_path) -> None:
    """An empty parquet file should fail with an actionable message."""
    path = tmp_path / "empty.parquet"
    pd.DataFrame(columns=["household_key", "monetary"]).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="empty"):
        load_feature_matrix(path)


def test_load_feature_matrix_rejects_duplicate_households(tmp_path) -> None:
    """Clustering requires exactly one feature row per household."""
    path = tmp_path / "duplicates.parquet"
    pd.DataFrame({"household_key": [1, 1], "monetary": [2.0, 3.0]}).to_parquet(
        path, index=False
    )

    with pytest.raises(ValueError, match="duplicate household_key"):
        load_feature_matrix(path)


def test_load_feature_matrix_rejects_missing_values(tmp_path) -> None:
    """Missing model values should be resolved before scaling."""
    path = tmp_path / "missing.parquet"
    pd.DataFrame({"household_key": [1, 2], "monetary": [2.0, None]}).to_parquet(
        path, index=False
    )

    with pytest.raises(ValueError, match="missing values"):
        load_feature_matrix(path)


def test_evaluate_kmeans_rejects_insufficient_rows() -> None:
    """The configured minimum of four clusters requires at least five rows."""
    with pytest.raises(ValueError, match="At least 5 household rows"):
        evaluate_kmeans_k(np.ones((4, 2)))

