"""Unit tests for vector store retrieval."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from shopper_segmentation.rag import vectorstore


def test_score_to_distance_inverts_normalized_score() -> None:
    """Normalized similarity scores should map to lower-is-closer distances."""
    assert vectorstore._score_to_distance(0.9) == pytest.approx(0.1)
    assert vectorstore._score_to_distance(1.0) < vectorstore._score_to_distance(0.5)


def test_retrieve_cards_returns_expected_shape(monkeypatch) -> None:
    """retrieve_cards should preserve the legacy response contract."""
    mock_store = MagicMock()
    mock_store.similarity_search_with_score.return_value = [
        (
            Document(
                page_content="Segment 2 card",
                metadata={"segment_id": 2, "segment_name": "Premium Grocery Loyalists"},
            ),
            0.8,
        )
    ]

    monkeypatch.setenv("RETRIEVAL_BACKEND", "chroma")
    with patch("shopper_segmentation.rag.vectorstore.get_vectorstore", return_value=mock_store):
        cards = vectorstore.retrieve_cards("promo-sensitive shoppers", top_k=1)

    assert len(cards) == 1
    assert set(cards[0].keys()) == {"segment_id", "segment_name", "content", "distance"}
    assert cards[0]["segment_id"] == 2
    assert cards[0]["segment_name"] == "Premium Grocery Loyalists"
    assert cards[0]["content"] == "Segment 2 card"
    assert cards[0]["distance"] == pytest.approx(0.2)


def test_tfidf_retrieval_is_local_and_ranked(monkeypatch) -> None:
    """Default retrieval should work without loading a transformer model."""
    monkeypatch.setenv("RETRIEVAL_BACKEND", "tfidf")
    cards = [
        {"segment_id": 1, "segment_name": "Promo", "content": "coupon promotion response"},
        {"segment_id": 2, "segment_name": "Premium", "content": "premium high value basket"},
    ]
    with patch("shopper_segmentation.rag.vectorstore.build_all_cards", return_value=cards):
        results = vectorstore.retrieve_cards("high value premium", top_k=1)

    assert results[0]["segment_id"] == 2
    assert 0 <= results[0]["distance"] <= 1


def test_get_vectorstore_rebuilds_when_collection_empty() -> None:
    """An empty persisted collection should trigger a rebuild."""
    mock_store = MagicMock()
    mock_store._collection.count.return_value = 0

    with patch("shopper_segmentation.rag.vectorstore.Chroma", return_value=mock_store):
        with patch("shopper_segmentation.rag.vectorstore._get_embeddings", return_value=MagicMock()):
            with patch(
                "shopper_segmentation.rag.vectorstore.build_vectorstore",
                return_value=MagicMock(),
            ) as build_mock:
                vectorstore.get_vectorstore()

    build_mock.assert_called_once()
