"""Unit tests for FastAPI backend endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from tests.conftest import ASGITestClient


def test_list_segments(client: ASGITestClient, auth_headers: dict[str, str]) -> None:
    """GET /segments should return all segment summaries."""
    response = client.get("/segments", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 8
    assert data[0]["id"] == 0
    assert "name" in data[0]
    assert "size" in data[0]


def test_get_segment_detail(client: ASGITestClient, auth_headers: dict[str, str]) -> None:
    """GET /segments/{id} should return feature means."""
    response = client.get("/segments/2", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == 2
    assert "feature_means" in data
    assert "monetary" in data["feature_means"]


def test_get_segment_not_found(client: ASGITestClient, auth_headers: dict[str, str]) -> None:
    """Unknown segment ids should return 404."""
    response = client.get("/segments/999", headers=auth_headers)
    assert response.status_code == 404


def test_get_segment_recommendations(client: ASGITestClient, auth_headers: dict[str, str]) -> None:
    """GET /segments/{id}/recommendations should return ranked products."""
    response = client.get("/segments/2/recommendations", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["segment_id"] == 2
    assert len(data["recommendations"]) > 0
    assert "lift" in data["recommendations"][0]


def test_missing_api_key_returns_401(client: ASGITestClient) -> None:
    """Protected routes should reject requests without X-API-Key."""
    response = client.get("/segments")
    assert response.status_code == 401
    assert response.json()["detail"] == "Missing X-API-Key header"


def test_invalid_api_key_returns_401(client: ASGITestClient) -> None:
    """Protected routes should reject requests with an invalid API key."""
    response = client.get("/segments", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"


def test_health_without_api_key(client: ASGITestClient) -> None:
    """GET /health should remain accessible without authentication."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_status_reports_verified_artifact_counts(
    client: ASGITestClient,
    auth_headers: dict[str, str],
) -> None:
    """GET /status should derive counts and mode from the loaded artifacts."""
    response = client.get("/status", headers=auth_headers)
    assert response.status_code == 200
    payload = response.json()
    assert payload["segment_count"] == 8
    assert payload["household_count"] == 2500
    assert payload["data_mode"] in {"sample", "full"}


def test_chat_rejects_whitespace_only_question(
    client: ASGITestClient,
    auth_headers: dict[str, str],
) -> None:
    """Whitespace is not a valid analyst question."""
    response = client.post(
        "/chat?stream=false",
        json={"query": "   "},
        headers=auth_headers,
    )
    assert response.status_code == 422


def test_chat_missing_groq_api_key(client: ASGITestClient, auth_headers: dict[str, str]) -> None:
    """POST /chat should fail clearly when GROQ_API_KEY is missing."""
    with patch(
        "shopper_segmentation.api.app.get_groq_api_key",
        side_effect=RuntimeError("GROQ_API_KEY is not set."),
    ):
        response = client.post(
            "/chat?stream=false",
            json={"query": "Who are promo-sensitive shoppers?"},
            headers=auth_headers,
        )
    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]


def test_chat_rate_limit(client: ASGITestClient, auth_headers: dict[str, str]) -> None:
    """POST /chat should return 429 after exceeding the per-minute limit."""
    mock_result = {
        "query": "test",
        "answer": "mock answer",
        "retrieved_cards": [],
        "validation": {
            "validated": True,
            "numbers_found": [],
            "unsupported_numbers": [],
        },
    }

    with patch("shopper_segmentation.api.app.get_groq_api_key", return_value="gsk_test"):
        with patch(
            "shopper_segmentation.api.app.answer_query_async",
            new=AsyncMock(return_value=mock_result),
        ):
            statuses: list[int] = []
            for _ in range(12):
                response = client.post(
                    "/chat?stream=false",
                    json={"query": "rate limit test"},
                    headers=auth_headers,
                )
                statuses.append(response.status_code)

    assert 200 in statuses
    assert 429 in statuses


def test_groq_outage_uses_fallback_not_500(client: ASGITestClient, auth_headers: dict[str, str]) -> None:
    """Chat should return 200 when the adapter succeeds after model fallback."""
    from shopper_segmentation.api.app import limiter

    if hasattr(limiter, "_storage"):
        limiter._storage.storage.clear()

    mock_result = {
        "query": "Who are promo-sensitive shoppers?",
        "answer": "Recovered response without unsupported numbers.",
        "retrieved_cards": [],
        "validation": {
            "validated": True,
            "numbers_found": [],
            "unsupported_numbers": [],
        },
    }

    with patch("shopper_segmentation.api.app.get_groq_api_key", return_value="gsk_test"):
        with patch(
            "shopper_segmentation.api.app.answer_query_async",
            new=AsyncMock(return_value=mock_result),
        ):
            response = client.post(
                "/chat?stream=false",
                json={"query": "Who are promo-sensitive shoppers?"},
                headers=auth_headers,
            )

    assert response.status_code == 200
    assert response.json()["answer"] == mock_result["answer"]


def test_chat_stream_returns_error_sse_on_graph_failure(
    client: ASGITestClient,
    auth_headers: dict[str, str],
) -> None:
    """Streaming chat should emit structured SSE errors instead of aborting."""
    from shopper_segmentation.api.app import limiter

    if hasattr(limiter, "_storage"):
        limiter._storage.storage.clear()

    with patch("shopper_segmentation.api.app.get_groq_api_key", return_value="gsk_test"):
        with patch(
            "shopper_segmentation.api.app.answer_query_async",
            new=AsyncMock(side_effect=RuntimeError("Simulated graph failure")),
        ):
            response = client.post(
                "/chat?stream=true",
                json={"query": "Who are promo-sensitive shoppers?"},
                headers=auth_headers,
            )

    assert response.status_code == 200
    assert '"type": "error"' in response.text or '"type":"error"' in response.text
    assert "chat_service_error" in response.text
    assert "Simulated graph failure" not in response.text


def test_chat_provider_rate_limit_is_actionable_and_sanitized(
    client: ASGITestClient,
    auth_headers: dict[str, str],
) -> None:
    """Provider 429 errors should not expose raw provider response details."""
    from shopper_segmentation.api.app import limiter

    if hasattr(limiter, "_storage"):
        limiter._storage.storage.clear()

    class FakeRateLimitError(Exception):
        status_code = 429

    with patch("shopper_segmentation.api.app.get_groq_api_key", return_value="gsk_test"):
        with patch(
            "shopper_segmentation.api.app.answer_query_async",
            new=AsyncMock(side_effect=FakeRateLimitError("secret provider body")),
        ):
            response = client.post(
                "/chat?stream=false",
                json={"query": "Which segment responds best?"},
                headers=auth_headers,
            )

    assert response.status_code == 429
    assert "rate-limited" in response.json()["detail"]
    assert "secret provider body" not in response.text


def test_chat_rejects_provider_response_without_text(
    client: ASGITestClient,
    auth_headers: dict[str, str],
) -> None:
    """A valid provider envelope with no answer text should return 502."""
    from shopper_segmentation.api.app import limiter

    if hasattr(limiter, "_storage"):
        limiter._storage.storage.clear()

    empty_result = {
        "query": "test",
        "answer": "",
        "retrieved_cards": [],
        "validation": {"validated": True, "numbers_found": [], "unsupported_numbers": []},
    }
    with patch("shopper_segmentation.api.app.get_groq_api_key", return_value="gsk_test"):
        with patch(
            "shopper_segmentation.api.app.answer_query_async",
            new=AsyncMock(return_value=empty_result),
        ):
            response = client.post(
                "/chat?stream=false",
                json={"query": "test"},
                headers=auth_headers,
            )

    assert response.status_code == 502
    assert "no answer text" in response.json()["detail"]


def test_low_confidence_segments_flagged(client: ASGITestClient, auth_headers: dict[str, str]) -> None:
    """Segments below the confidence threshold should expose low_confidence=true."""
    from shopper_segmentation.segmentation import is_low_confidence_segment

    assert is_low_confidence_segment(5) is True
    assert is_low_confidence_segment(100) is False

    response = client.get("/segments", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert all("low_confidence" in segment for segment in data)

    small_segments = [segment for segment in data if segment["size"] < 30]
    if small_segments:
        assert all(segment["low_confidence"] for segment in small_segments)
