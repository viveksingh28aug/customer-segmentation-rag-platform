"""Unit tests for resilient chat streaming client behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from frontend.api_client import (
    STREAM_MAX_RETRIES,
    ChatStreamError,
    _consume_chat_sse,
    _post_chat_stream_once,
    post_chat_stream,
)


def _mock_response(lines: list[str], status_code: int = 200) -> httpx.Response:
    """Build a mock httpx response whose iter_lines yields given SSE lines."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.iter_lines.return_value = iter(lines)
    response.raise_for_status = MagicMock()
    return response


def test_consume_chat_sse_returns_done_event() -> None:
    """Should return the terminal done payload from SSE lines."""
    response = _mock_response(
        [
            'data: {"type": "thread_id", "thread_id": "abc"}',
            'data: {"type": "token", "content": "Hello"}',
            'data: {"type": "done", "answer": "Hello", "thread_id": "abc"}',
        ]
    )

    tokens: list[str] = []
    result = _consume_chat_sse(response, on_token=tokens.append)

    assert result["answer"] == "Hello"
    assert result["thread_id"] == "abc"
    assert tokens == ["Hello"]


def test_consume_chat_sse_raises_on_backend_error_event() -> None:
    """Backend SSE error events should map to ChatStreamError."""
    response = _mock_response(
        [
            'data: {"type": "error", "error": {"code": "stream_failed", "message": "boom"}}',
        ]
    )

    with pytest.raises(ChatStreamError) as exc_info:
        _consume_chat_sse(response)

    assert exc_info.value.code == "stream_failed"
    assert exc_info.value.message == "boom"


def test_consume_chat_sse_raises_on_incomplete_stream() -> None:
    """Missing done event should raise stream_incomplete."""
    response = _mock_response(
        [
            'data: {"type": "thread_id", "thread_id": "abc"}',
        ]
    )

    with pytest.raises(ChatStreamError) as exc_info:
        _consume_chat_sse(response)

    assert exc_info.value.code == "stream_incomplete"


def test_consume_chat_sse_rejects_malformed_event() -> None:
    """Malformed provider/API stream events should return a parse error."""
    response = _mock_response(["data: {not-json}"])

    with pytest.raises(ChatStreamError) as exc_info:
        _consume_chat_sse(response)

    assert exc_info.value.code == "parse_error"


def test_consume_chat_sse_rejects_empty_answer() -> None:
    """A completed stream without answer text is not a successful response."""
    response = _mock_response(['data: {"type": "done", "answer": "  "}'])

    with pytest.raises(ChatStreamError) as exc_info:
        _consume_chat_sse(response)

    assert exc_info.value.code == "empty_response"


def test_consume_chat_sse_maps_remote_protocol_error() -> None:
    """RemoteProtocolError during iteration should map to connection_lost."""
    response = _mock_response([])
    response.iter_lines.side_effect = httpx.RemoteProtocolError(
        "peer closed connection without sending complete message body"
    )

    with pytest.raises(ChatStreamError) as exc_info:
        _consume_chat_sse(response)

    assert exc_info.value.code == "connection_lost"


def test_post_chat_stream_retries_transient_connection_lost() -> None:
    """Transient connection_lost errors should retry up to two times."""
    side_effects = [
        ChatStreamError("connection_lost", "lost"),
        ChatStreamError("connection_lost", "lost"),
        {"type": "done", "answer": "ok", "thread_id": "t1"},
    ]

    with patch(
        "frontend.api_client._post_chat_stream_once",
        side_effect=side_effects,
    ) as mock_once:
        with patch("frontend.api_client.time.sleep"):
            result = post_chat_stream("hello")

    assert result["answer"] == "ok"
    assert mock_once.call_count == 3


def test_post_chat_stream_does_not_retry_backend_error() -> None:
    """Explicit backend errors should not be retried."""
    with patch(
        "frontend.api_client._post_chat_stream_once",
        side_effect=ChatStreamError("stream_failed", "boom"),
    ) as mock_once:
        with pytest.raises(ChatStreamError) as exc_info:
            post_chat_stream("hello")

    assert exc_info.value.code == "stream_failed"
    assert mock_once.call_count == 1


def test_post_chat_stream_stops_after_max_retries() -> None:
    """Should fail after exhausting all retry attempts."""
    with patch(
        "frontend.api_client._post_chat_stream_once",
        side_effect=ChatStreamError("connection_lost", "lost"),
    ) as mock_once:
        with patch("frontend.api_client.time.sleep"):
            with pytest.raises(ChatStreamError) as exc_info:
                post_chat_stream("hello")

    assert exc_info.value.code == "connection_lost"
    assert mock_once.call_count == STREAM_MAX_RETRIES + 1


@pytest.mark.parametrize(
    ("status_code", "expected_code"),
    [(401, "authentication_error"), (429, "rate_limited"), (503, "service_unavailable")],
)
def test_stream_http_error_body_is_read_before_parsing(
    status_code: int,
    expected_code: str,
) -> None:
    """Regression: non-2xx streamed bodies must not raise httpx.ResponseNotRead."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            headers={"content-type": "application/json"},
            json={"detail": "Provider unavailable"},
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with patch("frontend.api_client.httpx.Client", return_value=client):
        with pytest.raises(ChatStreamError) as exc_info:
            _post_chat_stream_once("hello", None, None)

    assert exc_info.value.code == expected_code
    assert "Provider unavailable" in exc_info.value.message
    assert "streaming response content" not in exc_info.value.message


def test_stream_rejects_non_sse_success_response() -> None:
    """A successful HTML/JSON response must not be parsed as an SSE stream."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="not an event stream",
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with patch("frontend.api_client.httpx.Client", return_value=client):
        with pytest.raises(ChatStreamError) as exc_info:
            _post_chat_stream_once("hello", None, None)

    assert exc_info.value.code == "content_type_error"


def test_stream_http_error_redacts_secret_like_detail() -> None:
    """Backend error text must not echo credentials into the dashboard."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            502,
            headers={"content-type": "application/json"},
            json={"detail": "authorization=Bearer-secret api_key=gsk-sensitive"},
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with patch("frontend.api_client.httpx.Client", return_value=client):
        with pytest.raises(ChatStreamError) as exc_info:
            _post_chat_stream_once("hello", None, None)

    assert "Bearer-secret" not in exc_info.value.message
    assert "gsk-sensitive" not in exc_info.value.message
