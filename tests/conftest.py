"""Shared pytest fixtures for API tests."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
import pytest

# Tests are hermetic: never send LangSmith traces or make provider calls.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ.pop("LANGCHAIN_API_KEY", None)

TEST_API_KEY = "test-secret-key"


@pytest.fixture(scope="session", autouse=True)
def initialize_artifacts() -> None:
    """Ensure pipeline artifacts exist before any tests run."""
    from shopper_segmentation.artifacts import ensure_artifacts

    ensure_artifacts()


@pytest.fixture(autouse=True)
def api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set API_KEY for all tests."""
    monkeypatch.setenv("API_KEY", TEST_API_KEY)


class ASGITestClient:
    """Small synchronous facade over httpx's async in-process ASGI transport."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self.loop = asyncio.new_event_loop()
        self.async_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://testserver",
        )

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        return self.loop.run_until_complete(
            self.async_client.request(method, path, **kwargs)
        )

    def get(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", path, **kwargs)

    def close(self) -> None:
        self.loop.run_until_complete(self.async_client.aclose())
        self.loop.close()


@pytest.fixture
def client() -> ASGITestClient:
    """Return a deadlock-free, networkless FastAPI client."""
    from shopper_segmentation.api.app import app

    test_client = ASGITestClient(app)
    try:
        yield test_client
    finally:
        test_client.close()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Return headers with a valid API key."""
    return {"X-API-Key": TEST_API_KEY}
