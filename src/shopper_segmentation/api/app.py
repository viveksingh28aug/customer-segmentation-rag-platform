"""FastAPI backend for shopper segmentation and RAG chatbot."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from shopper_segmentation.api.auth import verify_api_key
from shopper_segmentation.api.sessions import resolve_thread_id
from shopper_segmentation.artifacts import (
    ArtifactError,
    ensure_artifacts,
    load_profiles,
    load_recommendations,
)
from shopper_segmentation.paths import segments_path
from shopper_segmentation.rag.agent.graph import get_compiled_graph_if_ready, shutdown_graph
from shopper_segmentation.rag.rag_chain import (
    MAX_QUERY_CHARS,
    answer_query_async,
    get_groq_api_key,
)
from shopper_segmentation.segmentation import MIN_SEGMENT_CONFIDENT_N, is_low_confidence_segment

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_ALLOWED_ORIGINS = "http://localhost:8504"


def get_rate_limit_key(request: Request) -> str:
    """Return rate-limit bucket key from API key, thread id, or client IP.

    Args:
        request: Incoming HTTP request.

    Returns:
        Identifier used for rate limiting.
    """
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return api_key

    thread_id = request.query_params.get("thread_id")
    if thread_id:
        return f"thread:{thread_id}"

    return get_remote_address(request)


def get_allowed_origins() -> list[str]:
    """Parse comma-separated CORS origins from environment.

    Returns:
        List of allowed origin URLs.
    """
    raw = os.getenv("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


limiter = Limiter(key_func=get_rate_limit_key)


async def rate_limit_exceeded_handler(
    request: Request,
    exc: RateLimitExceeded,
) -> JSONResponse:
    """Return an async 429 response without dispatching through a worker thread."""
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Wait briefly, then retry."},
        headers={"Retry-After": "60"},
    )


class SegmentSummary(BaseModel):
    """Summary view of a shopper segment."""

    id: int
    name: str
    size: int
    narrative: str
    low_confidence: bool


class SegmentDetail(BaseModel):
    """Detailed segment profile."""

    id: int
    name: str
    size: int
    narrative: str
    feature_means: dict[str, float]
    low_confidence: bool


class ProductRecommendation(BaseModel):
    """Product recommendation for a segment."""

    product_id: int
    department: str | None = None
    brand: str | None = None
    commodity_desc: str | None = None
    lift: float
    segment_purchase_rate: float
    population_purchase_rate: float
    segment_buyers: int
    segment_size: int


class SegmentRecommendations(BaseModel):
    """Recommendations payload for one segment."""

    segment_id: int
    segment_name: str
    recommendations: list[ProductRecommendation]


class ChatRequest(BaseModel):
    """Chat request body."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUERY_CHARS,
        description="Natural language analyst question",
    )
    thread_id: str | None = Field(
        default=None,
        description="Optional LangGraph session thread id for conversational memory",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """Trim questions and reject whitespace-only input."""
        clean = value.strip()
        if not clean:
            raise ValueError("Question cannot be empty.")
        return clean


class RetrievedSegment(BaseModel):
    """Segment card retrieved for a chat query."""

    segment_id: int
    segment_name: str
    distance: float


class ValidationResult(BaseModel):
    """Number validation result for chat responses."""

    validated: bool
    numbers_found: list[str]
    unsupported_numbers: list[str]


class ChatResponse(BaseModel):
    """Chat response with answer and validation metadata."""

    query: str
    answer: str
    retrieved_segments: list[RetrievedSegment]
    validation: ValidationResult


class ApplicationStatus(BaseModel):
    """Dashboard-facing artifact and optional-service status."""

    data_mode: str
    artifact_source: str
    segment_count: int
    household_count: int
    pca_available: bool
    chat_configured: bool


def _segment_low_confidence(segment: dict[str, Any]) -> bool:
    """Return whether a segment should be flagged as low confidence.

    Args:
        segment: Segment profile dictionary.

    Returns:
        True when the segment is below the configured household threshold.
    """
    if "low_confidence" in segment:
        return bool(segment["low_confidence"])
    return is_low_confidence_segment(int(segment["size"]), MIN_SEGMENT_CONFIDENT_N)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm lightweight artifacts without initializing optional AI services."""
    import logging

    logger = logging.getLogger(__name__)
    try:
        ensure_artifacts()
    except ArtifactError as exc:
        logger.error("Startup artifact initialization failed: %s", exc)
    yield
    if get_compiled_graph_if_ready() is not None:
        await shutdown_graph()


def _require_profiles() -> dict[str, Any]:
    """Load segment profiles or raise a service-unavailable error."""
    try:
        return load_profiles()
    except ArtifactError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _require_recommendations() -> dict[str, Any]:
    """Load segment recommendations or raise a service-unavailable error."""
    try:
        return load_recommendations()
    except ArtifactError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _get_segment_profile(segment_id: int) -> dict[str, Any]:
    """Return a segment profile by id.

    Args:
        segment_id: Segment identifier.

    Returns:
        Segment profile dictionary.

    Raises:
        HTTPException: If segment is not found.
    """
    profiles = _require_profiles()["segments"]
    for segment in profiles:
        if int(segment["id"]) == segment_id:
            return segment
    raise HTTPException(status_code=404, detail=f"Segment {segment_id} not found")


def _get_segment_recommendations(segment_id: int) -> dict[str, Any]:
    """Return recommendations for a segment.

    Args:
        segment_id: Segment identifier.

    Returns:
        Recommendation record for the segment.

    Raises:
        HTTPException: If segment is not found.
    """
    for segment in _require_recommendations()["segments"]:
        if int(segment["segment_id"]) == segment_id:
            return segment
    raise HTTPException(status_code=404, detail=f"Segment {segment_id} not found")


app = FastAPI(
    title="Shopper Segmentation & Personalization API",
    description=(
        "REST API for household segment profiles, product recommendations, "
        "and RAG-powered analyst chat."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

protected = [Depends(verify_api_key)]


@app.get("/segments", response_model=list[SegmentSummary], tags=["segments"], dependencies=protected)
async def list_segments() -> list[SegmentSummary]:
    """List all shopper segments with summary metadata."""
    return [
        SegmentSummary(
            id=int(segment["id"]),
            name=str(segment["name"]),
            size=int(segment["size"]),
            narrative=str(segment["narrative"]),
            low_confidence=_segment_low_confidence(segment),
        )
        for segment in _require_profiles()["segments"]
    ]


@app.get(
    "/segments/{segment_id}",
    response_model=SegmentDetail,
    tags=["segments"],
    dependencies=protected,
)
async def get_segment(segment_id: int) -> SegmentDetail:
    """Get detailed profile for a single segment."""
    segment = _get_segment_profile(segment_id)
    return SegmentDetail(
        id=int(segment["id"]),
        name=str(segment["name"]),
        size=int(segment["size"]),
        narrative=str(segment["narrative"]),
        feature_means={k: float(v) for k, v in segment["feature_means"].items()},
        low_confidence=_segment_low_confidence(segment),
    )


@app.get(
    "/segments/{segment_id}/recommendations",
    response_model=SegmentRecommendations,
    tags=["segments"],
    dependencies=protected,
)
async def get_recommendations(segment_id: int) -> SegmentRecommendations:
    """Get top product recommendations for a segment."""
    segment = _get_segment_recommendations(segment_id)
    recommendations = [
        ProductRecommendation(
            product_id=int(rec["product_id"]),
            department=rec.get("department"),
            brand=rec.get("brand"),
            commodity_desc=rec.get("commodity_desc"),
            lift=float(rec["lift"]),
            segment_purchase_rate=float(rec["segment_purchase_rate"]),
            population_purchase_rate=float(rec["population_purchase_rate"]),
            segment_buyers=int(rec["segment_buyers"]),
            segment_size=int(rec["segment_size"]),
        )
        for rec in segment.get("recommendations", [])
    ]
    return SegmentRecommendations(
        segment_id=int(segment["segment_id"]),
        segment_name=str(segment["segment_name"]),
        recommendations=recommendations,
    )


@app.get("/status", response_model=ApplicationStatus, tags=["system"], dependencies=protected)
async def application_status() -> ApplicationStatus:
    """Return verified data and optional-service status for the dashboard."""
    profiles = _require_profiles()
    segments = profiles.get("segments", [])
    metadata = profiles.get("metadata", {})
    source = str(metadata.get("source", "pipeline_output"))
    try:
        get_groq_api_key()
        chat_configured = True
    except RuntimeError:
        chat_configured = False
    return ApplicationStatus(
        data_mode="sample" if source == "bundled_sample_output" else "full",
        artifact_source=source,
        segment_count=len(segments),
        household_count=sum(int(segment.get("size", 0)) for segment in segments),
        pca_available=segments_path().exists(),
        chat_configured=chat_configured,
    )


def _sse(data: dict[str, object]) -> str:
    """Format a dictionary as a Server-Sent Event data line."""
    return f"data: {json.dumps(data, default=str)}\n\n"


def _sse_error(code: str, message: str) -> str:
    """Format a structured SSE error event."""
    return _sse({"type": "error", "error": {"code": code, "message": message}})


def _safe_chat_error(exc: Exception) -> tuple[str, str]:
    """Map provider/internal failures to non-sensitive public messages."""
    status_code = getattr(exc, "status_code", None)
    class_name = type(exc).__name__.lower()
    if status_code == 401 or "authentication" in class_name:
        return (
            "provider_authentication_error",
            "Groq rejected the configured API key. Check GROQ_API_KEY and restart the API.",
        )
    if status_code == 429 or "ratelimit" in class_name or "rate_limit" in class_name:
        return (
            "provider_rate_limited",
            "Groq rate-limited the request. Wait briefly, then retry.",
        )
    if "timeout" in class_name:
        return (
            "provider_timeout",
            "The language-model provider timed out. Retry the question.",
        )
    if "connection" in class_name:
        return (
            "provider_connection_error",
            "The API could not reach the language-model provider. Check connectivity and retry.",
        )
    if isinstance(exc, ValueError):
        return "invalid_chat_response", str(exc)[:300]
    return (
        "chat_service_error",
        "The analyst service could not complete the request. Check the API logs for details.",
    )


def _answer_chunks(answer: str) -> list[str]:
    """Split a completed answer into readable SSE chunks for the local UI."""
    return re.findall(r"\S+\s*", answer)


def _build_chat_response(result: dict[str, object]) -> ChatResponse:
    """Map an agent result dictionary to the public chat response model."""
    retrieved = [
        RetrievedSegment(
            segment_id=int(card["segment_id"]),
            segment_name=str(card["segment_name"]),
            distance=float(card["distance"]),
        )
        for card in result.get("retrieved_cards", [])
    ]
    validation_raw = result.get("validation") or {}
    validation = ValidationResult(
        validated=bool(validation_raw.get("validated", False)),
        numbers_found=[str(n) for n in validation_raw.get("numbers_found", [])],
        unsupported_numbers=[
            str(n) for n in validation_raw.get("unsupported_numbers", [])
        ],
    )
    return ChatResponse(
        query=str(result["query"]),
        answer=str(result.get("answer", "")),
        retrieved_segments=retrieved,
        validation=validation,
    )


async def _chat_event_stream(
    query: str,
    thread_id: str,
    emit_thread_id: bool,
) -> AsyncIterator[str]:
    """Yield a reliable SSE envelope around a non-streaming model call.

    Provider token streaming is intentionally disabled here. LangGraph's event
    stream includes router and retry-model events in addition to the final answer,
    which can duplicate or leak intermediate text. The complete grounded answer is
    generated first, then delivered in UI-friendly chunks.
    """

    logger.info(
        "Chat stream start thread_id=%s query_len=%s",
        thread_id,
        len(query),
    )
    token_count = 0
    try:
        if emit_thread_id:
            yield _sse({"type": "thread_id", "thread_id": thread_id})

        try:
            result = await answer_query_async(query, thread_id=thread_id)
        except Exception as exc:
            logger.exception(
                "Chat stream failed thread_id=%s error_type=%s",
                thread_id,
                type(exc).__name__,
            )
            code, message = _safe_chat_error(exc)
            yield _sse_error(code, message)
            return

        answer = str(result.get("answer") or "").strip()
        if not answer:
            yield _sse_error(
                "empty_response",
                "The language-model provider returned no answer text. Retry the request.",
            )
            return

        for chunk in _answer_chunks(answer):
            token_count += 1
            yield _sse({"type": "token", "content": chunk})

        payload = _build_chat_response(result).model_dump()
        payload["type"] = "done"
        payload["thread_id"] = thread_id
        logger.info(
            "Chat stream done thread_id=%s tokens=%s",
            thread_id,
            token_count,
        )
        yield _sse(payload)
    finally:
        logger.info("Chat stream end thread_id=%s tokens=%s", thread_id, token_count)


@app.post("/chat", tags=["chat"], dependencies=protected, response_model=None)
@limiter.limit("10/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    stream: bool = True,
):
    """Ask the analyst chatbot a question about segments and promotions."""
    try:
        get_groq_api_key()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    thread_id = resolve_thread_id(request, body.thread_id)

    if stream:
        logger.info(
            "Chat request stream=true thread_id=%s query_len=%s",
            thread_id,
            len(body.query),
        )
    else:
        logger.info(
            "Chat request stream=false thread_id=%s query_len=%s",
            thread_id,
            len(body.query),
        )

    if not stream:
        try:
            result = await answer_query_async(body.query, thread_id=thread_id)
        except Exception as exc:
            logger.exception(
                "Chat request failed thread_id=%s error_type=%s",
                thread_id,
                type(exc).__name__,
            )
            code, message = _safe_chat_error(exc)
            status_code = 429 if code == "provider_rate_limited" else 502
            raise HTTPException(status_code=status_code, detail=message) from exc
        if not str(result.get("answer") or "").strip():
            raise HTTPException(
                status_code=502,
                detail="The language-model provider returned no answer text. Retry the request.",
            )
        return _build_chat_response(result)

    return StreamingResponse(
        _chat_event_stream(body.query, thread_id, body.thread_id is None),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
