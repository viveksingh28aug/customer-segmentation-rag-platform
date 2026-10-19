# Testing

## Standard checks

From an activated virtual environment:

```bash
pytest tests/ -v
ruff check src tests frontend
python -m compileall -q src frontend tests
```

The suite is hermetic: provider calls are mocked, LangSmith is disabled, fixtures are synthetic, and the FastAPI client uses `httpx.ASGITransport` without opening a port.

## Coverage areas

- source CSV schema errors;
- RFM, promotion, category, and demographic features;
- empty, duplicate, missing, and insufficient clustering inputs;
- deterministic pipeline integration on small fixture CSVs;
- product lift and campaign confidence intervals;
- sample artifact initialization and API serving;
- RAG routing, grounding, guardrail retry, and response caching;
- TF-IDF and optional Chroma retrieval contracts;
- missing credentials, invalid questions, provider rate limits, empty answers, and sanitized errors;
- SSE parsing, malformed/interrupted streams, content-type checks, resource cleanup, and the original unread-stream regression;
- dashboard import/path and sample radar helpers.

## No real model calls

Do not add tests that require `GROQ_API_KEY`, external network access, or model downloads. Model behavior should be represented with `AsyncMock`/`MagicMock` responses, including failures.

## Manual smoke test

1. Start FastAPI and confirm `/health` returns `{"status":"ok"}`.
2. Start Streamlit and confirm the home page reports the served data mode and counts.
3. Open each page and verify sample mode does not show an empty PCA region.
4. Without `GROQ_API_KEY`, confirm Chat is disabled with setup guidance.
5. With a valid key, ask a suggested question and confirm an answer, retrieved segments, and validation status appear.
6. Stop FastAPI and confirm the dashboard reports an actionable connection error.

Real provider smoke tests are manual and optional; do not record their output as an automated test result.

