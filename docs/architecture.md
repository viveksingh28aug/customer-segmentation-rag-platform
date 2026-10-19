# Architecture

## Runtime boundaries

The application has two processes:

1. FastAPI loads aggregate artifacts and exposes status, segment, recommendation, and chat routes.
2. Streamlit calls those routes through `frontend/api_client.py`; it does not read analytical JSON directly except for local chart helpers that need optional PCA/features parquet.

The shared `API_KEY` protects analytical routes. `/health` is intentionally unauthenticated and reports liveness only.

## Batch data flow

| Stage | Inputs | Outputs |
|---|---|---|
| ETL | transactions, products, demographics | `household_features_raw.parquet` |
| Features | ETL parquet plus campaign, coupon, causal CSVs | `household_features.parquet` |
| Segmentation | engineered features | household assignments/PCA parquet, profile JSON/Markdown |
| Personalization | assignments, profiles, transactions, campaigns | recommendation and campaign JSON |

Schema validation runs before ETL. Segmentation rejects empty, duplicate-household, nonnumeric, missing, and infinite feature matrices. Randomized estimators use the fixed seed in `segmentation.py`.

## Sample artifacts

`artifacts.ensure_artifacts()` is idempotent and guarded by a process lock. If output JSON is missing, it runs the full pipeline only when all required CSVs exist. Otherwise, it copies the bundled aggregate sample. It never downloads source data.

Artifact metadata identifies `bundled_sample_output`; `/status` turns this into the user-facing data-mode label. PCA is independently detected from `household_segments.parquet`.

## Chat request flow

```text
Streamlit form
  -> httpx POST /chat?stream=true
  -> API key + Pydantic validation + rate limit
  -> LangGraph router
  -> structured lookup or local retrieval (TF-IDF by default)
  -> bounded context prompt
  -> Groq primary model, then configured fallback
  -> numeric guardrail; one regeneration attempt if needed
  -> complete validated answer
  -> SSE thread/token/done events
  -> Streamlit history + retrieved-segment references
```

The model request is non-streaming by design. LangGraph event streaming also contains router and retry output, which is unsuitable as a user answer. The API therefore completes grounding and validation first, then wraps the final answer in SSE chunks. This retains a responsive UI without exposing intermediate text or relying on provider stream internals.

On non-2xx SSE responses, the frontend reads the open `httpx` response body before accessing JSON or text. This is required by `httpx` and prevents `ResponseNotRead` from masking the real backend error.

## Retrieval modes

- `tfidf` is the default: eight small segment cards are indexed locally per query with scikit-learn. It has no model download and is suitable for laptops.
- `chroma` retains the original Chroma/MiniLM path. It can download model weights on first use and requires a compatible CPU PyTorch installation.

Both return the same card contract: segment id/name, source content, and lower-is-closer distance.

## Security and privacy

- credentials come from environment variables and are never returned in diagnostics;
- API-key comparison uses constant-time comparison;
- public provider errors are mapped to bounded, stable messages;
- prompt text is logged only as length or an 80-character query preview; full context is not logged by the API;
- LangSmith tracing is opt-in and disabled in `.env.example` and tests;
- source data, output artifacts, caches, local databases, and `.env` are ignored.

