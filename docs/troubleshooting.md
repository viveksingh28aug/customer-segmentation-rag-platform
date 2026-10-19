# Troubleshooting

## Dashboard says the API is unavailable

Start the backend from the repository root:

```bash
uvicorn --app-dir src shopper_segmentation.api.app:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000/health>. If Streamlit uses another address, set `API_BASE_URL`.

## 401 from the API

`API_KEY` is missing or differs between the backend and Streamlit processes. Put one non-placeholder value in the root `.env` and restart both processes. Do not paste it into logs or screenshots.

## Analyst Chat is disabled or returns 503

Set a valid `GROQ_API_KEY` and restart FastAPI. The dashboard deliberately does not label a local deterministic response as Groq output.

## Chat is rate-limited

Local chat requests are limited to 10 per minute per API key. Groq can also return 429 based on account limits. Wait, then retry. The UI identifies whether the rate limit came from the provider when that information is available.

## “Attempted to access streaming response content…”

This was caused by accessing `.text`/`.json()` on an unread `httpx` streaming response after a non-2xx API result. The client now calls `response.read()` inside the open stream context and has a regression test. If the old message persists, stop both processes, confirm the updated `frontend/api_client.py` is running, and restart.

## PCA is unavailable

This is expected in bundled sample mode. PCA needs household-level rows, which are intentionally not bundled. Set `DATA_DIR` to the folder containing all eight CSVs and run:

```bash
bash scripts/run_pipeline.sh
```

The distribution, KPIs, radar profile, and recommendations remain available without PCA.

## Pipeline reports a missing file or column

Compare `DATA_DIR` with the filename list in the README. Filenames and required column case must match. Validation reports the specific file and missing columns; it does not silently invent them.

## Chroma retrieval fails while importing PyTorch/CUDA

Use the lightweight default:

```env
RETRIEVAL_BACKEND=tfidf
```

The Chroma path is optional and depends on a CPU/GPU PyTorch build compatible with the local Python version. It is not required for the dashboard or chat.

## LangSmith warnings or unwanted network attempts

Set:

```env
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
```

Tracing is optional. Tests force it off.

## Python version problems

Use Python 3.12 for parity with CI and `requirements.lock`. Recreate, rather than reuse, a virtual environment made under a different Python minor version.

