# Contributing

## Local workflow

1. Use Python 3.12 and create a fresh virtual environment.
2. Install `requirements.lock`, then `pip install -e ".[dev]"`.
3. Copy `.env.example` to `.env` and use only local placeholder/test credentials.
4. Make focused changes that preserve the artifact and API contracts.
5. Run the tests, Ruff, and compile checks documented in `docs/testing.md`.

## Data and secrets

Never commit:

- `.env` or API credentials;
- dunnhumby source CSVs or other private customer data;
- generated parquet, Chroma indexes, checkpoints, logs, or local databases;
- prompts or traces containing sensitive customer data.

Tests must use small synthetic fixtures and mocked external services.

## Analytics changes

Explain any change to feature meaning, filtering, cluster selection, lift, or campaign calculations. Add a hand-checkable unit test and avoid presenting observational results as causal. Preserve fixed random seeds unless the change explicitly concerns reproducibility.

## UI changes

Keep the Streamlit application lightweight and keyboard-readable. Every page must have a useful missing/empty-data state. Dynamic counts and metrics must come from artifacts or the API, not hardcoded display values.

## Pull requests

Include the problem, implementation, tests actually run, and remaining limitations. Do not include generated screenshots unless they were captured from the changed application. Repository licensing and upstream attribution must be resolved by the maintainer before accepting outside contributions.

