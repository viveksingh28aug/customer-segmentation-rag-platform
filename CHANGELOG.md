# Changelog

All notable changes will be documented here. This project has not yet published a tagged release.

## Unreleased

### Added

- Professional Streamlit analytics layout and shared visual system
- Verified application-status endpoint and explicit sample/full-data states
- Lightweight TF-IDF retrieval with optional Chroma/MiniLM mode
- Grounded chat source references, numeric guardrails, and conversation reset
- Regression coverage for HTTP streaming, provider errors, data validation, and UI helpers
- Architecture, testing, troubleshooting, contribution, and security documentation
- GitHub issue forms, pull-request template, and dependency-update configuration

### Fixed

- Consumed streamed HTTP error bodies before parsing, preventing `httpx.ResponseNotRead`
- Sanitized provider failures and rejected malformed, interrupted, and empty chat responses
- Removed hardcoded dashboard metrics and hardcoded RAG household totals
- Added actionable missing-PCA and missing-credential states
- Added validation for clustering inputs and all CSV columns used downstream

### Changed

- Provider token streaming was replaced by a reliable SSE envelope around the complete grounded answer
- Groq model identifiers became configurable and use current defaults
- In-memory conversation checkpoints are now the lightweight default
- Python 3.12 is the recommended runtime

## Release policy

Do not create a public tag or release until upstream attribution and licensing are resolved and a compatible `LICENSE` file can be added legally.

