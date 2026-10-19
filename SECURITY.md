# Security policy

## Supported versions

Until the first public release, security fixes are applied to the current `main` branch only.

## Reporting a vulnerability

Do not open a public issue containing credentials, customer data, exploit details, or private source records. Use the repository owner's private security-reporting channel after it is configured on GitHub.

If private reporting is not yet enabled, contact the maintainer privately and include only:

- the affected component and version or commit;
- minimal reproduction steps using synthetic data;
- likely impact; and
- suggested remediation, if known.

Never attach `.env`, licensed dataset rows, full prompts containing private data, database files, or traces with authorization headers.

## Security boundaries

- The included API-key check is appropriate for a trusted local environment; it is not a complete internet-facing identity system.
- Do not expose FastAPI or Streamlit directly to the public internet without TLS, managed authentication, network controls, and production secret management.
- LangSmith tracing is opt-in because traces may contain prompts and retrieved context.
- The Chroma server is not exposed; the optional integration uses a local persistent client.

## Dependency reports

CI runs `pip-audit` against the lockfile. Dependabot is configured for monthly Python and GitHub Actions updates. Review transitive ML dependencies carefully because CPU/GPU package variants can differ substantially.

