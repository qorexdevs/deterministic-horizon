# Security Policy

## Supported Versions

The `1.0.x` line receives security fixes. Older preview versions are unsupported.

## Reporting a Vulnerability

Please **do not** open a public issue. Instead email the maintainers via the
address listed in `pyproject.toml` (or open a private security advisory through
GitHub). We aim to acknowledge reports within 72 hours and ship a fix within
14 days for high-severity issues.

## Scope

In scope:
- Code-execution or path-traversal in any CLI entry point
- Credential leakage via logging or error messages
- Supply-chain issues in pinned dependencies

Out of scope:
- Misuse of API keys you placed in `.env` (rotate them)
- Issues in third-party LLM providers
