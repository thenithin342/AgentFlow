# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] - 2026-07-07
### Added
- Observability features: `/health` per-component timing and `X-Request-ID` middleware.
- Test coverage for LTM, STM, Blog generation, and Wikipedia network failures.
- `pyproject.toml`, `LICENSE`, and GitHub Actions workflow.
- UX improvements: Character limit warning when approaching 16k chars.

### Fixed
- Replaced dangerous `.innerHTML` DOM manipulation with `rehype-sanitize` for React markdown.
- Standardized thread-locking across the LangGraph checkpoint layer.
- `asyncio.wait_for` added to LangGraph `ainvoke` calls.
- `setTimeout` race conditions cleaned up in React frontend error management.
- SQL queries in `list_threads` properly grouped.
- Added concurrency limiting semaphore in `list_threads` to prevent DB starvation.
