# Repository Guidelines

## Project Structure & Module Organization

PumpkinSpice is a Python 3.12 `src/`-layout package with a Vite/React frontend.
Core runtime code lives in `src/pumpkinspice/`; plugin implementations live in
`src/pumpkinspice/plugins/`; FastAPI web code lives in `src/pumpkinspice/web/`.
Tests are in `tests/` and mirror behavior by feature. Runtime configurations are
in `configs/`. Utility and database bootstrap scripts are in `scripts/`.
Frontend source is in `frontend/src/`.

## Build, Test, and Development Commands

- `uv sync`: install the Python package plus development tooling.
- `uv run pumpkinspice plugins`: list registered microkernel plugins.
- `uv run pumpkinspice run --config configs/offline.toml`: run the offline path
  without external services.
- `uv run ruff format --check . && uv run ruff check .`: verify formatting and
  lint rules.
- `uv run mypy`: run strict Python type checks for `src/`.
- `uv run pytest`: run tests with coverage; configuration enforces at least 80%.
- `cd frontend && pnpm install && pnpm build`: install and build the React SPA.
- `cd frontend && pnpm dev`: start the frontend dev server.

## Coding Style & Naming Conventions

Use Ruff for Python formatting and linting. Python targets 3.12, line length is
100, and MyPy runs in strict mode. Keep plugin code behind the protocols in
`src/pumpkinspice/contracts.py`; register plugin entry points in `pyproject.toml`
instead of importing concrete backends into the kernel. Use descriptive module
and test names such as `retrieval_pgvector.py` and `test_pgvector.py`. Frontend
TypeScript is strict and uses React JSX; keep components in PascalCase files.

## Testing Guidelines

Pytest discovers tests under `tests/` using `test_*.py` naming. Prefer focused
unit tests with fakes or `httpx.MockTransport` for external services. Do not
require live databases, LMStudio, or HeroBench for default tests. Maintain the
configured 80% coverage floor.

## Commit & Pull Request Guidelines

This checkout has no Git commits, so no project-specific convention can be
inferred. Use concise, imperative subjects such as `Add Arango seed script` or
`Fix prompt replan parsing`. Pull requests should describe the change, list
validation commands, link issues, and include screenshots for UI changes.

## Security & Configuration Tips

Runtime database and decoder access must use scoped, least-privilege credentials.
Do not commit `.env.local`, root database passwords, captures with secrets, or
machine-specific service URLs. Retrieval must remain plain top-k vector search;
HADES-style hybrid, rerank, or structural retrieval is build-side only.
