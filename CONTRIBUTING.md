# Contributing to PumpkinSpice

PumpkinSpice is held to release quality. Every change ships typed, linted,
tested, and documented.

## Setup

```bash
uv sync                      # core + dev tooling
uv run pre-commit install    # run the gates on every commit
```

## The quality gates (CI runs all of these)

```bash
uv run ruff format --check .    # formatting
uv run ruff check .             # lint
uv run mypy                     # strict type-check
uv run pytest                   # tests + coverage (>= 80%)
```

`uv run ruff format . && uv run ruff check --fix .` fixes most lint/format issues.

## Architecture rules (non-negotiable invariants)

These are release-blocking. A change that violates one will not be merged.

1. **The microkernel never imports a concrete backend.** Add capabilities as
   plugins behind a `Protocol` in `contracts.py`, registered via an entry point
   in `pyproject.toml`. The kernel resolves plugins by `(group, name)` only.
2. **Retrieval is plain top-k vector search.** It must never delegate to the
   HADES CLI's hybrid / rerank / structural retrieval. HADES is build-side only
   (seeding the corpus), never part of the runtime.
3. **Least privilege for data access.** Runtime plugins connect with scoped,
   read-only credentials from their own env/config, never the root DB passwords.
   No plugin may write to or read another experiment's database.

See `CLAUDE.md` and `pumpkinspice-Spec.md` for the full rationale (these encode
the experiment's fairness and OPSEC contract).

## Adding a plugin

1. Implement the slot's `Protocol` from `contracts.py` (a class taking a single
   `config: dict` constructor arg).
2. Register it under the slot's entry-point group in `pyproject.toml`.
3. `uv sync` to refresh entry points, then `uv run pumpkinspice plugins` to
   confirm discovery.
4. Add tests (mock external services with `httpx.MockTransport` or fakes).
