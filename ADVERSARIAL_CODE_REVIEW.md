# Adversarial Code Review

Date: 2026-07-03  
Scope: current workspace under `/home/todd/git/PunpkinSpice`  
Focus: security posture, abuse resistance, type-safety, and validation quality

## Executive Summary

The Python core has good baseline engineering discipline: Ruff passes, MyPy strict
passes, frontend TypeScript builds, secrets are ignored, and the database bootstrap
scripts intentionally create least-privilege runtime identities. The highest risk is
not the microkernel itself; it is the web backend’s operator API.

If `pumpkinspice serve` is reachable by anything other than a trusted local user, the
API should be treated as unsafe. It has no authentication, permissive CORS, arbitrary
backend URL proxying, capture disclosure endpoints, run-launch endpoints, and an MCP
manager that can persist and spawn arbitrary local commands. These are acceptable only
for a strictly loopback-only, single-user lab tool. They are not safe defaults for a
LAN-facing service.

Testing also has a release-blocking issue: the suite currently hangs in the first web
test under the installed FastAPI/Starlette/httpx stack. CI will likely wedge unless
versions or the test transport are corrected.

## Validation Performed

- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .` — passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run mypy` — passed for 34 source files.
- `cd frontend && pnpm build` — passed.
- `env UV_CACHE_DIR=/tmp/uv-cache timeout 120s uv run pytest -vv` — timed out.
- Isolated `TestClient(create_app()).get("/api/health")` also hung. Installed versions:
  FastAPI `0.138.1`, Starlette `1.3.1`, httpx `0.28.1`, pytest `9.1.1`.

No additional AST, tree-sitter, Bandit, Semgrep, or dependency-audit tools were
installed during this review.

## Findings

### Critical: Unauthenticated API Can Spawn Arbitrary Local Commands

`src/pumpkinspice/web/app.py:520` accepts MCP server definitions from any caller, with
caller-controlled `name`, `command`, `args`, and `enabled`. `src/pumpkinspice/web/mcp_host.py:69`
then passes those values into `stdio_client(...)`, which spawns the configured command
when `/api/chat` enters the tool loop at `src/pumpkinspice/web/app.py:586`.

Impact: if the web server is exposed beyond a trusted local operator, this is remote
command execution. The command is not shell-interpolated, which is good, but an attacker
can still choose an executable such as `python`, `sh`, `bash`, `node`, `npx`, or `uvx`
and supply arguments.

Enforce:

- Require authentication for every `/api/*` endpoint before enabling MCP.
- Disable MCP spawning by default behind an explicit local-only feature flag.
- Replace arbitrary `command` with an allowlisted registry of known MCP servers.
- Validate server names and args against schemas per allowed server.
- Run MCP servers in a sandboxed subprocess account with a locked-down environment.
- Add audit logging for MCP add/remove/enable and chat tool execution.

### Critical: No Authentication plus Permissive CORS

The FastAPI app sets `allow_origins=["*"]`, `allow_methods=["*"]`, and
`allow_headers=["*"]` at `src/pumpkinspice/web/app.py:411`. The same app exposes
state-changing endpoints for model settings, model loading, MCP server management,
run/trial launches, report annotations, stop controls, and capture reads. The CLI
defaults to `127.0.0.1` at `src/pumpkinspice/cli.py:390`, which helps, but a user can
bind to `0.0.0.0` without any additional protection.

Impact: a malicious web page, LAN peer, or local untrusted process can drive expensive
model calls, mutate settings, start long benchmark runs, read captures, or combine this
with MCP spawning.

Enforce:

- Add a required bearer token or session secret for all API endpoints.
- Restrict CORS to the served frontend origin; do not use `*`.
- Add CSRF protection if browser credentials are ever used.
- Warn or refuse startup when `--host` is not loopback unless auth is configured.

### High: User-Controlled URL Proxying Enables SSRF and Network Pivoting

The web API accepts caller-controlled backend URLs in several places:

- `ChatRequest.base_url` at `src/pumpkinspice/web/app.py:75`.
- `/api/decoder/models` at `src/pumpkinspice/web/app.py:431`.
- `/api/decoder/loaded` at `src/pumpkinspice/web/app.py:441`.
- `/api/model/available` at `src/pumpkinspice/web/app.py:468`.
- `/api/world/map` at `src/pumpkinspice/web/app.py:550`.
- `/api/chat` streams to `req.base_url` at `src/pumpkinspice/web/app.py:647`.

Impact: an attacker can make the server connect to internal services reachable from the
host. Even when response parsing limits what is returned, this still creates a network
oracle, service scanner, and request amplifier.

Enforce:

- Replace arbitrary URLs with configured backend IDs.
- If arbitrary URLs remain necessary, validate scheme, host, port, DNS resolution, and
redirect behavior.
- Reject private, link-local, loopback, metadata, and Unix-socket-style targets unless
explicitly configured by the operator.
- Apply short timeouts and response-size caps to all proxy endpoints.

### High: Run Launch Accepts Absolute Config Paths

`/api/runs` accepts `config` and converts it to `Path(req.config)`. If the path is
absolute, it is used directly at `src/pumpkinspice/web/app.py:701`. Relative values are
also not constrained to simple config names before appending `.toml`.

Impact: any API caller can ask the service to parse arbitrary readable TOML files on the
host. If a writable path is available, this can steer plugin settings, capture paths,
backend URLs, max turns, and world endpoints. Combined with unauthenticated access, this
is a dangerous trust-boundary bypass.

Enforce:

- Accept only config names matching a conservative pattern such as
  `^[A-Za-z0-9_.-]+$`.
- Resolve the candidate path and require it to remain under `configs_dir`.
- Do not accept absolute paths from the web API.
- Return only sanitized config names in errors; avoid disclosing server paths.

### High: Resource Exhaustion Controls Are Incomplete

Request models define unconstrained values for `max_turns`, `trials`, `disks`,
`temperature`, `max_tokens`, message count, and message size at
`src/pumpkinspice/web/app.py:74`, `src/pumpkinspice/web/app.py:82`,
`src/pumpkinspice/web/app.py:96`, and `src/pumpkinspice/web/app.py:111`. Trial count is
clamped to 50 at `src/pumpkinspice/web/app.py:748`, but `max_turns` remains unbounded
for runs and trials. Each run or batch starts daemon threads at
`src/pumpkinspice/web/runs.py:190` and `src/pumpkinspice/web/runs.py:235`.

Impact: an unauthenticated caller can start many long-running decode jobs, use unbounded
chat generations, and hold network connections for minutes. This can exhaust CPU/GPU,
RAM, file descriptors, threads, capture storage, and downstream model capacity.

Enforce:

- Add Pydantic `Field(ge=..., le=..., max_length=...)` constraints to all API models.
- Bound single-run `max_turns`, Hanoi `disks`, chat messages, sampler keys, and request
  body size.
- Add a global run queue and per-client concurrency limits.
- Add server-side cancellation and a maximum wall-clock duration per run.
- Default `max_tokens` to a safe cap for chat; use explicit operator override for
  unbounded benchmark runs.

### Medium: SQL Identifier Injection Through Configurable pgvector Names

`src/pumpkinspice/plugins/retrieval_pgvector.py:48` through
`src/pumpkinspice/plugins/retrieval_pgvector.py:54` accept table, column, and schema
names from config. Those identifiers are interpolated into SQL at
`src/pumpkinspice/plugins/retrieval_pgvector.py:87` and
`src/pumpkinspice/plugins/retrieval_pgvector.py:135`. Values are parameterized, which is
good, but identifiers are not.

Impact: a malicious or compromised config can alter SQL structure. In the intended
runtime this is partly mitigated by read-only database credentials, but it can still
exfiltrate unexpected tables or break query behavior if grants are broader than
expected.

Enforce:

- Use `psycopg.sql.Identifier` for table, schema, and column names.
- Prefer a static allowlist for supported schema/table layouts.
- Test that malicious identifier strings are rejected before query execution.

### Medium: Capture and Reasoning Data Are Exposed Without Access Control

`/api/captures` enumerates JSONL files at `src/pumpkinspice/web/app.py:887`, and
`/api/captures/{name}` returns full turn records at `src/pumpkinspice/web/app.py:896`.
Turn records include prompts, world state, retrieval notes, raw model output, and
`reasoning_content` captured into `Turn.reasoning` at `src/pumpkinspice/loop.py:199`.
The frontend displays reasoning at `frontend/src/components/Playground.tsx:75` and raw
turn output at `frontend/src/components/TurnView.tsx:42`.

Impact: captures can contain sensitive prompts, model reasoning traces, model IDs,
operator notes, local endpoint names, and benchmark state. The current `.gitignore`
protects `captures/`, but the web API does not.

Enforce:

- Put capture and report endpoints behind auth.
- Add a redaction mode for prompts, reasoning, endpoint URLs, and tool results.
- Make chain-of-thought capture opt-in, not default, if these artifacts may be shared.
- Use `Path.resolve()` and reject symlinks in capture reads to harden the existing
  parent check at `src/pumpkinspice/web/app.py:900`.
- Add file-size caps before `read_text()` to avoid loading huge captures into memory.

### Medium: Shared Run State Is Mutated Across Threads Without Locks

`RunManager` stores mutable dictionaries and `RunRecord.turns` lists at
`src/pumpkinspice/web/runs.py:109`. Background threads append turns and mutate status
while API handlers read the same objects at `src/pumpkinspice/web/app.py:805` and
`src/pumpkinspice/web/app.py:819`.

Impact: under concurrent runs, SSE streams, stop calls, and report updates, clients can
observe inconsistent data or race with mutation. CPython’s GIL reduces crash likelihood
but does not make this a well-defined concurrency model.

Enforce:

- Protect `_runs`, `_stops`, `_batch_stops`, and each `RunRecord` with an `RLock`.
- Return immutable snapshots from `get()` and `list()`.
- Keep capture file writes and registry writes outside critical sections where possible.

### Medium: Test Suite Hangs Under Current Dependency Set

The pytest suite times out at `tests/test_web.py::test_health_backends_plugins`. An
isolated script hangs on `TestClient(...).get("/api/health")` before the endpoint
returns, while emitting a Starlette warning that `httpx` use is deprecated and `httpx2`
should be installed.

Impact: CI can hang indefinitely because `.github/workflows/ci.yml:35` runs plain
`uv run pytest` with no timeout. This masks regressions and blocks reliable release
gates.

Enforce:

- Pin compatible FastAPI/Starlette/httpx versions or migrate tests to the supported
  client transport for the installed stack.
- Add `pytest-timeout` or a CI-level timeout for the test job.
- Add a minimal `/api/health` smoke test outside `TestClient` if necessary.

### Low: Type Safety Is Strong at Tooling Level but Weak at Trust Boundaries

The project runs MyPy strict and passed, which is a strong baseline. The weak spots are
boundary models: config uses `dict[str, Any]` in `src/pumpkinspice/config.py:36`,
plugin loading returns `Any` at `src/pumpkinspice/kernel.py:41`, and web request models
use broad `dict[str, Any]`/`list[dict[str, str]]` shapes at
`src/pumpkinspice/web/app.py:77` and `src/pumpkinspice/web/app.py:78`. A scan found the
largest `Any` concentrations in `src/pumpkinspice/web/app.py`, `relational.py`,
`parity.py`, `reporting.py`, and `contracts.py`.

Impact: static typing catches internal mistakes, but malformed runtime data can still
cross into plugin constructors, HTTP payloads, SQL identifier config, sampler settings,
and capture records.

Enforce:

- Introduce typed config dataclasses or Pydantic models per plugin slot.
- Validate sampler keys and numeric ranges before forwarding to model backends.
- Change `kernel.load_plugin(...) -> Any` into overloads or explicit protocol casts with
  runtime `isinstance(..., Protocol)` checks.
- Prefer `TypedDict` or Pydantic response models for capture/report/web payloads.

### Low: Artifact Hygiene Needs Tightening

`.env.local` is ignored and currently mode `600`, and `captures/` is ignored, which is
good. However, `git status` shows `.coverage` staged/modified and an untracked
`2026-06-29_08-40.png`. Coverage databases and screenshots should not be committed
unless intentionally documented fixtures.

Enforce:

- Add `.coverage`, `htmlcov/`, and ad-hoc screenshots to `.gitignore`.
- Review staged files before any commit; this workspace has many staged additions and
modifications, so `git diff --cached --stat` should be checked before publishing.

## Positive Security Controls

- Runtime Postgres and Arango bootstrap scripts create scoped read-only agent users and
  verify write/DDL denial.
- `.gitignore` excludes `.env`, `.env.*`, `*.env`, and `captures/`.
- `.env.local` exists locally with mode `600`; this review did not read its contents.
- SQLite report sorting uses an allowlist in `src/pumpkinspice/reporting.py:60`.
- Most SQL values are parameterized; the main SQL concern is identifier interpolation.
- React output rendering uses normal JSX text interpolation, not `dangerouslySetInnerHTML`.
- Default web binding is loopback-only.

## Recommended Priority Order

1. Fix the web trust boundary: auth, restricted CORS, loopback enforcement, and MCP
   disable-by-default.
2. Remove arbitrary URL and absolute config path inputs from the web API.
3. Add request validation, run concurrency limits, and wall-clock timeouts.
4. Fix the pytest/TestClient dependency incompatibility and add a test timeout.
5. Harden pgvector identifier handling.
6. Add typed runtime config models and reduce `Any` at web/plugin boundaries.
7. Add capture redaction/access control and artifact ignore rules.

## Remediation Record (2026-07-02)

All accepted findings were remediated in a single hardening pass (three parallel
implementation agents plus integration), verified by the full gate suite
(ruff format, ruff check, mypy strict, pytest: 137 passed, 85% branch coverage,
frontend build clean) and a live end-to-end check against a token-configured
server instance.

| Finding | Status | Fix |
|---|---|---|
| MCP arbitrary command spawn | fixed | mutations + chat tool-spawn refused unless PUMPKINSPICE_API_TOKEN is configured or PUMPKINSPICE_ALLOW_MCP=1 (local opt-in) |
| No auth + wildcard CORS | fixed | opt-in bearer auth (PUMPKINSPICE_API_TOKEN; constant-time compare; /api/health exempt); CORS middleware only when PUMPKINSPICE_CORS_ORIGINS is set; serve refuses non-loopback bind without a token unless --insecure |
| SSRF via caller base_url | fixed | allowlist = configured backends + HeroBench + PUMPKINSPICE_EXTRA_BACKENDS; applied to chat, decoder/models, decoder/loaded, model/available, world/map; verified 400 on metadata IP |
| Absolute config paths | fixed | name regex ^[A-Za-z0-9_.-]+$ + resolve() confined to configs_dir; verified 400 on /etc/passwd and ../ traversal |
| Unbounded request fields | fixed | pydantic Field bounds: max_turns 1-1000, trials 1-50, disks 1-12, temperature 0-2, max_tokens <= 65536, messages <= 200 |
| Capture read exposure | fixed | name+suffix validation, resolve() symlink-escape rejection, 50 MiB size cap (413) |
| pgvector identifier injection | fixed | psycopg sql.Identifier composition for all identifiers + ^[A-Za-z_][A-Za-z0-9_]*$ validation at construction (ValueError) |
| RunManager thread safety | fixed | RLock over _runs/_stops/_batch_stops + snapshot list(); turn-append hot path deliberately lock-free (documented) |
| Test-suite hang | not reproduced | full suite passes in ~3s under the exact versions cited; defensive pytest-timeout (120s/test) + CI timeout-minutes: 20 added anyway |
| Artifact hygiene | fixed | .coverage unstaged and gitignored (with htmlcov/, .pytest_cache/); root-level *.png ignored |

Frontend: bearer token stored in localStorage (Settings > API token), attached
on every fetch and SSE call site; 401 responses surface a pointer to Settings.

Deliberately deferred (tracked, low urgency for a single-operator LAN tool):
typed per-plugin config models / Any-reduction at trust boundaries, and a
capture redaction mode (auth now gates capture access).

Operating modes:
- Local (default): pumpkinspice serve  -> loopback, no token, MCP mutations
  require PUMPKINSPICE_ALLOW_MCP=1.
- LAN: PUMPKINSPICE_API_TOKEN=<secret> pumpkinspice serve --host 0.0.0.0
  -> auth enforced; paste the token into Settings > API token in the UI.
  Non-loopback binds without a token are refused (override: --insecure).
