"""FastAPI app: decoder-agnostic chat playground, run launching + live turn
streaming, and a capture browser. Drives the same harness the CLI does."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import re
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.responses import Response
from starlette.types import Scope

from .. import __version__, kernel
from ..config import RunConfig, load_config
from ..reporting import RunRegistry
from .mcp_host import McpHost
from .mcp_servers import McpServer, McpServerStore
from .runs import RunManager, _stochastic_sampler
from .settings import AppSettings, SettingsStore

# OpenAI-compatible backend presets. The LMStudio endpoint moves (LAN host vs
# localhost), so it is env-configurable; default to localhost (stable) and override
# with PUMPKINSPICE_LMSTUDIO_URL for a LAN host. Any OpenAI-compatible server works.
LMSTUDIO_URL = os.environ.get("PUMPKINSPICE_LMSTUDIO_URL", "http://localhost:1234")
BACKENDS = {
    "lmstudio": LMSTUDIO_URL,
    "ollama": "http://localhost:11434",
    "vllm": "http://localhost:8000",
}
EMBED_URL = LMSTUDIO_URL
EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"
HEROBENCH_URL = "http://127.0.0.1:8000"
_MAX_TOOL_ROUNDS = 8  # cap the agentic Chat tool loop (Phase 3a)

# Conservative charset for caller-supplied config/capture names: no path
# separators, so absolute paths and traversal are impossible by construction.
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _allowed_base_urls() -> set[str]:
    """The outbound-URL allowlist: the backend presets, HeroBench, and any extra
    OpenAI-compatible hosts from PUMPKINSPICE_EXTRA_BACKENDS (comma-separated).
    Computed per call so env changes take effect without rebuilding the app."""
    extra = os.environ.get("PUMPKINSPICE_EXTRA_BACKENDS", "")
    urls = {*BACKENDS.values(), HEROBENCH_URL, LMSTUDIO_URL, *extra.split(",")}
    return {u.strip().rstrip("/") for u in urls if u.strip()}


def _check_base_url(url: str) -> str:
    """SSRF guard for caller-supplied base_url values: only the configured
    backends may be targeted, so the server cannot be steered at internal
    services (cloud metadata, DBs, ...). Returns the normalized url."""
    normalized = url.strip().rstrip("/")
    if normalized not in _allowed_base_urls():
        raise HTTPException(
            status_code=400,
            detail=f"base_url {normalized!r} not allowed: only configured backends "
            "(see /api/backends, or extend PUMPKINSPICE_EXTRA_BACKENDS) may be targeted",
        )
    return normalized


# Retrieval choices for the World composer -> (plugin name, slot config). The
# decoder model is ambient (whatever LMStudio has loaded); only retrieval x task
# vary here, with the controlled constants (DB creds, embed model) fixed.
_PG = {
    "dsn_env": "PUMPKINSPICE_PG_DSN",
    "table": "kg.belief_nodes",
    "top_k": 6,
    "embed_url": EMBED_URL,
    "embed_model": EMBED_MODEL,
}
_ARANGO = {
    "url": "http://localhost:8529",
    "database": "herobench_kg",
    "collection": "belief_nodes",
    "user_env": "ARANGO_AGENT_USER",
    "password_env": "ARANGO_AGENT_PASSWORD",
    "top_k": 6,
    "embed_url": EMBED_URL,
    "embed_model": EMBED_MODEL,
}
RETRIEVAL_OPTIONS: dict[str, tuple[str, dict[str, Any]]] = {
    "null": ("null", {}),
    "pgvector": ("pgvector", _PG),
    "pgvector+relational": ("pgvector", {**_PG, "relational": True}),
    "arango": ("arango", _ARANGO),
}


class ChatRequest(BaseModel):
    base_url: str = LMSTUDIO_URL
    model: str | None = None  # omit -> LMStudio uses the loaded model
    messages: list[dict[str, str]] = Field(max_length=200)
    sampler: dict[str, Any] = Field(default_factory=dict)
    # 0 (or omitted) -> no output cap, generate until EOS
    max_tokens: int = Field(default=0, ge=0, le=65536)


class RunRequest(BaseModel):
    # Either a saved config name, or inline composer options (retrieval + task).
    config: str | None = None
    retrieval: str | None = None
    task: str | None = None
    prompt: str = "default"  # prompt strategy: default (reactive) | plan (Stage 2)
    max_turns: int = Field(default=50, ge=1, le=1000)
    # Stop-on-goal spec: an item code ("sticky_sword") stops on craft; "level>=N" or
    # "<skill>_level>=N" (e.g. "weaponcrafting_level>=5") stops on reaching that level.
    goal_item: str | None = None
    # set -> reproduce a stochastic trial at this seed (else greedy)
    seed: int | None = Field(default=None, ge=0)
    temperature: float = Field(default=0.7, ge=0, le=2)  # only used when seed is set


class TrialsRequest(BaseModel):
    # N stochastic trials of one config (composer options), reset between each.
    retrieval: str
    task: str
    prompt: str = "default"
    max_turns: int = Field(default=100, ge=1, le=1000)
    goal_item: str | None = None  # goal spec: item code, "level>=N", or "<skill>_level>=N"
    trials: int = Field(default=10, ge=1, le=50)
    temperature: float = Field(default=0.7, ge=0, le=2)
    character: str = (
        "character_1"  # per-batch character (use character_2.. to run a 2nd batch in parallel)
    )
    model: str | None = None  # per-batch model override (else the global model-under-test)


class HanoiRunRequest(BaseModel):
    # Single Hanoi run: no retrieval/task composer needed (rules are universal,
    # goal is always "all disks on C") -- just the strategy + puzzle size.
    prompt: str = "default"
    max_turns: int = Field(default=100, ge=1, le=1000)
    disks: int = Field(default=4, ge=1, le=12)
    seed: int | None = Field(default=None, ge=0)
    temperature: float = Field(default=0.7, ge=0, le=2)


class HanoiTrialsRequest(BaseModel):
    prompt: str = "default"
    max_turns: int = Field(default=100, ge=1, le=1000)
    disks: int = Field(default=4, ge=1, le=12)
    trials: int = Field(default=10, ge=1, le=50)
    temperature: float = Field(default=0.7, ge=0, le=2)
    model: str | None = None  # per-batch model override (else the global model-under-test)


class RunUpdate(BaseModel):
    # Operator annotations on a finished run (the Reports tab).
    label: str | None = None
    tags: list[str] | None = None
    notes: str | None = None


class ModelSettingsUpdate(BaseModel):
    # Partial update of the model under test + decode defaults (None -> unchanged).
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    history_window: int | None = None


class ModelLoad(BaseModel):
    model: str


class McpServerBody(BaseModel):
    # An MCP server entry for the in-UI manager (stdio launch).
    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    enabled: bool = True


class McpEnabled(BaseModel):
    enabled: bool


def _sse(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj)}\n\n"


_GOAL_LEVEL_RE = re.compile(r"^(?:([a-z_]+)_)?level>=(\d+)$")


def _apply_goal(run: dict[str, Any], goal: str | None) -> None:
    """Translate a goal spec string into run-config goal keys. An item code
    ("sticky_sword") stops the run on craft; "level>=N" stops on character level
    N; "<skill>_level>=N" (e.g. "weaponcrafting_level>=5") stops on that SKILL
    reaching N. Blank/None -> run the full turn budget."""
    goal = (goal or "").strip()
    if not goal:
        return
    m = _GOAL_LEVEL_RE.match(goal)
    if m:
        run["goal_level"] = int(m.group(2))
        if m.group(1):
            run["goal_skill"] = m.group(1)
    else:
        run["goal_item"] = goal


def _inline_config(
    retrieval: str,
    task: str,
    max_turns: int,
    prompt: str = "default",
    settings: AppSettings | None = None,
    goal_item: str | None = None,
    character: str = "character_1",
    model: str | None = None,
) -> RunConfig:
    """Build a run from World-composer options. The model under test + decode
    defaults come from app settings (Phase 2); empty/zero means 'use the loaded
    model' / unbounded / full history. goal_item is a goal SPEC (see _apply_goal):
    item code, "level>=N", or "<skill>_level>=N". character + model are per-batch
    overrides for parallel sweeps (a second model on a second GPU playing a second
    character so the two don't reset-war)."""
    plugin, rconf = RETRIEVAL_OPTIONS[retrieval]
    decoder: dict[str, Any] = {"base_url": LMSTUDIO_URL}
    run: dict[str, Any] = {
        "decoder": "lmstudio",
        "retrieval": plugin,
        "world": "herobench",
        "prompt": prompt,
        "capture": "jsonl",
        "task": task,
        "max_turns": max_turns,
    }
    _apply_goal(run, goal_item)
    if settings is not None:
        if settings.model:
            decoder["model"] = settings.model
        if settings.max_tokens > 0:
            decoder["max_tokens"] = settings.max_tokens
        if settings.temperature:
            decoder["sampler"] = {"temperature": settings.temperature}
        run["history_window"] = settings.history_window
    if model:  # per-batch model override beats the global model-under-test
        decoder["model"] = model
    return RunConfig(
        run=run,
        slots={
            "decoder": decoder,
            "retrieval": dict(rconf),
            "world": {"base_url": HEROBENCH_URL, "character": character},
            "prompt": {},
            "capture": {},
        },
    )


HANOI_GRAMMAR = """\
Actions and their args:
  move: {"from": "<peg>", "to": "<peg>"}   -- move the top disk of peg <from> onto peg
        <to>. Pegs are named A, B, C. You may only move the TOP disk of a peg (the
        last one placed there). You may NEVER place a larger disk on top of a
        smaller disk."""

# One Hanoi-flavored system prompt per prompt-strategy shape (mirrors the
# structure of each strategy's own HeroBench SYSTEM_* constant exactly, with
# HANOI_GRAMMAR substituted for ACTION_GRAMMAR): the `system` config key every
# builder already supports (see prompt_default/_plan/_replan/_executor) lets
# Hanoi reuse all four strategies unmodified -- only the domain text differs.
HANOI_SYSTEMS: dict[str, str] = {
    "default": f"""\
You are a capable agent solving the Tower of Hanoi.

{HANOI_GRAMMAR}

Reason every turn in THREE labeled steps, then act:

Reflect: Look at your most recent action in "Recent actions". Did it succeed? If it
  FAILED, diagnose WHY (moved a nonexistent disk, tried to place a larger disk on a
  smaller one, or named an unknown peg) and say what you will change. If there is no
  prior action, write "first turn, no prior action".
Thought: Using the Goal and the World state (pegs, disk sizes, moves so far), decide
  the single best LEGAL move toward solving the puzzle. Do not repeat a move that
  just failed for the same reason.
Action: output exactly ONE action as a JSON object on its own line, then STOP:
  {{"action": "move", "args": {{"from": "<peg>", "to": "<peg>"}}}}
""",
    "plan": f"""\
You are a capable agent solving the Tower of Hanoi.

{HANOI_GRAMMAR}

On the FIRST turn, write a PLAN: a numbered list of concrete moves (each "move disk
from X to Y") that moves the entire stack from peg A to peg C.

On EVERY turn, reason in THREE labeled steps, then act:

Reflect: Look at your most recent action and where you are in your committed plan.
  Did the last action succeed? If it FAILED, diagnose why and state which plan step
  you are on. (First turn: "first turn".)
Thought: Pick the single best LEGAL action toward the CURRENT plan step. Do not
  repeat an action that just failed.
Action: output exactly ONE action as a JSON object on its own line, then STOP:
  {{"action": "move", "args": {{"from": "<peg>", "to": "<peg>"}}}}

Do NOT rewrite the plan once committed -- follow it.
""",
    "replan": f"""\
You are a capable agent solving the Tower of Hanoi.

{HANOI_GRAMMAR}

On the FIRST turn, write a PLAN: a numbered list of concrete moves that moves the
entire stack from peg A to peg C.

Your plan is a LIVING plan. On EVERY turn, reason in three labeled steps, then act:

Reflect: Look at your most recent action and where you are in your plan. Did it
  succeed? If it FAILED, the world has contradicted your plan -- diagnose why and
  DECIDE: keep the plan, or revise it. (First turn: "first turn".)
Thought: Pick the single best LEGAL action toward your current step. Do not repeat
  an action that just failed for the same reason.
Action: output exactly ONE action as a JSON object on its own line, then STOP:
  {{"action": "move", "args": {{"from": "<peg>", "to": "<peg>"}}}}

To REVISE the plan, write an updated "## Plan" (numbered steps) BEFORE your Reflect.
Revise only when the world has actually contradicted the plan -- do not churn it.
""",
    "executor": f"""\
You are a capable agent solving the Tower of Hanoi.

{HANOI_GRAMMAR}

You work with a PLAN EXECUTOR: you write your plan ONCE as a JSON object; the
harness stores it, tracks the current step, and advances a step automatically the
moment its done_when condition holds in the world state. Each turn you are shown
ONLY the current step -- act toward THAT step; do not re-derive the plan.

Plan format (a single JSON object on its own line):
{{"plan": [{{"step": 1, "description": "<one concrete objective>", "done_when": {{...}}}}, ...]}}

Every step's done_when must use this form:
  {{"state": {{"<key>": <expected_value>}}}}   -- e.g. {{"state": {{"solved": true}}}}
    for the final step (all disks moved to peg C). You may also check partial
    progress against the "pegs" field of the World state shown to you each turn,
    e.g. {{"state": {{"pegs": {{"A": [], "B": [], "C": [4, 3, 2, 1]}}}}}}.

Each turn, reason briefly, then act:
Reflect: did your last action succeed? If it FAILED, diagnose why (no disk on that
  peg, or tried to place a larger disk on a smaller one).
Action: output exactly ONE action as a JSON object on its own line, then STOP:
  {{"action": "move", "args": {{"from": "<peg>", "to": "<peg>"}}}}
If the CURRENT step is already complete but was not auto-advanced, add
"step_done": true to your action JSON and act toward the NEXT step instead.
Rewrite the plan (a new {{"plan": [...]}} line before your action) ONLY when the
world has genuinely contradicted it.
""",
}


def _inline_hanoi_config(
    prompt: str,
    max_turns: int,
    disks: int,
    settings: AppSettings | None = None,
    model: str | None = None,
) -> RunConfig:
    """Build a Hanoi run: no retrieval (the rules are universal, so they belong in
    the system prompt, not a corpus -- see world_hanoi.py), the goal is always "all
    disks on peg C" (HanoiWorld's self-reported "solved" state), and a fresh
    HanoiWorld instance per run IS the reset (no external reset call needed)."""
    task = f"Move the entire {disks}-disk stack from peg A to peg C."
    decoder: dict[str, Any] = {"base_url": LMSTUDIO_URL}
    run: dict[str, Any] = {
        "decoder": "lmstudio",
        "retrieval": "null",
        "world": "hanoi",
        "prompt": prompt,
        "capture": "jsonl",
        "task": task,
        "max_turns": max_turns,
        "goal_state_key": "solved",
    }
    if settings is not None:
        if settings.model:
            decoder["model"] = settings.model
        if settings.max_tokens > 0:
            decoder["max_tokens"] = settings.max_tokens
        if settings.temperature:
            decoder["sampler"] = {"temperature": settings.temperature}
        run["history_window"] = settings.history_window
    if model:
        decoder["model"] = model
    return RunConfig(
        run=run,
        slots={
            "decoder": decoder,
            "retrieval": {},
            "world": {"disks": disks},
            "prompt": {"system": HANOI_SYSTEMS.get(prompt, HANOI_SYSTEMS["default"])},
            "capture": {},
        },
    )


class _SpaStaticFiles(StaticFiles):
    """Serve the SPA but never let the browser cache index.html, so a rebuilt
    bundle is picked up on a plain refresh. Hashed assets are immutable."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        ctype = response.headers.get("content-type", "")
        if ctype.startswith("text/html"):
            response.headers["Cache-Control"] = "no-cache"
        elif path.startswith("assets/"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


def create_app(
    *,
    configs_dir: Path | None = None,
    captures_dir: Path | None = None,
    api_token: str | None = None,
) -> FastAPI:
    repo_root = Path(__file__).resolve().parents[3]
    configs_dir = configs_dir or repo_root / "configs"
    captures_dir = captures_dir or repo_root / "captures"
    registry = RunRegistry(captures_dir / "results.db")
    runs = RunManager(captures_dir, registry)
    settings = SettingsStore(captures_dir / "settings.json")
    mcp_store = McpServerStore(captures_dir / "mcp_servers.json")
    # Opt-in bearer auth: no token (arg or env) -> open, for loopback-only use.
    token = api_token if api_token is not None else os.environ.get("PUMPKINSPICE_API_TOKEN")
    # Registering an MCP server = running arbitrary commands on this host, so it
    # is gated behind auth (or an explicit local-only opt-in for tokenless use).
    mcp_allowed = bool(token) or os.environ.get("PUMPKINSPICE_ALLOW_MCP") == "1"

    app = FastAPI(title="PumpkinSpice", version=__version__)

    if token:
        expected = f"Bearer {token}".encode()

        @app.middleware("http")
        async def require_api_token(
            request: Request, call_next: Callable[[Request], Awaitable[Response]]
        ) -> Response:
            # Every /api/ route except the health probe needs the bearer token.
            # Static SPA files stay open (they hold no secrets and take no actions).
            path = request.url.path
            if path.startswith("/api/") and path != "/api/health":
                supplied = request.headers.get("authorization", "").encode()
                if not secrets.compare_digest(supplied, expected):
                    return JSONResponse(
                        status_code=401, content={"detail": "missing or invalid API token"}
                    )
            return await call_next(request)

    # The SPA is served same-origin (mounted below), so no CORS is needed by
    # default; the frontend dev server proxies /api. Cross-origin use is opt-in.
    cors_origins = [
        o.strip() for o in os.environ.get("PUMPKINSPICE_CORS_ORIGINS", "").split(",") if o.strip()
    ]
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _require_mcp() -> None:
        # See mcp_allowed above: MCP mutations are command registration.
        if not mcp_allowed:
            raise HTTPException(
                status_code=403,
                detail="MCP command registration requires PUMPKINSPICE_API_TOKEN "
                "(or PUMPKINSPICE_ALLOW_MCP=1 for local-only use)",
            )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/backends")
    def backends() -> list[dict[str, str]]:
        return [{"name": k, "base_url": v} for k, v in BACKENDS.items()]

    @app.get("/api/plugins")
    def plugins() -> dict[str, list[str]]:
        return kernel.discover()

    @app.get("/api/decoder/models")
    async def decoder_models(base_url: str) -> list[str]:
        base_url = _check_base_url(base_url)
        try:
            async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
                resp = await client.get("/v1/models")
                resp.raise_for_status()
                return [m["id"] for m in resp.json().get("data", [])]
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"decoder unreachable: {exc}") from exc

    @app.get("/api/decoder/loaded")
    async def decoder_loaded(base_url: str = LMSTUDIO_URL) -> list[dict[str, Any]]:
        # The model(s) LMStudio currently has loaded (state=loaded), LLMs only.
        # PumpkinSpice polls this; it does not load/unload models. [] if backend
        # is down or lacks the native endpoint (e.g. Ollama/vLLM).
        base_url = _check_base_url(base_url)
        try:
            async with httpx.AsyncClient(base_url=base_url, timeout=8.0) as client:
                resp = await client.get("/api/v0/models")
                resp.raise_for_status()
                data = resp.json().get("data", [])
        except httpx.HTTPError:
            return []
        loaded: list[dict[str, Any]] = []
        for m in data:
            mid = str(m.get("id", ""))
            if (
                m.get("state") != "loaded"
                or m.get("type") == "embeddings"
                or "embed" in mid.lower()
            ):
                continue
            loaded.append(
                {k: m.get(k) for k in ("id", "arch", "quantization", "loaded_context_length")}
            )
        return loaded

    # --- Model under test + decode defaults (Phase 2) ----------------------------
    @app.get("/api/model/available")
    async def model_available(base_url: str = LMSTUDIO_URL) -> list[str]:
        # All LLM ids LMStudio knows about (loaded or not), embeddings excluded.
        base_url = _check_base_url(base_url)
        try:
            async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
                resp = await client.get("/v1/models")
                resp.raise_for_status()
                ids = [str(m.get("id", "")) for m in resp.json().get("data", [])]
        except httpx.HTTPError:
            return []
        return [m for m in ids if m and "embed" not in m.lower()]

    @app.get("/api/model/settings")
    def model_settings() -> dict[str, Any]:
        return dataclasses.asdict(settings.get())

    @app.post("/api/model/settings")
    def model_settings_update(body: ModelSettingsUpdate) -> dict[str, Any]:
        return dataclasses.asdict(
            settings.update(
                model=body.model,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
                history_window=body.history_window,
            )
        )

    @app.post("/api/model/load")
    async def model_load(body: ModelLoad) -> dict[str, Any]:
        # Warm-load a model into LMStudio via a tiny request (LMStudio JIT-loads on
        # use). Returns ok/error; the badge then reflects the newly loaded model.
        payload = {
            "model": body.model,
            "messages": [{"role": "user", "content": "ok"}],
            "max_tokens": 1,
        }
        try:
            async with httpx.AsyncClient(
                base_url=LMSTUDIO_URL.rstrip("/"), timeout=300.0
            ) as client:
                resp = await client.post("/v1/chat/completions", json=payload)
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502, detail=f"could not load {body.model}: {exc}"
            ) from exc
        return {"loaded": body.model}

    # --- MCP server manager (Phase 3a; Chat-only) --------------------------------
    @app.get("/api/mcp/servers")
    def mcp_servers() -> list[dict[str, Any]]:
        return [dataclasses.asdict(s) for s in mcp_store.all()]

    @app.post("/api/mcp/servers")
    def mcp_upsert(body: McpServerBody) -> dict[str, Any]:
        _require_mcp()
        s = mcp_store.upsert(
            McpServer(name=body.name, command=body.command, args=body.args, enabled=body.enabled)
        )
        return dataclasses.asdict(s)

    @app.post("/api/mcp/servers/{name}/enabled")
    def mcp_set_enabled(name: str, body: McpEnabled) -> dict[str, bool]:
        _require_mcp()
        if not mcp_store.set_enabled(name, body.enabled):
            raise HTTPException(status_code=404, detail="server not found")
        return {"enabled": body.enabled}

    @app.delete("/api/mcp/servers/{name}")
    def mcp_delete(name: str) -> dict[str, bool]:
        _require_mcp()
        if not mcp_store.delete(name):
            raise HTTPException(status_code=404, detail="server not found")
        return {"deleted": True}

    @app.get("/api/retrieval-options")
    def retrieval_options() -> list[str]:
        return list(RETRIEVAL_OPTIONS)

    @app.get("/api/prompt-options")
    def prompt_options() -> list[str]:
        # Prompt strategies = the experiment stages: 'default' (reactive, Stage 1),
        # 'plan' (commit-and-execute, Stage 2). Discovered, so new ones show up.
        return kernel.discover().get("prompt", [])

    @app.get("/api/world/map")
    async def world_map(base_url: str = HEROBENCH_URL) -> list[dict[str, Any]]:
        # Proxy HeroBench's static tile map for the World viewer.
        base_url = _check_base_url(base_url)
        try:
            async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
                resp = await client.get("/maps")
                resp.raise_for_status()
                raw = resp.json()
            tiles = raw.get("data", raw) if isinstance(raw, dict) else raw
            return [
                {"x": t["x"], "y": t["y"], "name": t.get("name"), "content": t.get("content")}
                for t in tiles
            ]
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"herobench unreachable: {exc}") from exc

    def _chat_base(req: ChatRequest, cfg: AppSettings) -> dict[str, Any]:
        """Shared LMStudio payload bits: decode defaults + model under test."""
        payload: dict[str, Any] = {**req.sampler}
        if cfg.temperature and "temperature" not in payload:
            payload["temperature"] = cfg.temperature
        cap = req.max_tokens or cfg.max_tokens
        if cap > 0:
            payload["max_tokens"] = cap
        model = req.model or cfg.model
        if model:
            payload["model"] = model
        return payload

    async def _chat_with_tools(req: ChatRequest, cfg: AppSettings) -> AsyncIterator[str]:
        """Agentic loop: expose the enabled MCP servers' tools, execute tool_calls,
        loop until the model answers. Surfaces tool activity as SSE events."""
        base = _chat_base(req, cfg)
        servers = mcp_store.all()
        messages: list[dict[str, Any]] = [dict(m) for m in req.messages]
        try:
            async with (
                McpHost(servers) as host,
                httpx.AsyncClient(base_url=req.base_url.rstrip("/"), timeout=600.0) as client,
            ):
                for _ in range(_MAX_TOOL_ROUNDS):
                    resp = await client.post(
                        "/v1/chat/completions",
                        json={**base, "messages": messages, "tools": host.tools},
                    )
                    if resp.status_code != 200:
                        yield _sse({"error": f"HTTP {resp.status_code}", "detail": resp.text[:500]})
                        return
                    msg = resp.json()["choices"][0]["message"]
                    tool_calls = msg.get("tool_calls")
                    if msg.get("reasoning_content"):
                        yield _sse({"reasoning": msg["reasoning_content"]})
                    if not tool_calls:
                        if msg.get("content"):
                            yield _sse({"delta": msg["content"]})
                        break
                    messages.append(
                        {
                            "role": "assistant",
                            "content": msg.get("content") or "",
                            "tool_calls": tool_calls,
                        }
                    )
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        name = fn.get("name", "")
                        try:
                            args = json.loads(fn.get("arguments") or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        yield _sse({"tool_call": {"name": name, "args": args}})
                        result = await host.call(name, args)
                        yield _sse({"tool_result": {"name": name, "result": result[:4000]}})
                        messages.append(
                            {"role": "tool", "tool_call_id": tc.get("id", ""), "content": result}
                        )
                else:
                    yield _sse({"delta": "\n[stopped: too many tool rounds]"})
                yield _sse({"done": True})
        except Exception as exc:  # spawn/connection/tool failures -> a clean error event
            yield _sse({"error": str(exc)})

    @app.post("/api/chat")
    async def chat(req: ChatRequest) -> StreamingResponse:
        req.base_url = _check_base_url(req.base_url)  # before any outbound request
        cfg = settings.get()
        # When MCP servers are enabled (and MCP is allowed at all -- see
        # _require_mcp), run the agentic tool-calling loop; otherwise keep the
        # plain token-streaming path (faster, no subprocess spawn).
        if mcp_allowed and any(s.enabled for s in mcp_store.all()):
            return StreamingResponse(_chat_with_tools(req, cfg), media_type="text/event-stream")

        async def gen() -> AsyncIterator[str]:
            payload: dict[str, Any] = {
                "messages": req.messages,
                "stream": True,
                **_chat_base(req, cfg),
            }
            try:
                async with (
                    httpx.AsyncClient(base_url=req.base_url.rstrip("/"), timeout=180.0) as client,
                    client.stream("POST", "/v1/chat/completions", json=payload) as resp,
                ):
                    if resp.status_code != 200:
                        body = (await resp.aread()).decode(errors="replace")
                        yield _sse({"error": f"HTTP {resp.status_code}", "detail": body[:500]})
                        return
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:") :].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        delta = chunk["choices"][0].get("delta", {})
                        # Reasoning models (Qwen3, etc.) stream chain-of-thought in
                        # `reasoning_content`; the final answer is in `content`.
                        if delta.get("content"):
                            yield _sse({"delta": delta["content"]})
                        if delta.get("reasoning_content"):
                            yield _sse({"reasoning": delta["reasoning_content"]})
                yield _sse({"done": True})
            except httpx.HTTPError as exc:
                yield _sse({"error": str(exc)})

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/api/configs")
    def list_configs() -> list[dict[str, Any]]:
        import tomllib

        out: list[dict[str, Any]] = []
        for path in sorted(configs_dir.glob("*.toml")):
            try:
                raw = tomllib.loads(path.read_text())
            except (OSError, tomllib.TOMLDecodeError):
                continue
            run = raw.get("run", {})
            out.append(
                {
                    "name": path.stem,
                    "task": run.get("task", ""),
                    "retrieval": run.get("retrieval"),
                    "world": run.get("world"),
                    "max_turns": run.get("max_turns"),
                }
            )
        return out

    @app.post("/api/runs")
    def start_run(req: RunRequest) -> dict[str, Any]:
        if req.config:
            # Confine config selection to bare names resolving inside configs_dir:
            # no separators (so no absolute paths or traversal), then re-checked
            # after resolution. Never echo server paths back, only the name.
            if not _SAFE_NAME_RE.match(req.config):
                raise HTTPException(status_code=400, detail="invalid config name")
            path = (configs_dir / f"{req.config}.toml").resolve()
            if not path.is_relative_to(configs_dir.resolve()) or not path.exists():
                raise HTTPException(status_code=404, detail=f"config not found: {req.config}")
            cfg, label = load_config(path), req.config
        elif req.retrieval and req.task is not None:
            if req.retrieval not in RETRIEVAL_OPTIONS:
                raise HTTPException(status_code=400, detail=f"unknown retrieval {req.retrieval!r}")
            if req.prompt not in kernel.discover().get("prompt", []):
                raise HTTPException(status_code=400, detail=f"unknown prompt {req.prompt!r}")
            cfg = _inline_config(
                req.retrieval, req.task, req.max_turns, req.prompt, settings.get(), req.goal_item
            )
            # Label carries the stage so Stage-1 vs Stage-2 captures are distinguishable.
            label = f"{req.retrieval}-{req.prompt}"
        else:
            raise HTTPException(status_code=400, detail="provide `config`, or `retrieval` + `task`")
        # A seed reproduces a stochastic trial (same sampler as the trials harness); tag
        # it seed/temp so it is as findable + re-runnable as a batch trial. Blank = greedy.
        tags: list[str] = []
        if req.seed is not None and not req.config:
            cfg.slots["decoder"]["sampler"] = _stochastic_sampler(req.temperature, req.seed)
            tags = [f"seed:{req.seed}", f"temp:{req.temperature}"]
        try:
            record = runs.start(cfg, label, tags=tags)
        except Exception as exc:
            # Plugin construction failed (e.g. missing scoped DB creds) -- surface a
            # clean message instead of a 500.
            raise HTTPException(status_code=400, detail=f"could not start run: {exc}") from exc
        return {"id": record.id, "config_name": record.config_name, "status": record.status}

    @app.post("/api/runs/{run_id}/stop")
    def stop_run(run_id: str) -> dict[str, bool]:
        # Cooperative stop: the loop ends after its current turn finishes.
        if not runs.stop(run_id):
            raise HTTPException(status_code=404, detail="run not found")
        return {"stopping": True}

    @app.post("/api/trials")
    def start_trials(req: TrialsRequest) -> dict[str, Any]:
        # N stochastic trials with a fresh reset between each, all tagged batch:<id>.
        if req.retrieval not in RETRIEVAL_OPTIONS:
            raise HTTPException(status_code=400, detail=f"unknown retrieval {req.retrieval!r}")
        if req.prompt not in kernel.discover().get("prompt", []):
            raise HTTPException(status_code=400, detail=f"unknown prompt {req.prompt!r}")
        n = max(1, min(50, req.trials))  # bound the batch size
        cfg = _inline_config(
            req.retrieval,
            req.task,
            req.max_turns,
            req.prompt,
            settings.get(),
            req.goal_item,
            req.character,
            req.model,
        )
        label = f"{req.model or settings.get().model or 'mut'}-{req.prompt}-{req.character}"
        try:
            batch_id = runs.start_trials(cfg, label, n, req.temperature)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"could not start trials: {exc}") from exc
        return {"batch": batch_id, "trials": n}

    @app.post("/api/trials/{batch_id}/stop")
    def stop_batch(batch_id: str) -> dict[str, bool]:
        # Abort the whole batch: no further trials start; the in-flight one ends cleanly.
        if not runs.stop_batch(batch_id):
            raise HTTPException(status_code=404, detail="batch not found")
        return {"stopping": True}

    # --- Hanoi: a second, vocabulary-disjoint benchmark playground -----------------
    # Reuses runs.start/start_trials, SSE streaming, stop, batch-stop, and Reports
    # verbatim (all generic over any World) -- only the config builder differs.
    @app.post("/api/hanoi/runs")
    def start_hanoi_run(req: HanoiRunRequest) -> dict[str, Any]:
        if req.prompt not in kernel.discover().get("prompt", []):
            raise HTTPException(status_code=400, detail=f"unknown prompt {req.prompt!r}")
        cfg = _inline_hanoi_config(req.prompt, req.max_turns, req.disks, settings.get())
        tags: list[str] = []
        if req.seed is not None:
            cfg.slots["decoder"]["sampler"] = _stochastic_sampler(req.temperature, req.seed)
            tags = [f"seed:{req.seed}", f"temp:{req.temperature}"]
        label = f"hanoi{req.disks}-{req.prompt}"
        try:
            record = runs.start(cfg, label, tags=tags)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"could not start run: {exc}") from exc
        return {"id": record.id, "config_name": record.config_name, "status": record.status}

    @app.post("/api/hanoi/trials")
    def start_hanoi_trials(req: HanoiTrialsRequest) -> dict[str, Any]:
        if req.prompt not in kernel.discover().get("prompt", []):
            raise HTTPException(status_code=400, detail=f"unknown prompt {req.prompt!r}")
        n = max(1, min(50, req.trials))
        cfg = _inline_hanoi_config(req.prompt, req.max_turns, req.disks, settings.get(), req.model)
        label = f"{req.model or settings.get().model or 'mut'}-hanoi{req.disks}-{req.prompt}"
        try:
            batch_id = runs.start_trials(cfg, label, n, req.temperature)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"could not start trials: {exc}") from exc
        return {"batch": batch_id, "trials": n}

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        return [
            {
                "id": r.id,
                "config_name": r.config_name,
                "status": r.status,
                "turns": len(r.turns),
                "task": r.task,
                "tags": r.tags,
            }
            for r in runs.list()
        ]

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        record = runs.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        return {
            "id": record.id,
            "config_name": record.config_name,
            "task": record.task,
            "plugins": record.plugins,
            "status": record.status,
            "error": record.error,
            "turns": record.turns,
        }

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str) -> StreamingResponse:
        record = runs.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")

        async def gen() -> AsyncIterator[str]:
            sent = 0
            while True:
                while sent < len(record.turns):
                    yield _sse({"event": "turn", "turn": record.turns[sent]})
                    sent += 1
                if record.status != "running":
                    yield _sse({"event": "end", "status": record.status, "error": record.error})
                    return
                # Heartbeat: turns can take 30-70s to decode (growing prompt); keep
                # the connection alive and let the client show it is still live.
                yield ": keepalive\n\n"
                await asyncio.sleep(2.0)

        return StreamingResponse(gen(), media_type="text/event-stream")

    # --- Reports (the persistent run registry) -----------------------------------
    @app.get("/api/reports/runs")
    def reports_runs(
        benchmark: str | None = None,
        model: str | None = None,
        strategy: str | None = None,
        tag: str | None = None,
        sort: str = "finished_at",
    ) -> list[dict[str, Any]]:
        return registry.list_runs(
            benchmark=benchmark, model=model, strategy=strategy, tag=tag, sort=sort
        )

    @app.get("/api/reports/runs/{run_id}")
    def reports_run(run_id: str) -> dict[str, Any]:
        rec = registry.get(run_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="run not found")
        return rec

    @app.post("/api/reports/runs/{run_id}")
    def reports_update(run_id: str, body: RunUpdate) -> dict[str, Any]:
        ok = registry.update(run_id, label=body.label, tags=body.tags, notes=body.notes)
        if not ok:
            raise HTTPException(status_code=404, detail="run not found or nothing to update")
        return registry.get(run_id) or {}

    @app.get("/api/reports/leaderboard")
    def reports_leaderboard(benchmark: str | None = None) -> list[dict[str, Any]]:
        return registry.leaderboard(benchmark=benchmark)

    @app.get("/api/captures")
    def list_captures() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if captures_dir.exists():
            for path in sorted(captures_dir.glob("*.jsonl"), reverse=True):
                lines = path.read_text().splitlines() if path.stat().st_size else []
                out.append({"name": path.name, "turns": len(lines)})
        return out

    @app.get("/api/captures/{name}")
    def get_capture(name: str) -> list[dict[str, Any]]:
        # Confine reads to real .jsonl files directly inside captures_dir: a
        # conservative name (no separators), then resolve() so a symlink cannot
        # escape the directory, then a size cap before slurping the file.
        if not name.endswith(".jsonl") or not _SAFE_NAME_RE.match(name):
            raise HTTPException(status_code=404, detail="capture not found")
        path = (captures_dir / name).resolve()
        if path.parent != captures_dir.resolve() or not path.is_file():
            raise HTTPException(status_code=404, detail="capture not found")
        if path.stat().st_size > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="capture too large")
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    # Serve the built SPA (frontend/dist) if present, so `pumpkinspice serve`
    # hosts the whole app on one port. API routes above take precedence.
    dist = repo_root / "frontend" / "dist"
    if dist.exists():
        app.mount("/", _SpaStaticFiles(directory=dist, html=True), name="frontend")

    return app
