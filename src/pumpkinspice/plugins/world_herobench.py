"""HeroBench REST client (spec section 3; the world + scoring seam).

HeroBench is a faithful ArtifactsMMO-style API:
  state:   GET  /characters/{name}
  actions: POST /my/{name}/action/{verb}
    move      -> {"x": int, "y": int}
    fight     -> {}            (or /action/fight/{quantity})
    gathering -> {"quantity": int}
    crafting  -> {"code": str, "quantity": int}
    equip     -> {"slot": str, "code": str, "quantity"?: int}
    rest      -> {}

Verb aliases (gather->gathering, craft->crafting) are normalized so the prompt
can use the short names from the spec. Default base URL http://127.0.0.1:8000.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from ..contracts import Action, ActionResult, WorldState

DEFAULT_BASE_URL = "http://127.0.0.1:8000"

# Map agent verbs -> (HeroBench action path segment, list of expected body keys).
_VERB_ALIASES = {"gather": "gathering", "craft": "crafting"}

# Endpoints with a single scalar Body param expect the bare value as the JSON
# body (e.g. `1`), not an object. FastAPI only embeds (`{"name": value}`) when
# there are multiple body params (move's x/y, craft's code/quantity). Maps the
# action verb to the single arg whose bare value is the body.
_SCALAR_BODY = {"gathering": "quantity"}


def _fail_reason(status: int, body: dict[str, Any]) -> str:
    """A concise, agent-useful failure reason from HeroBench's error body, so the
    CoT can Reflect ('go to (1,5)', 'missing 37 copper_ore') instead of a bare status
    code. HeroBench nests the detail as body["error"]["message"] = an info dict with
    errors/workshop/skill_level/missing_items (craft failures come back as 500);
    unwrap it rather than dropping it on the floor."""
    info: Any = body
    err = body.get("error")
    if isinstance(err, dict):
        msg = err.get("message")
        info = msg if isinstance(msg, dict) else err
    if isinstance(info, dict):
        errs = info.get("errors")
        if isinstance(errs, dict) and errs.get("on_workshop_tile") is False:
            ws = info.get("workshop") or {}
            return (
                f"HTTP {status}: wrong tile -- go to the workshop at "
                f"{ws.get('needed')} (you are at {ws.get('current')})"
            )
        if isinstance(errs, dict) and errs.get("needed_skill_level") is False:
            sk = info.get("skill_level") or {}
            return (
                f"HTTP {status}: {sk.get('skill')} skill too low "
                f"(need {sk.get('needed')}, have {sk.get('current')})"
            )
        if info.get("missing_items"):
            return f"HTTP {status}: missing items {info['missing_items']}"
    if isinstance(err, dict) and isinstance(err.get("message"), str):
        return f"HTTP {status}: {err['message']}"
    return f"HTTP {status}"


class HeroBenchWorld:
    name = "herobench"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.base_url = str(config.get("base_url", DEFAULT_BASE_URL)).rstrip("/")
        self.character = config.get("character", "hero")
        self.timeout = float(config.get("timeout", 60.0))
        transport = httpx.HTTPTransport(retries=int(config.get("retries", 2)))
        self._client = httpx.Client(
            base_url=self.base_url, timeout=self.timeout, transport=transport
        )

    def get_state(self) -> WorldState:
        """Fetch the character state. One transient failure is retried once (a
        blip must not abort a long run); a persistent failure raises with
        context -- there is nothing sensible to play without world state."""
        last: Exception | None = None
        for attempt in (0, 1):
            try:
                resp = self._client.get(f"/characters/{self.character}")
                resp.raise_for_status()
                return WorldState(raw=resp.json(), source=self.name)
            except httpx.HTTPError as exc:
                last = exc
                if attempt == 0:
                    time.sleep(0.5)
        raise RuntimeError(
            f"HeroBench get_state failed for {self.character!r} after retry: {last}"
        ) from last

    def act(self, action: Action) -> ActionResult:
        verb = _VERB_ALIASES.get(action.kind, action.kind)
        json_body: Any
        if verb == "fight" and "quantity" in action.args:
            # fight supports a /{quantity} path variant (no body)
            path = f"/my/{self.character}/action/fight/{int(action.args['quantity'])}"
            json_body = None
        elif verb in _SCALAR_BODY:
            # single scalar Body param -> send the bare value, not an object
            path = f"/my/{self.character}/action/{verb}"
            json_body = action.args.get(_SCALAR_BODY[verb], 1)
        else:
            path = f"/my/{self.character}/action/{verb}"
            json_body = dict(action.args) or None
        try:
            resp = self._client.post(path, json=json_body)
        except httpx.HTTPError as exc:
            return ActionResult(ok=False, status_code=0, error=str(exc))
        ok = resp.is_success
        try:
            data = resp.json()
        except ValueError:
            data = {"text": resp.text}
        if not isinstance(data, dict):
            data = {"data": data}
        return ActionResult(
            ok=ok,
            status_code=resp.status_code,
            data=data,
            error=None if ok else _fail_reason(resp.status_code, data),
        )
