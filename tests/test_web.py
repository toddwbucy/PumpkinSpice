"""Web backend: non-network endpoints + an offline run lifecycle.

Network endpoints (chat stream, decoder models, SSE) are integration-tested live.
Skipped unless the web extra is installed (CI installs it)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from pumpkinspice.web.app import _inline_config, create_app

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def _client(captures_dir: Path) -> TestClient:
    return TestClient(create_app(configs_dir=CONFIGS, captures_dir=captures_dir))


def test_health_backends_plugins(tmp_path: Path) -> None:
    c = _client(tmp_path)
    assert c.get("/api/health").json()["status"] == "ok"
    names = {b["name"] for b in c.get("/api/backends").json()}
    assert {"lmstudio", "ollama", "vllm"} <= names
    plugins = c.get("/api/plugins").json()
    assert "echo" in plugins["decoder"] and "arango" in plugins["retrieval"]


def test_configs_listed(tmp_path: Path) -> None:
    names = {x["name"] for x in _client(tmp_path).get("/api/configs").json()}
    assert "offline" in names


def test_run_lifecycle_offline(tmp_path: Path) -> None:
    c = _client(tmp_path)
    r = c.post("/api/runs", json={"config": "offline"})
    assert r.status_code == 200
    run_id = r.json()["id"]

    detail = {}
    for _ in range(200):
        detail = c.get(f"/api/runs/{run_id}").json()
        if detail["status"] != "running":
            break
        time.sleep(0.05)

    assert detail["status"] == "done"
    assert len(detail["turns"]) == 6  # offline.toml max_turns
    assert detail["turns"][0]["action"]["kind"] == "move"
    # a capture file was written for this run
    caps = {x["name"] for x in c.get("/api/captures").json()}
    assert any(n.startswith(f"web_offline_{run_id}") for n in caps)


def test_reports_record_annotate_and_leaderboard(tmp_path: Path) -> None:
    c = _client(tmp_path)
    run_id = c.post("/api/runs", json={"config": "offline"}).json()["id"]
    for _ in range(200):
        if c.get(f"/api/runs/{run_id}").json()["status"] != "running":
            break
        time.sleep(0.05)
    # the finished run is persisted to the registry (written in the run thread's finally)
    rec = None
    for _ in range(100):
        rec = next((x for x in c.get("/api/reports/runs").json() if x["id"] == run_id), None)
        if rec:
            break
        time.sleep(0.05)
    assert rec is not None and isinstance(rec["metrics"], dict)
    # annotate: label + tags + notes, then filter by tag
    upd = c.post(f"/api/reports/runs/{run_id}", json={"label": "smoke", "tags": ["offline"]})
    assert upd.status_code == 200 and upd.json()["label"] == "smoke"
    tagged = c.get("/api/reports/runs", params={"tag": "offline"}).json()
    assert [r["id"] for r in tagged] == [run_id]
    # leaderboard + 404s
    assert isinstance(c.get("/api/reports/leaderboard").json(), list)
    assert c.get("/api/reports/runs/nope").status_code == 404


def test_mcp_server_crud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # MCP registration is command execution, so it needs the local-only opt-in
    # (or an API token); the CRUD itself is unchanged once allowed.
    monkeypatch.setenv("PUMPKINSPICE_ALLOW_MCP", "1")
    c = _client(tmp_path)
    assert c.get("/api/mcp/servers").json() == []
    up = c.post(
        "/api/mcp/servers",
        json={"name": "memory", "command": "npx", "args": ["-y", "@mcp/server-memory"]},
    ).json()
    assert up["name"] == "memory" and up["enabled"] is True
    assert [s["name"] for s in c.get("/api/mcp/servers").json()] == ["memory"]
    # toggle enabled
    assert c.post("/api/mcp/servers/memory/enabled", json={"enabled": False}).status_code == 200
    assert c.get("/api/mcp/servers").json()[0]["enabled"] is False
    assert c.post("/api/mcp/servers/nope/enabled", json={"enabled": True}).status_code == 404
    # delete
    assert c.delete("/api/mcp/servers/memory").status_code == 200
    assert c.get("/api/mcp/servers").json() == []
    assert c.delete("/api/mcp/servers/memory").status_code == 404


def test_mcp_mutations_403_without_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Without a token or PUMPKINSPICE_ALLOW_MCP=1, MCP mutations (= registering
    # commands to run on this host) are refused; the list stays readable.
    monkeypatch.delenv("PUMPKINSPICE_ALLOW_MCP", raising=False)
    monkeypatch.delenv("PUMPKINSPICE_API_TOKEN", raising=False)
    c = _client(tmp_path)
    assert c.get("/api/mcp/servers").json() == []  # reads stay open
    r = c.post("/api/mcp/servers", json={"name": "m", "command": "npx"})
    assert r.status_code == 403 and "PUMPKINSPICE_API_TOKEN" in r.json()["detail"]
    assert c.post("/api/mcp/servers/m/enabled", json={"enabled": True}).status_code == 403
    assert c.delete("/api/mcp/servers/m").status_code == 403


def test_chat_skips_mcp_servers_when_not_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An enabled server on disk must NOT be spawned by /api/chat when MCP is not
    # allowed -- the plain streaming path runs as if no servers were enabled.
    import httpx

    import pumpkinspice.web.app as app_module
    from pumpkinspice.web.mcp_servers import McpServer, McpServerStore

    monkeypatch.delenv("PUMPKINSPICE_ALLOW_MCP", raising=False)
    monkeypatch.delenv("PUMPKINSPICE_API_TOKEN", raising=False)
    McpServerStore(tmp_path / "mcp_servers.json").upsert(McpServer(name="m", command="nope"))
    spawned: list[object] = []
    monkeypatch.setattr(app_module, "McpHost", lambda servers: spawned.append(servers))

    body = 'data: {"choices": [{"delta": {"content": "hi"}}]}\n\ndata: [DONE]\n\n'
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=transport, **kw))

    r = _client(tmp_path).post("/api/chat", json={"messages": [{"role": "user", "content": "q"}]})
    assert r.status_code == 200 and '"delta": "hi"' in r.text
    assert spawned == []  # McpHost never constructed


def test_api_token_auth(tmp_path: Path) -> None:
    c = TestClient(create_app(configs_dir=CONFIGS, captures_dir=tmp_path, api_token="s3cret"))
    assert c.get("/api/health").status_code == 200  # health stays open for probes
    r = c.post("/api/runs", json={"config": "offline"})
    assert r.status_code == 401 and r.json()["detail"] == "missing or invalid API token"
    assert c.get("/api/configs", headers={"Authorization": "Bearer wrong"}).status_code == 401
    ok = {"Authorization": "Bearer s3cret"}
    assert c.get("/api/configs", headers=ok).status_code == 200
    # passes auth; the 404 is the endpoint's own validation, not the middleware
    assert c.post("/api/runs", json={"config": "nope"}, headers=ok).status_code == 404


def test_base_url_allowlist_blocks_ssrf(tmp_path: Path) -> None:
    c = _client(tmp_path)
    r = c.get("/api/decoder/models", params={"base_url": "http://169.254.169.254"})
    assert r.status_code == 400 and "configured backends" in r.json()["detail"]
    r = c.post("/api/chat", json={"base_url": "http://evil.example", "messages": []})
    assert r.status_code == 400
    # a configured backend passes the allowlist (502 = nothing listening is fine)
    r = c.get("/api/decoder/models", params={"base_url": "http://localhost:11434"})
    assert r.status_code != 400


def test_run_config_name_confinement(tmp_path: Path) -> None:
    c = _client(tmp_path)
    # absolute paths and traversal are rejected outright by the name charset
    assert c.post("/api/runs", json={"config": "/etc/passwd"}).status_code == 400
    assert c.post("/api/runs", json={"config": "../offline"}).status_code == 400
    # a missing (but well-formed) name 404s and never echoes a server path
    r = c.post("/api/runs", json={"config": "does_not_exist"})
    assert r.status_code == 404 and "/" not in r.json()["detail"]


def test_request_field_bounds(tmp_path: Path) -> None:
    c = _client(tmp_path)
    assert c.post("/api/runs", json={"config": "offline", "max_turns": 0}).status_code == 422
    assert c.post("/api/hanoi/trials", json={"disks": 99}).status_code == 422


def test_capture_read_hardening(tmp_path: Path) -> None:
    c = _client(tmp_path)
    (tmp_path / "ok.jsonl").write_text('{"a": 1}\n')
    assert c.get("/api/captures/ok.jsonl").json() == [{"a": 1}]
    assert c.get("/api/captures/ok.txt").status_code == 404  # .jsonl only
    # a symlink pointing outside captures_dir is rejected after resolution
    outside = tmp_path.parent / f"{tmp_path.name}-outside.jsonl"
    outside.write_text('{"x": 1}\n')
    (tmp_path / "link.jsonl").symlink_to(outside)
    assert c.get("/api/captures/link.jsonl").status_code == 404
    # oversized captures are refused before reading (sparse file: no real disk use)
    big = tmp_path / "big.jsonl"
    big.touch()
    os.truncate(big, 50 * 1024 * 1024 + 1)
    assert c.get("/api/captures/big.jsonl").status_code == 413


def test_run_missing_config_404(tmp_path: Path) -> None:
    r = _client(tmp_path).post("/api/runs", json={"config": "does_not_exist"})
    assert r.status_code == 404


def test_run_build_failure_is_clean_400(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # arango retrieval needs scoped creds; without them the run build fails -- the
    # API should surface a clean 400, not a 500.
    monkeypatch.delenv("ARANGO_AGENT_USER", raising=False)
    monkeypatch.delenv("ARANGO_AGENT_PASSWORD", raising=False)
    r = _client(tmp_path).post("/api/runs", json={"config": "arango_real"})
    assert r.status_code == 400
    assert "scoped read-only" in r.json()["detail"]


def test_inline_config_builds_run() -> None:
    cfg = _inline_config("pgvector+relational", "Craft a copper dagger.", 30)
    assert cfg.plugin_name("retrieval") == "pgvector"
    assert cfg.slot_config("retrieval")["relational"] is True
    assert cfg.task == "Craft a copper dagger." and cfg.max_turns == 30
    assert "model" not in cfg.slot_config("decoder")  # model is ambient (the loaded one)
    assert cfg.plugin_name("prompt") == "default"  # Stage 1 by default


def test_inline_config_plan_strategy() -> None:
    cfg = _inline_config("pgvector", "t", 10, prompt="plan")
    assert cfg.plugin_name("prompt") == "plan"  # Stage 2 planning prompt


def test_inline_config_goal_item_stops_run() -> None:
    # a goal item -> the loop stops on craft; blank/None -> runs to max_turns
    cfg = _inline_config("null", "Craft a copper dagger.", 50, "default", None, "copper_dagger")
    assert cfg.run["goal_item"] == "copper_dagger"
    assert "goal_item" not in _inline_config("null", "x", 10, "default", None, None).run


def test_stop_unknown_run_404(tmp_path: Path) -> None:
    assert _client(tmp_path).post("/api/runs/nope/stop").status_code == 404


def test_inline_config_per_batch_character_and_model() -> None:
    from pumpkinspice.web.settings import AppSettings

    # per-batch overrides (parallel sweep): a 2nd character + model override
    cfg = _inline_config(
        "null", "t", 10, "default", AppSettings(model="globalM"), None, "character_2", "overrideM"
    )
    assert cfg.slot_config("world")["character"] == "character_2"
    assert cfg.slot_config("decoder")["model"] == "overrideM"  # override beats the global
    # defaults stay character_1, no override (uses the global/ambient model)
    assert _inline_config("null", "t", 10).slot_config("world")["character"] == "character_1"


def test_stochastic_sampler_releases_greedy_pins() -> None:
    from pumpkinspice.web.runs import _stochastic_sampler

    s = _stochastic_sampler(0.7, 3)
    assert s["temperature"] == 0.7 and s["seed"] == 3  # per-seed reproducible
    assert s["top_k"] == 0  # THE fix: not 1 (greedy), so temperature actually bites
    assert s["top_p"] == 0.95


def test_trials_validation(tmp_path: Path) -> None:
    c = _client(tmp_path)
    assert c.post("/api/trials", json={"retrieval": "nope", "task": "t"}).status_code == 400
    bad_prompt = {"retrieval": "null", "task": "t", "prompt": "bogus"}
    assert c.post("/api/trials", json=bad_prompt).status_code == 400


def test_reset_character_posts_delete_then_create(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    from pumpkinspice.web.runs import reset_herobench_character

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real(transport=transport, **kw))
    reset_herobench_character("http://world", "character_1")
    assert seen == ["/characters/delete", "/characters/create"]


def test_make_record_carries_batch_tags(tmp_path: Path) -> None:
    from pumpkinspice.config import load_config
    from pumpkinspice.web.runs import RunManager

    cfg = load_config(CONFIGS / "offline.toml")
    mgr = RunManager(tmp_path)
    rec = mgr._make_record(cfg, "label", tags=["batch:abc", "seed:2"])
    assert rec.tags == ["batch:abc", "seed:2"]
    assert rec.id in {r.id for r in mgr.list()}


def test_stop_batch_aborts_and_stops_inflight_trial(tmp_path: Path) -> None:
    import threading

    from pumpkinspice.web.runs import RunManager, RunRecord

    mgr = RunManager(tmp_path)
    batch_ev = threading.Event()
    mgr._batch_stops["b1"] = batch_ev
    mgr._runs["r1"] = RunRecord(
        id="r1",
        config_name="c",
        task="t",
        plugins={},
        status="running",
        tags=["batch:b1", "seed:1"],
    )
    run_ev = threading.Event()
    mgr._stops["r1"] = run_ev
    assert mgr.stop_batch("b1") is True
    assert batch_ev.is_set()  # no further trials start
    assert run_ev.is_set()  # the in-flight trial is told to stop
    assert mgr.stop_batch("nope") is False  # unknown batch


def test_stop_batch_endpoint_unknown_404(tmp_path: Path) -> None:
    assert _client(tmp_path).post("/api/trials/nope/stop").status_code == 404


def test_single_run_seed_applies_sampler_and_tags(tmp_path: Path) -> None:
    # a seed on the single-run path reproduces a stochastic trial -> tagged seed/temp
    c = _client(tmp_path)
    r = c.post(
        "/api/runs",
        json={"retrieval": "null", "task": "t", "seed": 7, "temperature": 0.5, "max_turns": 1},
    )
    assert r.status_code == 200
    rid = r.json()["id"]
    rec = next(x for x in c.get("/api/runs").json() if x["id"] == rid)
    assert "seed:7" in rec["tags"] and "temp:0.5" in rec["tags"]


def test_trials_run_offline_with_seed_and_temp_tags(tmp_path: Path) -> None:
    from pumpkinspice.config import load_config
    from pumpkinspice.web.runs import RunManager

    cfg = load_config(CONFIGS / "offline.toml")
    # point the (suppressed) reset at a dead port so the test can't touch a live world
    cfg.slots["world"]["base_url"] = "http://127.0.0.1:9"
    mgr = RunManager(tmp_path)
    batch = mgr.start_trials(cfg, "lbl", 1, 0.5)
    recs: list = []
    for _ in range(200):
        recs = [r for r in mgr.list() if f"batch:{batch}" in r.tags]
        if recs and recs[0].status != "running":
            break
        time.sleep(0.05)
    assert len(recs) == 1 and recs[0].status == "done"
    assert "seed:1" in recs[0].tags and "temp:0.5" in recs[0].tags


def test_inline_config_applies_settings() -> None:
    from pumpkinspice.web.settings import AppSettings

    cfg = _inline_config(
        "pgvector",
        "t",
        10,
        settings=AppSettings(model="m1", max_tokens=256, history_window=5, temperature=0.5),
    )
    d = cfg.slot_config("decoder")
    assert d["model"] == "m1" and d["max_tokens"] == 256
    assert d["sampler"]["temperature"] == 0.5
    assert cfg.run["history_window"] == 5


def test_model_settings_roundtrip(tmp_path: Path) -> None:
    c = _client(tmp_path)
    assert c.get("/api/model/settings").json() == {
        "model": "",
        "temperature": 0.0,
        "max_tokens": 0,
        "history_window": 0,
    }
    upd = c.post("/api/model/settings", json={"model": "mistral-24b", "max_tokens": 256}).json()
    assert upd["model"] == "mistral-24b" and upd["max_tokens"] == 256
    assert c.get("/api/model/settings").json()["model"] == "mistral-24b"  # persisted
    assert isinstance(c.get("/api/model/available").json(), list)  # [] if LMStudio down


def test_retrieval_and_prompt_options(tmp_path: Path) -> None:
    c = _client(tmp_path)
    opts = c.get("/api/retrieval-options").json()
    assert "pgvector+relational" in opts and "arango" in opts and "null" in opts
    prompts = c.get("/api/prompt-options").json()
    assert "default" in prompts and "plan" in prompts


def test_run_unknown_prompt_400(tmp_path: Path) -> None:
    r = _client(tmp_path).post(
        "/api/runs", json={"retrieval": "null", "task": "x", "prompt": "nope"}
    )
    assert r.status_code == 400


def test_run_requires_config_or_options(tmp_path: Path) -> None:
    assert _client(tmp_path).post("/api/runs", json={}).status_code == 400


def test_run_unknown_retrieval_400(tmp_path: Path) -> None:
    r = _client(tmp_path).post("/api/runs", json={"retrieval": "nope", "task": "x"})
    assert r.status_code == 400


def test_spa_index_is_no_cache(tmp_path: Path) -> None:
    r = _client(tmp_path).get("/")
    if r.status_code == 404:
        pytest.skip("frontend not built (no frontend/dist)")
    assert r.headers.get("cache-control") == "no-cache"
    assert "text/html" in r.headers.get("content-type", "")


def test_inline_hanoi_config_builds_run() -> None:
    from pumpkinspice.web.app import HANOI_SYSTEMS, _inline_hanoi_config

    cfg = _inline_hanoi_config("executor", 100, 4)
    assert cfg.plugin_name("world") == "hanoi"
    assert cfg.plugin_name("retrieval") == "null"
    assert cfg.slot_config("world")["disks"] == 4
    assert cfg.run["goal_state_key"] == "solved"
    assert cfg.slot_config("prompt")["system"] == HANOI_SYSTEMS["executor"]
    assert "4-disk" in cfg.task
    # unknown strategy falls back to the default Hanoi system text, not a KeyError
    assert (
        _inline_hanoi_config("bogus", 10, 3).slot_config("prompt")["system"]
        == HANOI_SYSTEMS["default"]
    )


def test_hanoi_run_lifecycle_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # No live LMStudio in CI -- swap the decoder for the offline echo stub via a
    # tiny scripted solve (1 disk: A->C), same pattern as configs/offline.toml.
    import pumpkinspice.web.app as app_module

    def fake_cfg(prompt, max_turns, disks, settings=None, model=None):  # type: ignore[no-untyped-def]
        cfg = _inline_hanoi_config_orig(prompt, max_turns, disks, settings, model)
        cfg.run["decoder"] = "echo"
        cfg.slots["decoder"] = {"script": [{"action": "move", "args": {"from": "A", "to": "C"}}]}
        return cfg

    _inline_hanoi_config_orig = app_module._inline_hanoi_config
    monkeypatch.setattr(app_module, "_inline_hanoi_config", fake_cfg)
    c = _client(tmp_path)
    r = c.post("/api/hanoi/runs", json={"prompt": "default", "max_turns": 5, "disks": 1})
    assert r.status_code == 200
    run_id = r.json()["id"]
    detail = {}
    for _ in range(200):
        detail = c.get(f"/api/runs/{run_id}").json()
        if detail["status"] != "running":
            break
        time.sleep(0.05)
    assert detail["status"] == "done"
    assert detail["turns"][0]["world_state"]["disks"] == 1
    # persisted with benchmark="hanoi" and success (1-disk solve in 1 move)
    rec = None
    for _ in range(100):
        rec = next((x for x in c.get("/api/reports/runs").json() if x["id"] == run_id), None)
        if rec:
            break
        time.sleep(0.05)
    assert rec is not None
    assert rec["benchmark"] == "hanoi"
    assert rec["goal"] == "state:solved"
    assert rec["metrics"]["success"] is True


def test_hanoi_trials_unknown_prompt_400(tmp_path: Path) -> None:
    r = _client(tmp_path).post("/api/hanoi/trials", json={"prompt": "bogus"})
    assert r.status_code == 400


def test_start_trials_does_not_reset_herobench_for_non_herobench_world(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression: start_trials used to call reset_herobench_character
    unconditionally, which would blind-POST delete/create at a live HeroBench
    character even for a Hanoi (or any non-herobench) trials batch."""
    import pumpkinspice.web.runs as runs_module
    from pumpkinspice.config import RunConfig

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        runs_module, "reset_herobench_character", lambda url, ch: calls.append((url, ch))
    )
    cfg = RunConfig(
        run={
            "decoder": "echo",
            "retrieval": "null",
            "world": "hanoi",
            "prompt": "default",
            "capture": "jsonl",
            "task": "t",
            "max_turns": 1,
            "goal_state_key": "solved",
        },
        slots={
            "decoder": {"script": [{"action": "move", "args": {"from": "A", "to": "C"}}]},
            "retrieval": {},
            "world": {"disks": 1},
            "prompt": {},
            "capture": {},
        },
    )
    mgr = runs_module.RunManager(tmp_path)
    batch = mgr.start_trials(cfg, "lbl", 1, 0.5)
    for _ in range(200):
        recs = [r for r in mgr.list() if f"batch:{batch}" in r.tags]
        if recs and recs[0].status != "running":
            break
        time.sleep(0.05)
    assert calls == []  # never touched HeroBench's reset endpoint


def test_make_record_benchmark_from_world_plugin(tmp_path: Path) -> None:
    from pumpkinspice.web.app import _inline_hanoi_config
    from pumpkinspice.web.runs import RunManager

    mgr = RunManager(tmp_path)
    rec = mgr._make_record(_inline_hanoi_config("default", 10, 3), "lbl")
    assert rec.benchmark == "hanoi"
    hb_rec = mgr._make_record(_inline_config("null", "t", 10), "lbl2")
    assert hb_rec.benchmark == "herobench"
