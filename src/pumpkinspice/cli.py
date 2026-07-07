"""Command-line surface for the PumpkinSpice harness.

pumpkinspice plugins                      list discovered plugins per slot
pumpkinspice run --config <file>          play a HeroBench run with selected plugins
pumpkinspice parity --config <file>       decoder-parity gate (spec s4): greedy decode + artifact
pumpkinspice parity --compare A B         diff two parity artifacts (e.g. LMStudio vs SPU)
pumpkinspice transport --config <file>    transport micro-benchmark (spec s5): latency distribution
pumpkinspice sweep -c <cfg> -m a:256,b    run a config across models (':N' = per-model max_tokens cap)
pumpkinspice analyze <captures...>        metrics + cross-model comparison over captures
pumpkinspice mathbench -c <cfg> --data-dir MATH   decode a local MATH dataset -> Turn captures
pumpkinspice replay-metrics <caps> --model M -o M.jsonl  captures -> labeled metrics (needs 'replay')
pumpkinspice floortest <metrics.jsonl>    #7/#8 floor-test AUCs + kill verdicts (needs 'evaluate')
pumpkinspice serve                        run the web frontend API + SPA
"""

from __future__ import annotations

import argparse
import json
import os
import tomllib
from pathlib import Path
from typing import Any

import httpx

from . import kernel, parity, transport
from .config import RunConfig, load_config
from .logging import configure_logging, get_logger
from .loop import AgentLoop

log = get_logger("pumpkinspice.cli")


def _load_toml(path: str) -> dict[str, Any]:
    return tomllib.loads(Path(path).read_text())


# The decoder endpoint moves between hosts (localhost vs a LAN box); default to
# loopback and let the environment override, same as the web backend does.
_DEFAULT_DECODER_URL = os.environ.get("PUMPKINSPICE_LMSTUDIO_URL", "http://localhost:1234")


def _decoder_client(decoder_cfg: dict[str, Any], timeout: float) -> httpx.Client:
    base_url = str(decoder_cfg.get("base_url", _DEFAULT_DECODER_URL)).rstrip("/")
    return httpx.Client(base_url=base_url, timeout=timeout)


def _cmd_plugins(_args: argparse.Namespace) -> int:
    # Machine-readable listing on stdout (logs go to stderr).
    for slot, names in kernel.discover().items():
        print(f"{slot:10s} ({kernel.SLOTS[slot]}): {', '.join(names) or '(none)'}")
    return 0


def _build_loop(cfg: RunConfig) -> AgentLoop:
    parts: dict[str, Any] = {
        slot: kernel.load_plugin(slot, cfg.plugin_name(slot), cfg.slot_config(slot))
        for slot in kernel.SLOTS
    }
    top_k = int(cfg.slot_config("retrieval").get("top_k", 5))
    sampler = cfg.slot_config("decoder").get("sampler", {})
    return AgentLoop(
        decoder=parts["decoder"],
        retrieval=parts["retrieval"],
        world=parts["world"],
        prompt=parts["prompt"],
        capture=parts["capture"],
        task=cfg.task,
        top_k=top_k,
        sampler=sampler,
        history_window=int(cfg.run.get("history_window", 0)),
        goal_item=cfg.run.get("goal_item"),
        goal_level=cfg.run.get("goal_level"),
        goal_skill=cfg.run.get("goal_skill"),
        goal_state_key=cfg.run.get("goal_state_key"),
    )


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    selected = {slot: cfg.plugin_name(slot) for slot in kernel.SLOTS}
    log.info("plugins: %s", selected)
    log.info("task=%r max_turns=%d", cfg.task, cfg.max_turns)
    loop = _build_loop(cfg)
    turns = loop.play(cfg.max_turns)
    last = turns[-1] if turns else None
    log.info(
        "played %d turns; last action=%s ok=%s",
        len(turns),
        last.action if last else None,
        last.outcome.get("ok") if last else None,
    )
    return 0


def _cmd_parity(args: argparse.Namespace) -> int:
    if args.compare:
        a = json.loads(Path(args.compare[0]).read_text())
        b = json.loads(Path(args.compare[1]).read_text())
        report = parity.compare_artifacts(a, b)
        for r in report["results"]:
            log.info("parity %s: %s", "MATCH" if r["match"] else "DIVERGE", r)
        print(json.dumps(report, indent=2))
        return 0 if report["pass"] else 1

    if not args.config:
        log.error("parity needs --config <file> (or --compare A B)")
        return 2
    raw = _load_toml(args.config)
    decoder_cfg = raw.get("decoder", {})
    pcfg = raw.get("parity", {})
    prompts = pcfg.get("fixtures") or parity.DEFAULT_FIXTURES
    with _decoder_client(decoder_cfg, timeout=180.0) as client:
        artifact = parity.run_parity(
            client,
            prompts=prompts,
            model=decoder_cfg.get("model"),
            sampler=decoder_cfg.get("sampler"),
            max_tokens=int(pcfg.get("max_tokens", 64)),
        )
    out = args.out or pcfg.get("out", "captures/parity_lmstudio.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(artifact, indent=2))
    log.info(
        "parity: %d fixtures, deterministic=%s -> %s",
        len(artifact["fixtures"]),
        artifact["deterministic"],
        out,
    )
    if not artifact["deterministic"]:
        log.warning("LMStudio greedy decode is NOT reproducible -- pin sampler/seed before scoring")
    return 0


def _cmd_transport(args: argparse.Namespace) -> int:
    raw = _load_toml(args.config)
    decoder_cfg = raw.get("decoder", {})
    tcfg = raw.get("transport", {})
    iterations = int(args.iterations or tcfg.get("iterations", 50))
    with _decoder_client(decoder_cfg, timeout=60.0) as client:
        artifact = transport.run_transport(
            client,
            prompt=tcfg.get("prompt", "ping"),
            iterations=iterations,
            warmup=int(tcfg.get("warmup", 5)),
            max_tokens=int(tcfg.get("max_tokens", 1)),
            model=decoder_cfg.get("model"),
        )
    out = args.out or tcfg.get("out", "captures/transport_lmstudio.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(artifact, indent=2))
    d = artifact["minimal_decode_ms"]
    pg = artifact["models_ping_ms"]
    log.info("transport (%s, n=%d):", artifact["metadata"]["endpoint"], iterations)
    log.info("  ping  ms: p50=%.1f p90=%.1f p99=%.1f", pg["p50_ms"], pg["p90_ms"], pg["p99_ms"])
    log.info("  decode ms: p50=%.1f p90=%.1f p99=%.1f", d["p50_ms"], d["p90_ms"], d["p99_ms"])
    log.info("  -> %s", out)
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from . import analyze

    paths = [Path(p) for p in args.captures if Path(p).exists()]
    missing = [p for p in args.captures if not Path(p).exists()]
    if missing:
        log.warning("skipping %d missing capture(s): %s", len(missing), ", ".join(missing))
    if not paths:
        log.error("no capture files found")
        return 2
    metrics = [
        analyze.load_metrics(
            p,
            goal_item=args.goal_item,
            goal_level=args.goal_level,
            goal_skill=args.goal_skill,
            goal_state_key=args.goal_state_key,
        )
        for p in paths
    ]
    print(analyze.comparison_table(metrics))
    if args.json:
        Path(args.json).write_text(json.dumps(analyze.metrics_as_dicts(metrics), indent=2))
        log.info("wrote metrics to %s", args.json)
    return 0


def _cmd_floortest(args: argparse.Namespace) -> int:
    """Evaluate the #7/#8 floor test over a labeled-metrics JSONL (produced by the
    replay step) and print the report; optionally write the JSON."""
    from .introspect import evaluate as ev

    try:
        turns = ev.load_labeled_turns(args.metrics)
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        log.error("could not read labeled metrics %s: %s", args.metrics, exc)
        return 2
    if not turns:
        log.error("no labeled turns in %s", args.metrics)
        return 2
    try:
        report = ev.evaluate_floor_test(
            turns, agentic_type=args.agentic_type, threshold=args.threshold
        )
    except ValueError as exc:  # e.g. incommensurable corpus
        log.error("evaluation failed: %s", exc)
        return 2
    print(report.summary())
    if args.json:
        Path(args.json).write_text(json.dumps(ev.report_to_dict(report), indent=2))
        log.info("wrote report to %s", args.json)
    return 0


def _cmd_mathbench(
    args: argparse.Namespace,
) -> int:  # pragma: no cover - needs a live decoder + data
    """Decode a local MATH dataset through the configured decoder into Turn captures."""
    from .introspect import bench_math as bm

    cfg = load_config(args.config)
    decoder = kernel.load_plugin("decoder", cfg.plugin_name("decoder"), cfg.slot_config("decoder"))
    # JsonlCapture appends: warn so a re-run does not silently concatenate a second
    # corpus into the same file (which would replay duplicates into the floor test).
    if Path(args.out).exists():
        log.warning(
            "%s exists; JsonlCapture appends -- delete it first for a clean corpus", args.out
        )
    capture = kernel.load_plugin("capture", "jsonl", {"path": args.out})
    levels = {int(x) for x in args.levels.split(",") if x.strip()} if args.levels else None
    problems = bm.load_math_dir(args.data_dir, levels=levels, limit=args.limit)
    log.info("MATH: %d problems -> %s", len(problems), args.out)
    try:
        turns = bm.run_math_benchmark(decoder, problems, capture, hard_level=args.hard_level)
    finally:
        capture.close()
    n_correct = sum(1 for t in turns if t.outcome.get("correct"))
    log.info(
        "decoded %d, correct %d (%.1f%%)",
        len(turns),
        n_correct,
        100 * n_correct / max(len(turns), 1),
    )
    return 0


def _cmd_replay_metrics(args: argparse.Namespace) -> int:  # pragma: no cover - needs a model
    """Teacher-force replay a capture JSONL into labeled trajectory-metrics JSONL."""
    from .introspect import pipeline
    from .introspect.replay import ReplayModel

    # For HeroBench planning captures, derive (planning, eventual-correct, tier-hard)
    # from the calibrated ramp; otherwise use the default outcome-based labeler.
    label_fn: pipeline.LabelFn = pipeline.labels_from_outcome
    if args.herobench_tier:
        from .introspect import bench_herobench as bh

        if args.herobench_tier not in bh.RAMP:
            log.error(
                "unknown --herobench-tier %r; choices: %s", args.herobench_tier, ", ".join(bh.RAMP)
            )
            return 2
        rows = [json.loads(ln) for ln in Path(args.captures).read_text().splitlines() if ln.strip()]
        label_fn = bh.make_label_fn(rows, bh.RAMP[args.herobench_tier])

    model = ReplayModel.from_pretrained(
        args.model,
        gguf_file=args.gguf,
        device=args.device,
        dtype=args.dtype,
        trajectory_span=args.span,
    )
    try:
        written, skipped = pipeline.replay_captures(
            model, args.captures, args.out, label_fn=label_fn
        )
    finally:
        model.close()
    log.info("wrote %d labeled turns (skipped %d) -> %s", written, skipped, args.out)
    return 0


def _cmd_reports_import(args: argparse.Namespace) -> int:  # pragma: no cover - thin IO glue
    """Backfill existing capture JSONL into the run registry so the Reports tab shows
    historical runs. Strategy is inferred from the filename; model/retrieval from the
    capture itself."""
    from . import analyze
    from .reporting import RunRegistry

    reg = RunRegistry(Path(args.db))
    paths = [Path(p) for p in args.captures if Path(p).exists() and p.endswith(".jsonl")]
    n = 0
    for p in paths:
        turns = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        if not turns:
            continue
        m = analyze.analyze_turns(p.stem, turns, goal_item=args.goal_item)
        stem = p.stem
        strategy = "replan" if "replan" in stem else "plan" if "plan" in stem else "default"
        reg.record(
            {
                "id": stem,
                "benchmark": args.benchmark,
                "model": m.model,
                "strategy": strategy,
                "retrieval": m.backend,
                "task": str(turns[0].get("task") or ""),
                "goal": args.goal_item or "",
                "max_turns": len(turns),
                "started_at": "",
                "finished_at": "",
                "status": "imported",
                "metrics": analyze.metrics_as_dicts([m])[0],
                "capture_path": str(p),
                "tags": ["imported"],
            }
        )
        n += 1
    print(f"imported {n} runs into {args.db}")
    return 0


def _parse_model_spec(spec: str) -> list[tuple[str, int]]:
    """Parse a sweep model spec into (model, max_tokens) pairs.

    Each comma-separated entry is ``model`` or ``model:<int>``; the optional
    ``:<int>`` is a per-model max_tokens cap (0 or omitted = unbounded). Cap a
    rambling non-reasoning model (e.g. ``:256``) for speed; leave reasoning models
    uncapped so their (growing) thinking is never truncated. Model ids may contain
    ``/`` but not ``:``, so the last ``:`` separates the cap.
    """
    out: list[tuple[str, int]] = []
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            model, _, cap = entry.rpartition(":")
            out.append((model.strip(), int(cap)))
        else:
            out.append((entry, 0))
    return out


def _cmd_sweep(args: argparse.Namespace) -> int:  # pragma: no cover - runs real model plays
    from . import analyze

    _load_env_local()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path("configs") / f"{args.config}.toml"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    captures: list[Path] = []
    for model, cap in _parse_model_spec(args.models):
        cfg = load_config(cfg_path)
        cfg.slots["decoder"]["model"] = model
        cfg.slots["decoder"]["max_tokens"] = cap  # 0 = unbounded
        path = out_dir / f"{cfg_path.stem}__{model.replace('/', '_')}.jsonl"
        cfg.slots["capture"]["path"] = str(path)
        log.info("sweep: model=%s max_tokens=%s -> %s", model, cap or "unbounded", path)
        turns = _build_loop(cfg).play(cfg.max_turns)
        log.info("  %s: %d turns", model, len(turns))
        captures.append(path)

    print(
        analyze.comparison_table(
            [
                analyze.load_metrics(
                    p,
                    goal_item=args.goal_item,
                    goal_level=args.goal_level,
                    goal_skill=args.goal_skill,
                    goal_state_key=args.goal_state_key,
                )
                for p in captures
            ]
        )
    )
    return 0


def _load_env_local(env_file: Path | None = None) -> None:
    """Load scoped DB credentials from the repo's .env.local into the environment,
    so `serve` can build retrieval plugins without the operator sourcing it first.
    Already-set vars win (existing values are kept); secrets stay in the gitignored
    file."""
    env_file = env_file or Path(__file__).resolve().parents[2] / ".env.local"
    if not env_file.exists():
        return
    loaded = []
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in os.environ:
            os.environ[key] = value.strip()
            loaded.append(key)
    if loaded:
        log.info("loaded %d scoped credential(s) from .env.local", len(loaded))


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from .web.app import create_app
    except ImportError:
        log.error("the web frontend needs the 'web' extra: uv sync --extra web")
        return 2
    _load_env_local()
    # .env.local may provide the token; an explicit --token still wins.
    token = args.token or os.environ.get("PUMPKINSPICE_API_TOKEN") or None
    if args.host not in ("127.0.0.1", "localhost", "::1") and not token and not args.insecure:
        log.error(
            "refusing to bind %s without auth: pass --token (or set "
            "PUMPKINSPICE_API_TOKEN), or pass --insecure to serve unauthenticated",
            args.host,
        )
        return 2
    log.info("serving PumpkinSpice API on http://%s:%d", args.host, args.port)
    uvicorn.run(create_app(api_token=token), host=args.host, port=args.port, log_level="warning")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pumpkinspice", description=__doc__)
    p.add_argument(
        "--log-level",
        default=None,
        help="DEBUG/INFO/WARNING/ERROR (or $PUMPKINSPICE_LOG_LEVEL; default INFO)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("plugins", help="list discovered plugins").set_defaults(func=_cmd_plugins)

    runp = sub.add_parser("run", help="play a HeroBench run")
    runp.add_argument("--config", "-c", required=True, help="path to a run TOML")
    runp.set_defaults(func=_cmd_run)

    parityp = sub.add_parser("parity", help="decoder-parity gate (spec s4)")
    parityp.add_argument("--config", "-c", help="TOML with [decoder] and [parity]")
    parityp.add_argument("--out", "-o", help="artifact output path")
    parityp.add_argument("--compare", nargs=2, metavar=("A", "B"), help="diff two parity artifacts")
    parityp.set_defaults(func=_cmd_parity)

    transportp = sub.add_parser("transport", help="transport micro-benchmark (spec s5)")
    transportp.add_argument(
        "--config", "-c", required=True, help="TOML with [decoder] and [transport]"
    )
    transportp.add_argument("--out", "-o", help="artifact output path")
    transportp.add_argument("--iterations", "-n", type=int, help="override iteration count")
    transportp.set_defaults(func=_cmd_transport)

    analyzep = sub.add_parser("analyze", help="metrics + cross-model comparison over captures")
    analyzep.add_argument("captures", nargs="+", help="capture .jsonl files")
    analyzep.add_argument("--goal-item", help="success = this item code is in the final inventory")
    analyzep.add_argument("--goal-level", type=int, help="success = final character level >= N")
    analyzep.add_argument(
        "--goal-skill", help="scope --goal-level to a skill (e.g. weaponcrafting)"
    )
    analyzep.add_argument(
        "--goal-state-key", help="success = state[key] is truthy (e.g. 'solved' for Hanoi)"
    )
    analyzep.add_argument("--json", help="also write the metrics as JSON to this path")
    analyzep.set_defaults(func=_cmd_analyze)

    floortestp = sub.add_parser(
        "floortest", help="evaluate the #7/#8 floor test over a labeled-metrics JSONL"
    )
    floortestp.add_argument("metrics", help="labeled-metrics .jsonl (from the replay step)")
    floortestp.add_argument(
        "--agentic-type", default="tool_use", help="task_type kill #1 (d_rho hard/easy) applies to"
    )
    floortestp.add_argument(
        "--threshold", type=float, default=0.7, help="kill AUC threshold (pre-registered 0.7)"
    )
    floortestp.add_argument("--json", help="also write the report as JSON to this path")
    floortestp.set_defaults(func=_cmd_floortest)

    mathp = sub.add_parser("mathbench", help="decode a local MATH dataset into Turn captures")
    mathp.add_argument("--config", "-c", required=True, help="config selecting the decoder")
    mathp.add_argument("--data-dir", required=True, help="directory of MATH release JSON files")
    mathp.add_argument("--out", "-o", default="captures/math.jsonl", help="capture output path")
    mathp.add_argument("--levels", help="comma-separated difficulty levels to include (e.g. 3,4,5)")
    mathp.add_argument("--limit", type=int, help="cap the number of problems")
    mathp.add_argument("--hard-level", type=int, default=4, help="level >= this counts as hard")
    mathp.set_defaults(func=_cmd_mathbench)

    replayp = sub.add_parser(
        "replay-metrics", help="teacher-force replay captures into labeled metrics (needs 'replay')"
    )
    replayp.add_argument("captures", help="capture .jsonl to replay")
    replayp.add_argument("--model", required=True, help="model id or path for the replay model")
    replayp.add_argument("--out", "-o", required=True, help="labeled-metrics .jsonl output")
    replayp.add_argument("--gguf", help="gguf_file to dequantize the same GGUF the harness served")
    replayp.add_argument("--device", default="cpu", help="torch device (cpu, cuda, ...)")
    replayp.add_argument(
        # The load dtype DOES affect the geometry (the residual stream is computed at
        # it; the float64 cast is post-hoc) -- but empirically bf16 vs fp32 shifts d_rho
        # ~0.3%. fp32 SDPA also falls back to the seq^2 math kernel, so long traces
        # (10k+ tokens) need bf16 to fit a 48GB card. bf16 is the practical default;
        # the dtype is recorded in each metrics row so corpora can't mix precisions.
        "--dtype",
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
        help="model load dtype; bfloat16 fits long traces on a 48GB card, fp32 is highest fidelity",
    )
    replayp.add_argument(
        "--span", default="output", choices=["output", "full"], help="trajectory span"
    )
    replayp.add_argument(
        # Tier names mirror bench_herobench.RAMP; validated against it at runtime.
        "--herobench-tier",
        help="label as a HeroBench planning tier (control_gather/chicken_level2/"
        "copper_dagger/yellow_slime/weaponcrafting5); scores eventual correctness + difficulty",
    )
    replayp.set_defaults(func=_cmd_replay_metrics)

    importp = sub.add_parser(
        "reports-import", help="backfill capture .jsonl into the run registry (Reports tab)"
    )
    importp.add_argument("captures", nargs="+", help="capture .jsonl files")
    importp.add_argument("--db", default="captures/results.db", help="registry SQLite path")
    importp.add_argument("--benchmark", default="herobench")
    importp.add_argument("--goal-item", help="success = this item code crafted this run")
    importp.set_defaults(func=_cmd_reports_import)

    sweepp = sub.add_parser("sweep", help="run a config across models, then compare (Stage 1)")
    sweepp.add_argument("--config", "-c", required=True, help="run config (name or path)")
    sweepp.add_argument(
        "--models",
        "-m",
        required=True,
        help="comma-separated model ids; append ':N' for a per-model max_tokens cap "
        "(0/omitted = unbounded), e.g. 'mistral-small-24b-instruct-2501:256,qwen/qwen3.6-27b'",
    )
    sweepp.add_argument(
        "--out-dir", default="captures/sweep", help="where to write per-model captures"
    )
    sweepp.add_argument("--goal-item", help="success = this item code is in the final inventory")
    sweepp.add_argument("--goal-level", type=int, help="success = final character level >= N")
    sweepp.add_argument("--goal-skill", help="scope --goal-level to a skill (e.g. weaponcrafting)")
    sweepp.add_argument(
        "--goal-state-key", help="success = state[key] is truthy (e.g. 'solved' for Hanoi)"
    )
    sweepp.set_defaults(func=_cmd_sweep)

    servep = sub.add_parser("serve", help="run the web frontend API (needs the 'web' extra)")
    servep.add_argument("--host", default="127.0.0.1")
    servep.add_argument("--port", type=int, default=8077)
    servep.add_argument(
        "--token",
        default=os.environ.get("PUMPKINSPICE_API_TOKEN"),
        help="bearer token required on /api/* (default: $PUMPKINSPICE_API_TOKEN; empty = no auth)",
    )
    servep.add_argument(
        "--insecure",
        action="store_true",
        help="allow binding a non-loopback host without an API token",
    )
    servep.set_defaults(func=_cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    func: Any = args.func
    result = func(args)
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
