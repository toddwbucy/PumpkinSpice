"""Command-line surface for the PumpkinSpice harness.

pumpkinspice plugins                      list discovered plugins per slot
pumpkinspice run --config <file>          play a HeroBench run with selected plugins
pumpkinspice parity --config <file>       decoder-parity gate (spec s4): greedy decode + artifact
pumpkinspice parity --compare A B         diff two parity artifacts (e.g. LMStudio vs SPU)
pumpkinspice transport --config <file>    transport micro-benchmark (spec s5): latency distribution
pumpkinspice sweep -c <cfg> -m a:256,b    run a config across models (':N' = per-model max_tokens cap)
pumpkinspice analyze <captures...>        metrics + cross-model comparison over captures
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


_DEFAULT_DECODER_URL = "http://192.168.0.203:1234"


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
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    func: Any = args.func
    result = func(args)
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
