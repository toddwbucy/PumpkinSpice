#!/usr/bin/env python
"""Build-side seeder: materialize the MATH500 dataset into the on-disk layout the
MATH runner reads (load_math_dir), for the floor-test reasoning arm.

This reproduces the corpus of Masoomi et al., "Geometric Signatures of Reasoning"
(arXiv:2607.01571): the three MATH500 categories Algebra, Counting & Probability,
and Precalculus, with difficulty annotations 1-5. It writes one JSON per problem as
`{problem, level: "Level N", type: <subject>, solution}` -- the schema load_math_dir
parses -- under `<out>/<Subject>/<id>.json`, and a `manifest.json` recording the
paper's exact selection (9 questions per category, 3 at each of levels 1/3/5, except
Counting & Probability which has only 2 level-1 problems in MATH500).

Gold answer: MATH500 ships a clean `answer` field. We keep the original worked
`solution` (which the grader extracts the boxed gold from) but GUARANTEE a parseable
boxed gold by appending `\\boxed{answer}` when the solution has none, so grading never
silently fails on a gold with no box.

Source: HuggingFaceH4/MATH-500 (the canonical 500-problem subset). Fetches the single
public `test.jsonl` over HTTP (no `datasets` dependency, no auth). Idempotent:
overwrites by id.

Run:
  uv run --extra introspect python scripts/seed_math500.py \
      --out ~/olympus/Data/benchmarks/math500_json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

# Make the package importable when run as a plain script (for the boxed-gold check).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pumpkinspice.introspect.bench_math import last_boxed_string

# The three categories the paper uses, and the difficulty levels it draws from.
PAPER_CATEGORIES = ("Algebra", "Counting & Probability", "Precalculus")
PAPER_LEVELS = (1, 3, 5)  # easy / medium / hard
PAPER_PER_LEVEL = 3  # 3 questions per (category, level), subject to availability

# Filesystem-safe directory name per subject (the real subject string is kept in `type`).
_DIR = {
    "Algebra": "Algebra",
    "Counting & Probability": "Counting_and_Probability",
    "Precalculus": "Precalculus",
}


def _rows(dataset_id: str) -> list[dict]:
    # The dataset ships a single public test.jsonl; fetch and parse it directly (follow the
    # LFS redirect) rather than pulling the heavy `datasets` library for a one-time job.
    url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/test.jsonl"
    resp = httpx.get(url, follow_redirects=True, timeout=30.0)
    resp.raise_for_status()
    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


def _ensure_boxed(solution: str, answer: str) -> str:
    """Guarantee the stored solution yields a boxed gold the grader can extract."""
    if last_boxed_string(solution) is not None:
        return solution
    return f"{solution}\n\\boxed{{{answer}}}"


def materialize(rows: list[dict], out: Path) -> dict[str, dict]:
    """Write the 3 categories to `<out>/<Subject>/<id>.json`; return per-(subject,level) counts."""
    counts: dict[str, dict] = {c: dict.fromkeys(range(1, 6), 0) for c in PAPER_CATEGORIES}
    for r in rows:
        subject = r["subject"]
        if subject not in PAPER_CATEGORIES:
            continue
        level = int(r["level"])
        # unique_id looks like "test/precalculus/807.json" -> take the numeric stem for the filename.
        stem = Path(str(r["unique_id"])).stem or f"{subject}_{counts[subject][level]}"
        rec = {
            "problem": r["problem"],
            "level": f"Level {level}",
            "type": subject,
            "solution": _ensure_boxed(str(r["solution"]), str(r["answer"])),
        }
        d = out / _DIR[subject]
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{stem}.json").write_text(json.dumps(rec, ensure_ascii=False))
        counts[subject][level] += 1
    return counts


def paper_selection(rows: list[dict]) -> list[dict]:
    """The paper's exact question set: PAPER_PER_LEVEL per (category, level in {1,3,5}),
    in dataset order (deterministic), capped by availability (C&P has only 2 level-1)."""
    picked: list[dict] = []
    for cat in PAPER_CATEGORIES:
        for lv in PAPER_LEVELS:
            got = [r for r in rows if r["subject"] == cat and int(r["level"]) == lv][
                :PAPER_PER_LEVEL
            ]
            picked.extend({"unique_id": r["unique_id"], "subject": cat, "level": lv} for r in got)
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output directory for the materialized corpus")
    ap.add_argument("--dataset", default="HuggingFaceH4/MATH-500", help="HF dataset id")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    rows = _rows(args.dataset)
    counts = materialize(rows, out)
    manifest = paper_selection(rows)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    total = sum(sum(lv.values()) for lv in counts.values())
    print(f"materialized {total} problems (3 categories) -> {out}")
    for cat in PAPER_CATEGORIES:
        avail = counts[cat]
        print(f"  {cat:26s} levels 1/3/5 = {avail[1]}/{avail[3]}/{avail[5]}")
    print(f"paper selection: {len(manifest)} questions -> {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
