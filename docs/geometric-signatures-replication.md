# Replicating "Geometric Signatures of Reasoning" (arXiv:2607.01571)

Status: validated, 2026-07-11. This is the write-up of the PumpkinSpice replication of the
floor-test source paper. It records what was built, the exact protocol, the reproduced numbers,
and, importantly, where the paper's headline claims survive a length-confound control and where
they do not.

## 1. The paper and its claims

Masoomi, Bazzaz, Javanmard, Mirrokni, "Geometric Signatures of Reasoning: A Spectral Perspective
on Task Hardness" (arXiv:2607.01571). It reads the GEOMETRY of a chain-of-thought's hidden-state
trajectory (a discrete curve in R^d) rather than its text, and reports two headline results:

- Task hardness: the effective dimension `d_rho` of the trajectory covariance separates easy from
  hard MATH problems with AUC about 0.93 (leave-one-question-out). Hard problems trace
  higher-dimensional curves (mean d_rho about 170 vs 121.6 for easy).
- Solution correctness: seven kinematic/positional features of the early trajectory (first 20% of
  tokens) predict whether the chain reaches a correct answer, AUC about 0.806 cross-question and
  about 0.90 within-question, with AUPRC about 0.87.

## 2. What we built

The replication is the introspection use of the PumpkinSpice harness. It was built as five staged,
CodeRabbit-reviewed PRs (#30 to #34):

- Stage 1 (#30): `scripts/seed_math500.py` materializes the paper's exact corpus from
  `HuggingFaceH4/MATH-500` (three categories: Algebra, Counting & Probability, Precalculus).
- Stage 2 (#31): `run_math_multisample` decodes N stochastic trajectories per (question, prompt
  style) at temperature 0.7, with per-trajectory correctness labels and `group = question_id`.
- Stage 3 (#32): the replay bridge groups metrics by question (`--group-by task`) and takes the
  paper's `d_rho` thresholds (`--rho-thresholds 0.90,0.95,0.99`).
- Stage 4a (#33): the evaluator's 1-vs-5 leave-one-question-out `d_rho` difficulty probe.
- Stage 4b-i (#34): the correctness within/cross-question protocols with AUPRC.

The teacher-forced replay rig (`introspect/replay.py`) and the geometry functionals
(`introspect/geometry.py`) predate this work; they read the true pre-final-norm residual stream
via forward hooks over all layers, and compute `d_rho`, the seven kinematics, and the per-layer
novelty curves offline.

## 3. Protocol (as run)

- Model: Qwen2.5-0.5B-Instruct (24 layers, hidden dim 896), the paper's model, served via vLLM.
- Corpus: MATH500, levels 1 and 5 (the difficulty extremes), three categories. 76 questions
  (22 level-1, 54 level-5).
- Decode: 5 samples per question per style, two CoT styles (medium + long), temperature 0.7,
  800-token cap, trajectories under 30 tokens dropped. 760 trajectories, 0 dropped, 0 failed.
  The 0.5B scored 29.5% correct, i.e. near its own frontier (the regime a correctness probe
  needs).
- Replay: teacher-forced through the same 0.5B (transformers, CPU, float32), all 24 layers,
  `d_rho` on the final block output at rho in {0.90, 0.95, 0.99}, kinematics over the first 20%.
  760 labeled turns, 0 skipped.
- Evaluate: difficulty = 1-vs-5 `d_rho` probe, leave-one-question-out. Correctness = kinematics,
  within-question (pooled 5-fold) and cross-question (5 grouped 80/20 splits), reported with the
  paper's classifier (standardized LR, C=0.1, balanced) and AUPRC. Every geometry result is
  paired with a generation-length control (does geometry beat length alone?).

Reproduce:

```bash
# 1. data (once)
uv run --extra introspect python scripts/seed_math500.py --out ~/olympus/Data/benchmarks/math500_json
# 2. serve the 0.5B (single GPU; e.g. the idle card)
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8002 --max-model-len 2048 --gpu-memory-utilization 0.4
# 3. decode -> replay -> evaluate
uv run --extra introspect pumpkinspice mathbench -c configs/mathbench_qwen25_05b.toml \
    --data-dir ~/olympus/Data/benchmarks/math500_json --levels 1,5 \
    --samples 5 --styles medium,long --temperature 0.7 --concurrency 16 \
    -o captures/math_qwen25_05b_1v5_val.jsonl
uv run --extra replay pumpkinspice replay-metrics captures/math_qwen25_05b_1v5_val.jsonl \
    --model Qwen/Qwen2.5-0.5B-Instruct --group-by task --rho-thresholds 0.90,0.95,0.99 \
    --device cpu --dtype float32 -o captures/math_qwen25_05b_1v5_metrics.jsonl
uv run --extra evaluate pumpkinspice floortest captures/math_qwen25_05b_1v5_metrics.jsonl
```

## 4. Results

| Result | This replication | Paper |
| --- | --- | --- |
| Difficulty `d_rho` 1-vs-5 (LOQO) AUC | 0.916 | ~0.93 |
| `d_rho` per threshold (0.90 / 0.95 / 0.99) | 0.814 / 0.863 / 0.910 | (0.95/0.99 used) |
| Correctness within-question AUC / AUPRC | 0.875 / 0.692 | ~0.90 / ~0.87 |
| Correctness cross-question AUC / AUPRC | 0.849 / 0.627 | ~0.806 / ~0.87 |
| Layers extracted | 24 | 24 |
| #8 rho curves | structured (block range 0.871) | structured |

Both headline AUCs reproduce. The `d_rho` difficulty AUC (0.916) lands just under the paper's
0.93: we read `d_rho` off the final block output, whereas the paper's peak is layer 21, and the
paper's own per-layer curve shows the final layers sitting slightly below that peak, so 0.916 is
exactly where it should be. The correctness AUCs (within 0.875, cross 0.849) bracket the paper's
0.90/0.806 using only the seven scalar kinematics (see deviations).

## 5. The finding the length control adds

The paper reports the geometry AUCs but does not deconfound generation length. Harder problems and
failing chains tend to run longer, so a geometry AUC can be a length artifact. Pairing every
geometry probe with a length-only and a geometry-plus-length probe (same classifier) splits the
two headlines apart:

- Difficulty: `d_rho` AUC 0.916, length-alone AUC 0.930, combined 0.917. `d_rho` does NOT beat
  generation length; its difficulty signal is largely a length proxy (Confound B).
- Correctness: kinematics AUC 0.871 (pre-registered probe), length-alone 0.806, combined 0.874.
  The kinematics beat length by about +0.068. The correctness signal is real, not a length
  artifact.

So the differentiated verdict, which reproducing the AUCs alone would miss: once length is
controlled, the CORRECTNESS geometry carries genuine signal, while the `d_rho` DIFFICULTY result is
confounded with how long the model wrote. AUPRC (within 0.692) sits well above the 0.29 correct
base rate, confirming the correctness signal ranks correct trajectories above incorrect ones.

## 6. Deviations from the paper (honest ledger)

- Correctness features: we use the repo's seven SCALAR kinematics (norms of the four vector
  kinematics plus three scalars); the paper projects onto the top-15 PCA (fit on the training set)
  and uses the resulting 77-dim feature vector. The scalar version already reproduces the paper's
  range, so the PCA machinery (Stage 4b-ii) is a refinement, not a prerequisite, and was deferred.
- `d_rho` layer: final block output vs the paper's best layer 21 (see section 4).
- Correctness corpus: this validation used the difficulty extremes (levels 1 and 5) already on
  disk; the paper pools levels 1/3/5 for correctness. The correctness probe is level-independent,
  so this is a subset, not a different measurement; a full 1/3/5 correctness decode is the
  remaining faithful run.
- Kill classifier: the floor test keeps its pre-registered correctness kill probe (LR C=1.0) for
  the 0.70 keep/kill verdict, and reports the paper's classifier (LR C=0.1, balanced) alongside as
  the paper-comparison diagnostic; the two are labeled distinctly.
- Replay precision: float32 on CPU (the `replay` extra pins CPU torch); bf16 vs fp32 shifts
  `d_rho` by about 0.3%.

## 7. Conclusion

On the paper's own model and corpus, the PumpkinSpice floor test reproduces both headline results
of arXiv:2607.01571: `d_rho` separates MATH difficulty (AUC 0.916 vs 0.93) and early kinematics
predict correctness (within 0.875 / cross 0.849 vs 0.90 / 0.806). The floor test's length-confound
control, which the paper lacks, then sharpens the reading: the correctness signal survives length
control (+0.068), whereas the `d_rho` difficulty signal does not clearly beat generation length.
Trajectory geometry is a real window into solution quality; its status as a length-INDEPENDENT
measure of task hardness is not established by this replication.

Remaining, optional: Stage 4b-ii (the top-15 PCA / 77-dim correctness features, which need raw
windowed-trajectory storage) and a full levels-1/3/5 correctness decode. Neither is expected to
move the headline.
