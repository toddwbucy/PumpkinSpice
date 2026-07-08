# Floor-test v2 design (issues #7 / #8)

Status: DRAFT for operator review. Motivated by preliminary v1 results (Qwen3-8B) that
surfaced two confounds in the v1 apparatus. This doc specs the redesign. ASCII only.

## 1. What the floor test is

A pre-registered, disconfirmation-first check on whether cheap offline summaries of a
generation's hidden-state trajectory carry real signal (issues #7, #8). Metrics, computed
by teacher-forced replay:

- `d_rho` (effective dimension): PCs of the trajectory covariance for 0.5/0.75/0.9 variance.
- Early kinematics: seven scalars over the first fifth of the trajectory.
- Per-layer novelty `rho_block` / `rho_mlp` (#8): novelty each layer adds, across depth.

Pre-registered kills (AUC < 0.70 -> dead):
- kill1: `d_rho` separates hard/easy on the agentic arm.
- kill2: early kinematics predict correctness within a task type.
- kill3: the kinematics probe transfers across task types.
- #8: the rho curves are structured (not flat/noise) -- a judgment, reported not auto-decided.

## 2. v1 result (preliminary, Qwen3-8B, reasoning=MATH + 4 planning tiers)

| kill | AUC | note |
| --- | --- | --- |
| kill1 d_rho hard/easy [planning] | 0.686 | KILL, confounded |
| kill2 kinematics->correct [reasoning] | 0.685 | KILL, clean but marginal |
| kill2 kinematics->correct [planning] | 0.978 | PASS, confounded (task identity) |
| kill3 reasoning->planning | 0.154 | KILL (inverted) |
| kill3 planning->reasoning | 0.448 | KILL |
| #8 rho curves | range 1.36 | structured (clean positive) |

Length control (geometry vs generation length; length = deterministic single-feature AUC;
marginal = combined - length is a noisy point estimate, NO confidence interval):
- correctness[reasoning]: geom 0.685, length 0.667, combined 0.715 (geom beyond length +0.047)
- difficulty[reasoning]:  geom 0.720, length 0.784, combined 0.757 (geom beyond length -0.027)
- kill3 length baseline (does length transfer across task types better than geometry?):
  reasoning->planning geometry 0.154 vs length 0.823; planning->reasoning geometry 0.448 vs
  length 0.667 -- LENGTH transfers, geometry does not (it inverts). The cross-task signal,
  such as it is, is length.

## 3. The two confounds v1 exposed

### Confound A -- per-run labeling (task identity)
`make_label_fn` stamps ONE `eventual_correct` and ONE `hard` on all ~100 turns of a tier.
With only ~5 tiers, any probe that "predicts correctness/difficulty" actually learns "which
task this is" -- different tasks trace different paths, so the kinematics separate them
trivially. The planning kill2 (0.978) is yellow_slime-vs-rest discrimination, not correctness.
The effective independent-unit count for the agentic arm is ~5, not ~500.

### Confound B -- generation length (thinking budget)
In thinking mode the model chooses its own generation length, spending more tokens on
hard/failing problems (planning: easy mean 1662 tokens, hard mean 2902). `d_rho` and the
kinematics are length-sensitive, so a geometry AUC can be a length artifact. The length
control shows this is real and per-cell: for reasoning DIFFICULTY, length (0.784) beats the
geometry probe (0.720) and adding geometry to length does not help (marginal -0.027 -- but
this is a noisy point estimate, not a verdict); reasoning CORRECTNESS keeps a small residual
(+0.047). The kill3 length baseline is the sharpest: length transfers across task types
(0.823) while the geometry transfer inverts (0.154), so the cross-task "signal" is length.

## 4. v2 design

### 4.1 Externalize the reasoning loop (the key change)
Turn OFF the model's internal thinking; drive reasoning as bounded harness steps
(ReAct: think -> act -> observe, one short generation each). This:
- controls generation length (uniform, bounded steps) -> geometry cannot hide behind
  thinking budget;
- keeps task capability (the loop still reasons, just visibly);
- makes the reasoning state observable to the harness;
- collapses compute (short generations -> cheap decode AND trivial replay; the O(T^3)
  eig that dominated v1 only hurt because T was ~10k of thinking).

Add "reasoning location" as an independent variable: internal-CoT vs harness-externalized,
same tasks/models, to measure where (if anywhere) the signal lives.

### 4.2 Per-episode metric (operator decision)
One trajectory-metric per EPISODE, from the representative (first/planning) step's
trajectory -- a single generation, matching the reasoning arm's one-generation-one-
trajectory structure, and ~50x cheaper to replay than aggregating all turns. Decode still
requires the full episode (outcome is episode-terminal); only replay is on the one step.

### 4.3 Deconfound the labels (Confound A)
- Correctness/transfer: MANY stochastic episodes per task (temperature > 0, varied seeds),
  each labeled by its OWN outcome, so correctness varies WITHIN a held-constant task.
- Difficulty: MULTIPLE distinct tasks per hard/easy bin (~10-15), so d_rho must find a
  difficulty signature that generalizes across tasks, not memorize one.
- Grouped cross-validation: group by episode (correctness) / task (difficulty) so no unit
  leaks across folds; report the effective independent-unit count, not the turn count.

### 4.4 Deconfound length (Confound B)
Uniform harness steps hold length ~constant. The evaluator's length-control (implemented)
reports, per probe, geometry vs length-alone vs combined, so every future run states
whether the geometry beats length. A cell where `combined ~ length` is a length artifact.
Hardening (from the PR #20 review): the length feature is SPAN-AWARE (output tokens for
span="output", prompt+output for span="full", recorded in TrajectoryMetrics) so a full-span
corpus cannot escape the control; the length AUC is a DETERMINISTIC single-feature AUC (a CV
logistic on one column deflates it and inflates the geometry's margin); the cross-transfer
kill3 gets its own length baseline (it is the probe most exposed to a transferring length
proxy); and `marginal` is documented as a noisy point estimate with no CI (do not read a
keep/kill sign off it alone -- a bootstrap CI is v2 work).

### 4.5 Frontier calibration
Within-task outcome variation only exists near ~50% success, and the frontier is
model-specific. Use a FIXED pool of ~20-30 graded tasks run on every model, and select the
near-50/50 subset per model for the correctness cells: comparable pool + per-model balance.
Needs a cheap calibration pass (a few episodes/task/model to estimate success rate).

## 5. Statistical power / compute

AUC reliability tracks the minority-outcome count. For a defensible 0.70 keep/kill call,
target ~300-400 balanced episodes per arm per model (95% CI ~ +/-0.065). Because v2
generations are short, this is cheap: reasoning ~400 problems x 1 generation; agentic ~400
episodes x ~40 short steps, batched through vLLM -> a few hours per model per arm, and
replay is minutes (short traces). The GPU-reduction optimization is therefore NOT needed
for v2 (it only helped the long internal-CoT traces we are removing).

## 6. Carries over vs new

Carries over: the harness, replay driver, geometry, evaluator core, #8, dtype/embed
provenance, the reasoning (MATH) arm structure, the length-control diagnostic.

New work: (a) an externalized ReAct harness step (thinking off); (b) a multi-task,
frontier-calibrated HeroBench set; (c) a stochastic per-episode decode runner; (d) grouped
CV + per-episode aggregation in the evaluator; (e) reasoning-arm difficulty control.

## 7. Success criteria (what validates vs kills the metric under v2)

- If, at controlled length and with per-episode labels, `geometry beyond length > 0` and
  kill2/kill3 clear 0.70 across models -> the geometry carries a real, general signal.
- If `combined ~ length` everywhere -> the v1 signal was thinking budget; the metric is a
  length proxy and should be dropped (or redefined length-invariant).
- #8 (depth structure) is evaluated separately and has held up clean so far.

## 8. Open decisions

1. Target episodes per arm (300-400 defensible vs 200 cheaper).
2. Task pool size and grading (how many tasks per difficulty bin).
3. Whether to keep the internal-CoT arm as a comparison baseline (needs the slow replay).
4. Exact "representative step" definition (first planning turn vs first N steps).
