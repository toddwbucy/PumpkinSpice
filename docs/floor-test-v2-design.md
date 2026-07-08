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

## 8. Resolved decisions (operator-approved)

1. **Episodes/arm:** ~360 (40 x 9 tasks) on the externalized arm; internal-CoT baseline ~120
   (matched subset -- the on/off contrast needs matched N, not full scale).
2. **Task ladder:** 4 bins x ~3 ways (concrete set in 8.1). Goals are **capability milestones**
   (defeat monster M), not XP levels -- leveling happens along the way but the scored goal is
   capability, which carries the planning content.
3. **Keep the internal-CoT arm** as a baseline: reasoning on/off is a clean factorial IV
   (same task/model, first-plan trajectory measured with thinking on vs externalized).
4. **Representative step = the FIRST planning turn** (matches the reasoning arm's
   one-generation-one-trajectory structure; the initial plan stands for the episode).
5. **ICL is DEFERRED to WeaverTools.** The v2 baseline is ICL-free: the harness sets state,
   the model plans cold from the observed state, no worked example in context. An in-context
   demonstration would confound the geometry (imitation + demo-relevance) and can trivialize
   correctness; it is Agent-1 / WeaverTools territory, not the conventional-RAG control.
6. **Context: bounded, NOT max.** The measured first-plan turn has a small, uniform context
   (state + retrieval + instruction). Generation cap differs by arm -- externalized ~512-1024
   (bounded plan/action), internal ~8-16k (room to think); that difference IS the on/off
   contrast. vLLM `--max-model-len ~32768`. Episode-*play* uses bounded history so the agent
   can adapt to its own observed failures (in-context working memory, not ICL); the
   *measurement* context stays small. On failure mid-episode the agent retries WITH the
   failure in context (ReAct/replan); no context reload.

### 8.1 The concrete task ladder (difficulty = element-match x level-for-damage)

Verified mechanics: damage = weapon attack value (`wooden_stick` earth 4, `copper_dagger` air 8,
`iron_dagger` air 24 but level 10), reduced by resistance; fights have a round cap, so
insufficient damage-per-round means you survive but never close the kill ("turtle"). Better
weapons are level-gated (character level to equip + crafting-skill level to craft). So higher
bins require BOTH the counter-element AND leveling (character HP + crafting skill), not gear at L1.

| bin | goal (capability milestone) | resist / gate | ways (multi-task per bin) |
| --- | --- | --- | --- |
| 0 anchor (scripted) | kill chicken (L1) + cook | none; stick earth 4 works | harness-executed; positive control |
| 1 | beat yellow_slime (L2, 70hp) | earth-resist -> craft copper_dagger (air 8), doable at L1 | craft-air / grind-out-level / gear+cook |
| 2 | beat green_slime (L4, 80hp) | air-resist -> dagger now resisted, L1 gear too weak -> MUST level | level+craft counter-element / out-level / armor+consumable |
| 3 | beat blue/red slime (L6/7) or cow (L8) | multi-element + high HP | full chain / grind+gear / mixed |

Bin 1 is element-only (pure planning at L1); bins 2-3 fuse element-match with leveling. The
"out-level" way is the deliberate low-planning path in each bin -- the guard so difficulty
cannot be read off activity type (see 4.3 rationale). Frontier calibration (4.5) finds each
model's ~50% bin empirically -- that is exactly where the level x gear gate bites for that model.

**Preamble (bin 0, harness-executed, NOT in model context):** move to a chicken, fight it, cook
the `raw_chicken`, heal -> every measured episode starts fed/armed at a known L1 baseline.

## 9. Build sequence (PR plan)

1. This spec (finalized decisions + ladder).
2. Externalized ReAct prompt strategy (thinking off, bounded think->act step) + a Qwen3
   no-think decode flag.
3. v2 task ladder as data + the harness-executed preamble (state setup, not in context).
4. Walking-skeleton smoke test: one task, a few stochastic episodes, end to end.
5. Stochastic per-episode runner (temperature 0.7, N episodes/task) + per-episode first-plan
   metric + episode-outcome labels.
6. Grouped CV in the evaluator (group by episode for correctness, by task for difficulty).
7. Frontier calibration pass + full runs (8B then 14B) + the internal-CoT baseline subset.
