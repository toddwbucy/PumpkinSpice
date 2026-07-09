# Floor-test v2 design (issues #7 / #8)

Status: v2 spec, revised after the PR #21 review (mechanics re-grounded in the repo's own
artifacts; internal contradictions resolved). ASCII only.

Scope: PumpkinSpice is the conventional-RAG *control* (Benchero 2.0 Agent 2) and, per the
operator directive, a general model-baselining / introspection testbed. This floor test is
the *general-baselining / introspection* use: it is ICL-free and conventional-RAG (the model
decides when to retrieve). It is NOT a Benchero scored run; the fairness contract for those is
separate. In-context demonstration (ICL) is explicitly out of scope here and deferred to
WeaverTools (see decision 5).

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
- kill3 length baseline: reasoning->planning geometry 0.154 vs length 0.823;
  planning->reasoning geometry 0.448 vs length 0.667 -- LENGTH transfers, geometry does not.

14B replication (reasoning arm): kill2 0.600, length 0.617 (geometry does not beat length);
difficulty length 0.754 > geometry 0.683; #8 range 1.17 (still structured). The #7 signal
does not strengthen with scale; #8 holds.

## 3. The two confounds v1 exposed

### Confound A -- per-run labeling (task identity)
`make_label_fn` stamps ONE `eventual_correct` and ONE `hard` on all ~100 turns of a tier.
With only ~5 tiers, any probe that "predicts correctness/difficulty" actually learns "which
task this is." The planning kill2 (0.978) is yellow_slime-vs-rest discrimination. The
effective independent-unit count for the agentic arm is ~5, not ~500.

### Confound B -- generation length (thinking budget)
In thinking mode the model chooses its own generation length, spending more tokens on
hard/failing problems. `d_rho` and the kinematics are length-sensitive, so a geometry AUC
can be a length artifact -- confirmed per-cell by the length control (section 2).

## 4. v2 design

### 4.1 Externalize the reasoning loop (the key change)
Turn OFF the model's internal thinking (Qwen3 `enable_thinking=false`); drive reasoning as
bounded harness steps (ReAct: observe -> plan/act -> observe). This: controls generation
length (uniform, bounded steps) so geometry cannot hide behind thinking budget; keeps task
capability (the loop still reasons, visibly); and collapses compute for this arm.

"Reasoning location" is an independent variable: internal-CoT vs harness-externalized, same
tasks/models. **The on/off contrast is on each arm's geometry-BEYOND-LENGTH marginal (the
length-control's combined-minus-length), NOT raw geometry** -- the two arms differ ~10-20x in
trajectory length by construction (decision 6), so a raw geometry difference is confounded
with length. Both arms therefore always run through the length control (4.4).

### 4.2 Per-episode metric
One trajectory-metric per EPISODE, from the FIRST planning turn's trajectory -- a single
generation, matching the reasoning arm's one-generation-one-trajectory structure, ~cheap to
replay. Decode runs the full episode (the outcome is episode-terminal); replay is on the one
first turn. The first turn is measured at span="output" so prompt length cannot leak into the
trajectory. The label is the episode's eventual outcome (4.3).

### 4.3 Deconfound the labels (Confound A)
- Correctness/transfer: MANY stochastic episodes per task (temperature > 0, distinct recorded
  seeds), each labeled by its OWN outcome, so correctness varies WITHIN a held-constant task.
- Difficulty: MULTIPLE distinct tasks per hard/easy bin so d_rho must find a difficulty
  signature that generalizes across tasks, not memorize one. Target >= 5 distinct tasks per
  bin for leave-one-task-out task-grouped CV. NOTE: v1's aspirational "10-15 per bin" is
  revised down to what the HeroBench monster/craft roster supports (enumerated in build
  step 3); if the roster caps it below ~5, that is a documented limitation on kill1's
  cross-task generalization power, not a silent downgrade.
- Grouped cross-validation: group by episode (correctness) / by task (difficulty) so no unit
  leaks across folds; report the effective independent-unit count, not the turn count.

### 4.4 Deconfound length (Confound B)
The evaluator's length control (shipped, PR #20) reports per probe geometry vs length-alone
vs combined, span-aware, with a deterministic single-feature length AUC and a kill3 length
baseline; `marginal` is a noisy point estimate (no CI). Every v2 run reads through it, and
the reasoning-location contrast (4.1) is defined on the marginal, not raw geometry.

### 4.5 Frontier calibration
Within-task outcome variation only exists near ~50% success, and the frontier is
model-specific. Use the FIXED task pool (all bins x ways, ~20 tasks) on every model and
select the near-50/50 subset per model for the correctness/transfer cells: comparable pool +
per-model balance. Cheap calibration pass (a few episodes/task/model to estimate success rate).

## 5. Statistical power / compute

AUC reliability tracks the minority-outcome count. For a defensible 0.70 keep/kill call,
target ~300-400 balanced episodes per arm per model (95% CI ~ +/-0.065).

- **Externalized arm (the keep/kill arm):** short generations -> cheap decode and near-free
  replay (short traces, so the O(T^3) eig is trivial). ~360 episodes/model is a few hours.
- **Internal-CoT arm (baseline, KEPT per decision 3):** long-thinking generations (T ~ 8-16k)
  -- exactly the regime whose O(T^3) eig dominated v1 replay. This arm still pays the slow
  replay; it is mitigated by the split-across-GPUs + on-GPU eigendecomposition (both proven
  this cycle) but is NOT "minutes". Its decode + slow replay is a real, separate budget line.
  It is a POWER-LIMITED, DIRECTIONAL comparison: ~120 episodes (95% CI ~ +/-0.11), read for
  the SIGN of the on/off effect on the geometry-beyond-length marginal, paired against a
  MATCHED externalized 120-episode subset (same tasks) -- not a standalone keep/kill.
  (Correction of the earlier draft, which wrongly said replay is "minutes" and the
  GPU-reduction is "not needed"; it is needed for this arm.)

## 6. Carries over vs new

Carries over: the harness, replay driver, geometry, evaluator core (incl. the shipped length
control), #8, dtype/embed provenance, the reasoning (MATH) arm.

New work: (a) an externalized ReAct prompt strategy + a Qwen3 no-think decode flag;
(b) the multi-task HeroBench ladder as data (8.1); (c) a stochastic per-episode decode runner
that records seed+sampler per episode; (d) a won-fight episode-outcome detector (8.2);
(e) grouped CV + per-episode aggregation in the evaluator; (f) any missing world-client verb
(e.g. `rest`) and any missing corpus mechanic node the ladder depends on (8.3).

## 7. Success criteria

- If, at controlled length and with per-episode labels, the geometry-beyond-length marginal
  is reliably > 0 AND kill2/kill3 clear 0.70 across models -> the geometry carries a real,
  general signal.
- If `combined ~ length` everywhere -> the v1 signal was thinking budget; the metric is a
  length proxy, to be dropped or redefined length-invariant.
- Reasoning-location (on/off) is read on the marginal: does thinking-on add signal beyond
  length that externalized does not?
- #8 (depth structure) is evaluated separately and has held clean across 8B/14B.

## 8. Resolved decisions (operator-approved)

1. **Episodes/arm:** ~360 externalized (the keep/kill arm), distributed across the ladder
   tasks with the frontier subset weighted for outcome balance; internal-CoT baseline ~120,
   paired against a matched externalized 120-subset (directional, see section 5).
2. **Task ladder:** capability-milestone goals (8.1). The scored goal is "won a fight vs
   monster M" (8.2), NOT an XP level and NOT the monster's drop.
3. **Keep the internal-CoT arm** as the reasoning on/off baseline (directional; section 5).
4. **Representative step = the FIRST planning turn** (span="output").
5. **ICL DEFERRED to WeaverTools.** v2 baseline is ICL-free: harness sets state, model plans
   cold; no worked example in context (would confound geometry + trivialize correctness).
6. **Context: bounded, NOT max.** The measured first-plan turn's context is `state +
   instruction + available tools` -- retrieval is MODEL-INITIATED (a conventional-RAG MCP
   tool call, per the hard constraint "the model decides when to retrieve"), so retrieval
   results are NOT injected into the first-turn context; they appear only in later turns.
   Generation cap differs by arm (externalized ~512-1024, internal ~8-16k) -- that difference
   IS the on/off contrast, which is why the contrast is read on the length-controlled marginal
   (4.1). **Both arms use the SAME sampling temperature** (so reasoning-location is not
   confounded with sampling entropy); seed+sampler are recorded per episode. vLLM
   `--max-model-len ~32768`. Episode-*play* uses bounded history (in-context working memory,
   not ICL); on a failed action the agent retries WITH the failure in context (ReAct/replan),
   no reload; the *measurement* (first turn) is pre-failure and small.

### 8.1 The concrete task ladder (difficulty = level-gap x counter-element)

Mechanics, grounded in the corpus (the agent's only knowledge source) and the Data files:
- A fight is WON or LOST based on the character's LEVEL and equipped WEAPON vs the monster's
  level and attacks; an under-leveled character loses repeatedly to a monster several levels
  above it; a lost fight yields 0 XP / 0 drops and is still an HTTP success
  (`corpus.py` `mechanic:combat_risk`). So "the action succeeded" != "the fight was won".
- Weapons carry an element + attack value and a crafting-skill level to make
  (`items.json`, `corpus.py` render_item): `wooden_stick` earth 4; `copper_dagger` air 8
  (weaponcrafting 1, 48 copper_ore -> 6 copper -> craft); higher weapons need higher
  crafting-skill levels. (No character-level *equip* gate is asserted -- unverified.)
- Monster resistances build the ladder (`monsters.json`): chicken L1 no-resist; yellow_slime
  L2 earth-resist 25; green_slime L4 air-resist 25; blue/red slime L6/7; cow L8.

Difficulty is therefore a joint gate of level-gap and counter-element: a small gap + the
right element is winnable low; a large gap loses regardless of gear, so leveling is required.

| bin | goal: win a fight vs | gate | ways (>= ~5 distinct tasks/bin, roster-limited; enumerated in build 3) |
| --- | --- | --- | --- |
| 0 (positive control) | chicken (L1) | none; starting stick works | measured, ~100% success; NOT scripted |
| 1 | yellow_slime (L2, gap 1) | earth-resist -> want air (`copper_dagger`); winnable near L1 | craft-air-weapon / level-then-fight / alt L2 monsters |
| 2 | green_slime (L4, gap 3) | air-resist (dagger now resisted) + larger gap -> MUST level + re-gear | level+craft counter-element / level-then-fight / alt L4 targets |
| 3 | blue/red slime (L6/7) or cow (L8) | large gap + high HP | deeper level + gear + consumables; alt high-tier targets |

Bin 0 is the positive control (any competent model wins). Bin 1 is element-only planning near
L1. Bins 2-3 fuse element-match with leveling (combat_risk: large gap loses regardless). The
"level-then-fight" way is the deliberate lower-planning path in each bin -- the guard so
difficulty cannot be read off activity type (4.3). Difficulty is a property of the TARGET
monster's bin, not the way; the way varies the approach, the episode outcome is won-vs-M.
No scripted preamble: the character spawns at a clean, uniform L1 baseline (stick equipped,
full HP, sap + ash_wood), and the measured first-plan turn is from that spawn.

### 8.2 Outcome scoring (won-fight-vs-M)

The per-turn capture records `outcome.data` = the full HeroBench action response
(`loop.py`; `world_herobench` sets `ActionResult.data = resp.json()`). A fight response
records its result (win/loss) and XP, independent of the ~5%-rate drop -- so an episode's
outcome = "did any fight vs monster M return a win". A new episode-outcome detector scans the
episode's turns for a won fight vs the goal monster (build step in section 9; the exact result
field is confirmed against a live fight response at build time). This is why the goal is
"won vs M" and NOT the drop (drop rate ~20 = ~5%/kill, per `bench_herobench`) and NOT an XP
level (v1 used `goal_level` only because it lacked a reliable kill signal; the win is in
`outcome.data`).

### 8.3 Corpus / action-surface prerequisites (fairness)
The RAG agent knows only what it can retrieve, so any mechanic the ladder depends on MUST be
a corpus node (else the ladder measures corpus coverage, not planning). `combat_risk`,
resistances, recipes, and `leveling_via_crafting` are present. Healing is via the `rest`
action (HP regen) and/or equipping a cooked consumable that auto-heals in a fight -- there is
NO "eat"/"consume" verb; build step 3 confirms `world_herobench` can issue `rest`/`equip` and
adds any missing verb. If the ladder relies on a mechanic not yet in the corpus, seeding it is
part of that build step.

## 9. Build sequence (PR plan)

1. This spec.
2. Externalized ReAct prompt strategy (bounded observe->plan->act) + a Qwen3 no-think decode
   flag (`chat_template_kwargs={enable_thinking:false}` via `extra_body`); same-temp for both
   arms.
3. HeroBench ladder as data (enumerate >= ~5 tasks/bin from the roster) + the won-fight
   outcome detector + any missing world-client verb (`rest`) + any missing corpus mechanic
   node; capture gains seed+sampler fields.
4. Walking-skeleton smoke test: one task, a few stochastic episodes, end to end (verify the
   won-fight detector against live fight responses).
5. Stochastic per-episode runner (same temperature both arms, recorded seed/sampler,
   N episodes/task) + per-episode first-plan-turn metric (span="output") + episode-outcome
   labels; grouped-CV wiring in the evaluator.
6. Frontier calibration pass (per-model ~50/50 subset selection).
7. Full runs: externalized 8B then 14B (keep/kill); internal-CoT matched 120-subset per model
   (directional on/off), on the slow replay (split + on-GPU eig).
