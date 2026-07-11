# Research Note: Learned auto-tuner / classifier for factorization (issue #110)

**Status:** Investigation complete — fill (§5.5), time/memory (§5.6), cheap-proxy race
(§5.7), and an end-to-end latched actual-time race prototype (§5.8). Findings: fill
selection is already solved; the time/memory objective has real headroom on large
matrices (§5.6) but no cheap proxy captures it (§5.7); the actual-time race captures only
~9 % over the already-good `Auto`, amortizes over a median ~206 same-pattern reuses (far
more than typical IPM), mis-picks on the transient first iterate, and ties `AutoRace` on
the stable tail (§5.8). **Net: the ordering-selection prize is real but marginal; do not
build a time-aware selector (race or model) without a specific high-repeat, large-matrix
workload.** (The AMF-in-AutoRace follow-up #111 was *withdrawn on review* — §5.7
corollary: it optimizes fill but regresses time on the largest matrices.) Deliverables: this note +
`study_ordering_gap` (fill) + `study_ordering_timemem` (time/memory + proxy race) +
`study_latched_race` (end-to-end latched race) binaries.
**Date:** 2026-07-02
**Related spec sections:** 5.1 (feature lifecycle); symbolic + numeric pipelines
**Key references:** issue #110; rslab (github.com/milanofthe/rslab) `src/auto_tune.rs`,
`src/tuning.rs`, `src/auto_tune_model_{ldlt,lu}.json`, `xtask/src/main.rs`,
`benches/train_tuner.py`; feral's own `Auto` dispatchers (`choose_adaptive`,
`pick_scaling_strategy`, `pick_ordering_preprocess`) and `AutoRace`.

## 1. What the issue proposes

Port rslab's "learned auto-tuner" idea to feral: a small ML model that, from a
matrix's cheap structural features, selects the solver configuration (ordering,
amalgamation, kernel gates, scaling, pivot threshold, memory mode) — guarded by a
deterministic memory backstop so it never uses more memory than the default, with
an out-of-distribution fallback to a deterministic exact-fill ordering race.
Trained on a complete-distribution corpus (curl-curl Maxwell, Stokes/KKT
saddle-point, convection-diffusion swept over grid-Péclet).

## 2. How rslab actually does it (verified against the repo)

rslab's tuner is **not** an end-to-end policy net that emits a config. It is a
**learned surrogate cost-model + explicit knob-grid search + deterministic
backstops**:

- **Model:** two small MLPs, one per factorization path (LDLᵀ and unsymmetric
  LU). The LDLᵀ net is a 3-layer FC `37 → 64 → 32 → 2`, ReLU hidden, linear out.
  It maps `(structural features ⊕ log-transformed knobs) → [log factor_ms,
  log peak_mb]` — a cost predictor, not a classifier.
- **Selection:** the tuner scores a candidate grid of configs through the net and
  picks the one minimizing `w·log(time) + (1−w)·log(mem)`, `w=0.7`. The knobs it
  varies are fed *into* the net as inputs, so the same net scores every candidate.
- **Inference is pure Rust, hand-rolled** — `struct Layer { w: Vec<Vec<f64>>, b:
  Vec<f64> }`, plain matmul+bias+ReLU. No candle/burn/tract/ONNX. Weights embedded
  at compile time via `include_str!` + `serde_json::from_str` (~2.5k f64s).
- **Three deterministic backstops (the important part):**
  1. *A-priori memory gate* — reject any candidate whose memory exceeds default by
     more than ~2% (`MEM_TOL_LN = 0.0198`); the real gate is computed from **exact
     symbolic fill + flops + a realistic memory floor**, not the MLP prediction, so
     "never more memory than default" holds *by construction* before any numeric work.
  2. *OOD fallback* — when `factor_flops > flops_ood_cap` the net is refused and the
     tuner returns `default().with_ordering(AutoRace)`, a deterministic exact-fill
     ordering race.
  3. *Deviation hysteresis* — must beat default by ≥8% (`MIN_GAIN`) to deviate,
     ≥20% to flip factorization method, calibrated to the machine's timing-noise floor.
- **Training is offline in Python.** `cargo xtask tune` runs sweep → `python
  benches/train_tuner.py` → hardware-calibrate → assemble → **held-out validate
  with a ship-gate that refuses to ship a model that regresses the default**.
  Weights ship as embedded JSON + a runtime-swappable `tuner_profile.json`.

The pattern to copy is the *discipline*: the ML only ever chooses among configs
that a deterministic layer has already proven are no worse than the default on the
guaranteed axis (memory), and it never extrapolates out of distribution.

## 3. feral's current state — the tuner would replace/augment this

### 3a. Configuration surface (knobs a tuner could set)
The public `Solver` is **symmetric LDLᵀ multifrontal only**; the unsymmetric LU
code (`src/lu/`) is a *separate* simplex-basis API with no auto LDLᵀ-vs-LU
dispatch. So a first tuner tunes the LDLᵀ `Solver`. Knobs (all in
`src/numeric/solver.rs` builders over `NumericParams` / `SupernodeParams` /
`BunchKaufmanParams`):

- **Ordering method** — `OrderingMethod` {Amd, Amf, MetisND, ScotchND, KahipND,
  Auto, AutoRace, External}. Default `Auto`.
- **Ordering preprocess** — `OrderingPreprocess` {None, LdltCompress, Auto=default}.
- **Ordering-escalation-growth** — `Option<f64>`, default `Some(1e24)`.
- **Scaling** — `ScalingStrategy` {InfNorm, Mc64Symmetric, Identity, External,
  Auto=default}.
- **Amalgamation** — `nemin` (default **16**), `AmalgamationStrategy`
  {Adjacency, Renumber, Auto=default}, small-leaf grouping.
- **Pivot threshold `u`** (LDLᵀ) — `bk.pivot_threshold`, effective default `1e-8`.
- **Kernel gates** — global FMA (`with_fma`), per-front FMA row gate
  (`with_fma_large_fronts`), BLAS-3 (internal, front-size gated),
  intra-front parallelism (internal, area-gated), block size 64, global
  parallel toggle + flop gate.
- **Stability** — delayed pivots, cascade-break (+ auto-arm), static-pivot
  threshold, SQD mode, MC64 cache.
- **Memory mode — DOES NOT EXIST.** No `MemoryMode`/`with_memory` knob; memory is
  managed structurally (contrib-block pooling). rslab's memory *knob* has no feral
  analogue, though its memory *backstop* (bound peak from symbolic fill) maps
  directly onto feral's `SymbolicFactorization.peak_contrib_bytes`.

### 3b. Current automatic selection (already sophisticated, hand-written)
- Ordering: `choose_adaptive` (cheap n/avg-deg/arrow-signature rules) or `AutoRace`
  (runs full symbolic on 4 orderings, keeps min `factor_nnz_estimate`).
- Preprocess: `pick_ordering_preprocess` + **verify-by-fill** (run both ways, keep
  `LdltCompress` only within 2× fill of `None` — issue #91).
- Runtime feedback: escalate to `LdltCompress` on pivot growth `>1e24` (#102);
  quality escalation bumps scaling then pivot-`u` on instability (Ipopt-style).
- Scaling: `pick_scaling_strategy` (arrow-head + slack-mass → Mc64 else InfNorm),
  sticky per pattern, with MC64 fallback policy.
- Amalgamation: `pick_amalgamation_strategy` (etree path-vs-bushy).

**These hand-written cheap-feature dispatchers are exactly rslab's target — feral
has independently built the deterministic-heuristic version of the same idea.** The
learned tuner is a natural evolution, not a greenfield capability.

### 3c. Features already computed (ready-made classifier inputs)
Cheap O(nnz) pattern scans the heuristics already do: `n`, `nnz`, `avg_deg`,
`diag_only/n`, `max_col_nnz`, low-degree-column fraction, arrow/bordered signature.
Rich symbolic outputs on `SymbolicFactorization`: per-front `ncol`/`nrow` shape
distribution, `factor_nnz_estimate`, `peak_contrib_bytes`, `etree`, `col_counts`.
Numeric diagnostics: pivot growth, inertia, `ScalingInfo`, `Mc64MatchStats`. This
covers ~all of rslab's 37-input feature vector except a couple of derived
tree-width / arithmetic-intensity terms that are trivial to add.

### 3d. Training infrastructure that already exists
`crates/feral-diagnostics/src/bin/bench_orderings.rs` already sweeps 4 orderings
per matrix recording fill+time; `bench_solver_corpus.rs` walks a corpus by family;
`bench_one_matrix.rs` emits a per-matrix sidecar (n, nnz, inertia, analyse/factor/
solve µs, rel_res, status) — a ready template for a training-label record. feral
has a 153k-matrix KKT corpus plus the MUMPS/SSIDS oracles under
`external_benchmarks/`. Label generation (best config per matrix) is a sweep
extension, not new infrastructure.

## 4. Feasibility under feral's hard constraints

- **Zero non-Rust deps / pure Rust / stable toolchain** — ✅ trivially met.
  rslab's own inference is a hand-rolled MLP + `serde_json` + `include_str!`; feral
  already depends on `serde`. No BLAS/ML crate needed. Offline training in Python
  is out-of-tree (a dev tool, like the existing `scripts/*.py`), weights ship as an
  embedded JSON artifact.
- **Correctness before performance / exact inertia** — ⚠️ **the governing
  constraint, and where feral differs from rslab.** feral's headline guarantee is
  *exact certified inertia*. A tuner must be scoped so it can only ever change
  *performance*, never *which inertia is reported*:
  - **Inertia-neutral knobs** (safe for ML to pick freely): ordering method,
    ordering preprocess, amalgamation (`nemin`/strategy), all kernel gates
    (FMA/BLAS-3/parallelism/block size), parallel toggle, MC64 cache. These change
    fill/FLOPs/scheduling but not the eliminated pivot sequence's signs. *Caveat:
    FMA and reduction order can flip a borderline pivot's Bunch-Kaufman
    classification at the rounding boundary (see `tried-and-rejected.md` FMA
    inertia-flip entries) — so even "neutral" kernel gates must be validated
    against the inertia gate, not assumed free.*
  - **Inertia-sensitive knobs** (ML must NOT own these, or must be backstopped by
    the existing deterministic escalation): scaling strategy and pivot threshold
    `u` change which pivots are accepted and can flip inertia on near-singular /
    borderline KKTs — precisely the matrices #102/#63 already handle with
    deterministic feedback loops. The safe design keeps scaling/`u` under the
    existing deterministic escalation and lets ML tune only the performance axis.
  - **Backstop analogue:** rslab's "never worse than default on memory" becomes
    feral's "never wrong on inertia and never worse than default on the corpus
    inertia gate." The `AutoRace` OOD fallback already exists deterministically.
- **Clean-room / no architectural dependency on references** — ✅ the idea is
  reimplemented from the design, not rslab's code (MIT, but we build our own).

## 5. Assessment & recommended scope

**Verdict: feasible and well-matched to feral, but the honest first increment is
narrow and measurement-driven, not a full multi-knob tuner.**

feral already *has* the deterministic version of this system (§3b) and it is
well-tuned. The learned tuner earns its keep only where the hand-written
dispatchers demonstrably leave performance on the table. The two places most
likely to pay off, in priority order:

1. **Ordering selection as a learned surrogate for `AutoRace`.** `AutoRace` runs
   full symbolic 4× to pick the min-fill ordering — accurate but expensive. A tiny
   surrogate that predicts `factor_nnz_estimate` (or factor µs) per ordering from
   cheap features, picking the argmin *without* running the race, is:
   (a) strictly inertia-neutral (ordering never changes inertia, modulo the FMA
   caveat which ordering doesn't touch), (b) backstopped trivially (fall back to
   `AutoRace` when features are OOD or predictions are close), (c) directly
   measurable against `AutoRace`/`Auto` on the existing corpus, and (d) reuses
   `bench_orderings` for labels. This is the lowest-risk, highest-clarity slice.

2. **Amalgamation `nemin` / kernel-gate tuning** — also inertia-neutral, but the
   payoff is smaller and the BLAS-3/FMA gates are already shape-dispatched.

Explicitly **out of scope for a first cut:** letting ML own scaling or pivot-`u`
(inertia-sensitive; keep deterministic), and any unsymmetric-LU model (separate
API, no auto-dispatch to hang it on yet).

**Recommended next step (if we proceed):** a focused *measurement study first* —
extend `bench_orderings` to dump per-matrix (features, per-ordering fill+time) over
the corpus, and quantify how often `choose_adaptive`'s cheap pick differs from the
`AutoRace` optimum and by how much. If the gap is small, the deterministic
heuristics already capture most of the value and a learned model is not worth the
maintenance surface. If the gap is material, we have both the motivation and the
labeled corpus in hand, and Phase 2 is the pure-Rust surrogate + ship-gate
(rslab's "never regress the default" discipline, adapted to the inertia gate).

## 5.5. Measurement results (the gating study — ran 2026-07-02)

Built `crates/feral-diagnostics/src/bin/study_ordering_gap.rs`: for one
representative matrix per family under `data/matrices/kkt` (fill depends only on
the pattern, shared across a family's IPM dumps), measure `factor_nnz_estimate`
under AMD / AMF / MetisND / ScotchND / KahipND / `Auto` / `AutoRace`, all with the
default preprocess (apples-to-apples; ordering is the only variable). **568 family
representatives** measured (of 705 family dirs; 135 store their `.mtx` in nested
subdirs the collector doesn't descend into — a coverage gap, not a bias, since it's
purely a directory-layout artifact). `oracle_best` = min fill over the 5 explicit
methods.

Headline numbers:

| metric | value |
|---|---|
| `Auto` == oracle optimum | **489/568 = 86.1 %** |
| `ratio_auto` = fill_auto / oracle_best | geomean **1.012**, p50 1.000, p90 **1.014**, p99 1.287, max **1.608** |
| families where `Auto` loses >2 % / >10 % / >50 % fill | 56 / 22 / 4 |
| `AutoRace` == oracle optimum | **546/568 = 96.1 %** |
| `ratio_race` = fill_autorace / oracle_best | geomean 1.0015, p90 1.000, max 1.115 |
| `AutoRace` symbolic time vs `Auto` | **8.95× slower** (292 ms vs 33 ms total) |

Two findings that change the recommendation:

1. **The entire meaningful fill tail is already recovered deterministically by
   `AutoRace`.** For *every one* of the 22 families where `Auto` loses >10 % fill,
   `ratio_race = 1.0000` — `AutoRace` hits the oracle. So a learned model is **not
   needed for reachable quality**: the deterministic system already reaches the
   optimum on the tail. The only thing ML could add is reaching that quality at
   `Auto`'s cost instead of `AutoRace`'s ~9× symbolic cost — and symbolic is done
   *once per pattern* and amortized across up to 3000 numeric factors in the IPM
   reuse workload, so that 9× is near-zero in the setting that dominates feral's
   corpus. The cost argument for ML is therefore weak.

2. **The one real deterministic gap is ML-free to close.** `AutoRace` misses the
   oracle on exactly 22/568 (3.9 %) families — and those are exactly the 22 where
   `Auto` *beats* the race, because `choose_adaptive` knows AMF and the `AutoRace`
   set {AMD, MetisND, ScotchND, KahipND} does **not** race AMF. `{AutoRace ∪ AMF}`
   equals the 5-method oracle by construction, so **adding AMF to the `AutoRace`
   race set makes the deterministic race oracle-optimal on 100 % of these families**
   — a trivial, deterministic, inertia-neutral change that dominates what a learned
   ordering surrogate could achieve on quality. (Tail families losing the most:
   VESUVIOU/VESUVIA n=3083 at 1.61×/1.59×, GAUSS2 n=758 at 1.58×, HAHN1 n=715 at
   1.56× — all fully recovered by AMD/ND methods the race already runs; the
   remaining Auto-only wins are AMF picks.)

### 5.5.1. Caveat — this study optimizes *fill*, which under-tests the real question

The study scores orderings by **fill** (`factor_nnz_estimate`), but fill is only a
*proxy* for what actually matters — factor wall-clock and peak memory. So the
"recommend against a learned ordering tuner" conclusion holds **narrowly**: for
ordering selection *scored by fill on KKT matrices*, not for the time/memory
objective a learned tuner is actually built around. `86 % fill-optimal` says nothing
about *time*-optimality.

Fill is blind to: dense-front work (∝ Σ `ncol·nrow²`, not linear in fill); achieved
BLAS-3 GFLOP/s (fat fronts run near peak, tiny fronts scalar — *more* fill in fat
fronts can be *faster*); parallel critical-path / etree balance; and peak transient
memory (`peak_contrib_bytes` can exceed `nnz(L)`). feral has a **documented
counterexample**: nql180 (`src/symbolic/mod.rs:311`,
`dev/research/issue-73-n100k-thin-regime.md`) has smaller fill *and* smaller
flop-proxy under MetisND (`nnz_L` 0.98×, flop 0.86×) yet AMF is **2.05× faster** on
real factor+solve (1.90 s vs 3.95 s) — "fill … is NOT a reliable speed predictor."

So a fill-optimal ordering can be materially time-suboptimal, and the nql180-type
divergence is precisely the regime where a **time** objective (and hence a learned
surrogate) could beat fill-based selection — a regime this fill study is structurally
incapable of seeing.

Why this does not automatically revive #110: feral already injects the time signal,
just not with an online model — it runs offline wall-time A/Bs per structural regime
and bakes the result into deterministic reroutes (thin-large / would-be-MetisND →
AMF, unconditionally, knowingly "wrongly demoting nql180" because AMF wins the whole
population). The real design axis is therefore not *fill vs time* but *how the time
signal gets in*: an online per-matrix surrogate (rslab) vs offline-measured discrete
structural branches (feral). feral's discrete branches win when the fill↔time
divergence clusters into a few nameable regimes; they lose to a learned time-surrogate
only when the divergence is high-dimensional and matrix-specific, with no clean branch
to capture it. **The correct gating experiment (superseding this fill study): predict
factor wall-clock and peak memory from cheap features across a distribution wider than
KKT, and measure how much a continuous model beats feral's discrete regime-branches on
the residual nql180-like cases.**

## 5.6. The time/memory data (ran 2026-07-02 — the experiment §5.5.1 called for)

Built `crates/feral-diagnostics/src/bin/study_ordering_timemem.rs`: for each of the
5 explicit orderings plus `Auto`/`AutoRace`, measure **numeric factor wall-clock**
(min-of-reps, symbolic reused — the IPM per-iteration cost) and a **peak-memory
model** (`16·nnz_l` stored factor + `peak_contrib_bytes` transient stack). Ran three
corpora: the 568 small KKT family reps, the 10 large matrices in `tests/data/large`
(n up to 345 k), and the 41/48 `kkt-mittelmann` thin-large families (n up to 260 k —
the nql180 / dtoc2 / pinene regime). Restricted to matrices whose fastest ordering
takes ≥ 50 µs (below that, timing is pure overhead).

| corpus (meaningful subset) | N | fill-best = time-best | time-regret of fill-pick (geo / max) | mem-regret (geo / max) | time-oracle vs `Auto` | vs `AutoRace` |
|---|---|---|---|---|---|---|
| KKT reps | 33 | 24 % | 1.02 / 1.13 | 1.01 / 2.12 | 1.02× | 1.02× |
| large (`tests/data/large`) | 10 | 30 % | **1.16 / 1.60** | 1.01 / 1.07 | **1.52×** | 1.06× |
| mittelmann (thin-large) | 41 | 22 % | **1.14 / 3.61** | **1.13 / 3.90** | **1.43×** | **1.12×** |

What the fill study missed, now visible:

1. **Fill is a bad speed predictor on big matrices, confirmed at scale.** Picking the
   min-fill ordering leaves ~14–16 % factor time on the table (geomean) on
   large/thin-large, up to **3.6×** on a single matrix (svanberg n=100 k: min-fill AMD
   is 3.61× slower than SCOTCH). The tiny KKT reps hid this (~2 %) purely because they
   are too small for the front-shape / BLAS-3 effects to matter.
2. **The objective compounds — this reverses the amortization argument.** Selection
   cost (AutoRace's 9× symbolic) amortizes away over IPM reuse, yes — but the thing
   being optimized here, *numeric factor time*, is paid on **every** one of the up-to-
   3000 iterations. A 10–15 % better ordering choice saves 10–15 % on every factor.
   So picking the ordering once (even expensively) to get a faster numeric factor
   thousands of times is exactly the trade feral's architecture rewards.
3. **Neither deterministic path dominates.** On mittelmann, `Auto` is faster on 18
   families, `AutoRace` on 22 (1 tie) — they win in different regimes (AutoRace's
   fill-race avoids the ND-blowup disasters like qap15's 9× and c-big's 5×; `Auto`'s
   thin-large→AMF reroute helps where the fill-min is slow). A perfect per-matrix
   **time-oracle beats the default `Auto` by 1.43–1.52×** and still beats the better
   `AutoRace` by **6–12 %** on large matrices. And both deterministic paths *jointly*
   miss cases: svanberg is min-fill for AMD, so `Auto` and `AutoRace` both pick the
   3.6×-slow ordering — only a time-aware selector escapes it.
4. **Memory is a separate, larger opportunity on thin-large.** Min-fill orderings use
   up to **3.9×** the peak memory of the memory-optimal ordering (dtoc2 n=104 k: AMF
   min-fill vs AMD), geomean +13 %. Fill and `AutoRace` optimize neither time nor
   memory directly, so a memory-constrained caller on this regime is badly served today.

**Interpretation.** There *is* real, non-amortizable headroom — ~1.4–1.5× total factor
time over the default `Auto`, ~10 % over `AutoRace`, plus up-to-3.9× peak-memory swings
— but it is (a) concentrated on large / thin-large matrices (negligible on small KKT),
and (b) *mostly* reachable deterministically. The natural next move is a
**time/memory-aware race** — race the candidates and select on numeric factor time /
peak memory instead of on fill. Whether that race can use a *cheap* proxy (no numeric
factoring) is the pivotal question, tested in §5.7.

## 5.7. Prototype — can a *cheap* deterministic time-race close the gap? (No.)

`study_ordering_timemem` also records two cheap symbolic *time* proxies per ordering
(no numeric work): `flop_proxy = Σ ncol·nrow²` (dense-front work) and `max_front`
(largest frontal dimension). If racing on one of these picked orderings near the
time-oracle, a deterministic proxy-race would capture the §5.6 headroom with no model
and no numeric factoring. It does not. On the mittelmann thin-large corpus (41
meaningful matrices), racing each proxy and picking its argmin ordering, per-matrix
time-regret vs the oracle:

| proxy raced on | total vs oracle | per-matrix regret (geo / p90 / max) |
|---|---|---|
| **fill** (what `AutoRace` uses) | **1.13×** | **1.134** / 1.33 / 3.60 |
| `flop_proxy` (Σ ncol·nrow²) | 1.26× | 1.198 / 1.90 / 3.60 |
| `max_front` | 1.28× | 1.208 / 1.94 / 3.60 |

**No cheap symbolic proxy reaches the oracle, and `fill` is the *best* of the three** —
`flop_proxy` and `max_front` are *worse* time predictors than fill. This is issue-73's
nql180 observation confirmed at corpus scale: the mapping from symbolic structure to
numeric factor time is not captured by any single cheap scalar. (On the large set the
totals are c-big-dominated and within timing noise, but the same ranking holds: flop
never beats fill.) **Corollary that reverses #111 (reviewed 2026-07-02):** because
`fill` disagrees with *time* and AMF is a low-fill-but-sometimes-slow ordering, adding
AMF to the fill-selecting `AutoRace` set is *not* the clean win #111 claimed. Measured
across 51 large/thin-large matrices, adding AMF changes `AutoRace`'s pick to a **slower**
ordering on **8** of them (c-big 1.60×, dtoc2 1.43×, dirichlet120 1.12×,
ex1/ex4/ex42_160 1.11×, bcsstk38 1.09×, cont5_1_l 1.02×) versus **6** where it helps
(r05 0.75×, rocket 0.78×, dtoc1nd 0.80×, bratu3d 0.86×); 37 unchanged. The two worst
regressions are the largest, most expensive matrices — exactly where `AutoRace` is used
— while #111's fill "wins" (VESUVIOU n=3083, GAUSS2 n=758, HAHN1 n=715) are all small
matrices where fill barely affects time. So #111 trades fill on matrices where it is
irrelevant for a time-regression risk where it is critical. It optimizes fill (the wrong
metric) and can regress time (the right one); recommend closing it rather than merging.
A fill-selecting race cannot be "fixed" by adding candidates — the selection criterion,
not the candidate set, is the limitation, and §5.8 already showed time-aware selection
isn't worth building.

**So a cheap deterministic time-race is off the table.** That leaves exactly two ways
to capture the §5.6 time headroom:

1. **Actual-time race, latched per pattern.** Factor the k candidate orderings once
   each, measure real factor time (and peak memory), keep the winner, and latch it for
   the pattern — feral already has per-pattern latch machinery (the `ordering_escalated`
   path). This reaches the oracle *by construction*. Its cost is k−1 extra numeric
   factors paid **once per pattern**, then amortized over the up-to-3000 IPM reuse
   iterations — i.e. ~0 in feral's dominant workload. Deterministic, correctness-
   preserving, no ML. **This is the recommended way to capture the headroom.**
2. **A learned multi-feature model** (rslab-style). The *only* way to beat the cheap
   single proxies without paying for actual factoring — a model over many features
   (front-shape distribution, tree width, arithmetic intensity, …) rather than one
   scalar. This is where #110's learned tuner would genuinely earn its keep, but it is
   justified over option 1 **only when the actual-time race is unaffordable** — i.e.
   one-shot / non-reuse solves, which is *not* feral's dominant IPM workload.

The §5.7 result therefore *appeared* to sharpen #110 to a crisp decision — an
actual-time race latched per pattern. But building that race end-to-end (§5.8) walked
the "amortizes to free" claim back substantially.

## 5.8. End-to-end prototype of the latched actual-time race (ran 2026-07-02)

Built `crates/feral-diagnostics/src/bin/study_latched_race.rs` — the actual-time race,
driving the public `Solver` API only (no core change; ordering choice is inertia-neutral
so there's no correctness exposure). For each mittelmann family's real IPM iterate
sequence it factors the full grid (5 orderings × iterates), plus `Auto` and `AutoRace`,
picks the ordering fastest on the first factor, "latches" it, and compares steady-state
per-iterate cost. Crucially it measures the steady state over **pattern-reused iterates
only** — real IPM sequences change pattern at the first iterate or two (the active set
settling: e.g. nql180 nnz 874 500 → 939 300 at iterate 1, then stable), and only the
reused tail is the long-run cost.

Results (29 families, iterates capped at 10):

- **Steady-state headroom is real but modest: the latched winner is 9.3 % faster per
  iterate than `Auto`** (geomean), up to ~40 % on specific large families (henon120
  0.59×, optmass 0.72×, cont5_2_2_l 0.61×) — but *slower* than `Auto` on several
  (dtoc1nd 1.20×, lane_emden120 1.27×) where the pick was wrong. High variance.
- **Versus `AutoRace` it is ~0 % (0.7 %).** On the pattern-stable tail `AutoRace` reuses
  its raced choice correctly, so it already matches the latched winner. (An earlier draft
  of this study saw `AutoRace` as ~79 % slower; that was an artifact of averaging in the
  one iterate where the pattern changes and `AutoRace` re-runs its whole symbolic race.
  Corrected: `AutoRace` is *not* pathological under stable reuse.)
- **Two obstacles kill the naive "race iterate 0, latch" plan:**
  1. **Representativeness is only 38 %.** Racing on the *transient* first iterate picks
     the steady-state-optimal ordering just 38 % of the time, because iterate 0's pattern
     differs from the stable pattern. (Fixable in principle: a per-*pattern* latch would
     race on the first occurrence of the *stable* pattern, not the transient one — but
     the prototype exposes that "race on the first factor you see" is wrong.)
  2. **Overhead needs ~200 iterations to amortize.** The k−1 extra first-factors cost a
     **median 206 stable iterations** to pay back at the 9 % per-iterate saving, and
     **11/29 families never amortize** (the latch wasn't faster than `Auto`). Typical IPM
     runs 20–50 iterations — well short. The "amortizes to ~0" claim in §5.7 assumed
     up-to-3000 reuses of one pattern; real single-solve sequences are far shorter, so the
     5-factor race overhead usually does *not* pay off.

**What this changes.** The actual-time race is not the free win §5.7 implied. Its
steady-state prize over the already-good `Auto` is ~9 % (concentrated on large problems),
its overhead amortizes only across hundreds of same-pattern factors (not typical IPM),
and it matches — not beats — `AutoRace` on the stable tail. And note the reversal it
implies: because the *race overhead* is the killer, a **learned predictor** (zero
inference cost, picks on the stable pattern with no trial factors) is actually the *more*
viable vehicle for capturing this specific ~9 % than the deterministic race is — the
opposite of §5.7's lean. But ~9 % on a subset of large problems is a modest prize either
way; `Auto` is already within it.

**Honest bottom line across §5.5–5.8.** Fill selection is solved (§5.5). The time/memory
objective has real headroom on large matrices (§5.6). No cheap proxy captures it (§5.7).
The actual-time race captures ~9 % over `Auto` but only amortizes over hundreds of
same-pattern reuses and mis-picks on the transient iterate (§5.8). Net: the ordering-
selection prize for feral is real but small (~9 % on large problems) and awkward to
capture; `Auto`/`AutoRace` are already close; neither a deterministic race nor a learned
model is clearly worth its cost for feral's typical IPM iterate counts. Even the one
follow-up this investigation spun off — #111, add AMF to `AutoRace` — was *withdrawn on
review*: it optimizes fill but regresses factor time on the largest matrices (§5.7
corollary), because the limitation is `AutoRace`'s fill *selection criterion*, not its
candidate set. Net action items from #110: **none to the core solver.** A time-aware
selector — race or model — is justified only by a specific high-repeat, large-matrix
workload that does not currently exist in the corpus.

## 6. Risks / open questions

- **Does the deterministic `Auto` already leave enough on the table to justify a
  model?** **Answered by §5.5: no, not for ordering.** `Auto` is oracle-optimal on
  86 % of KKT families and within 1.4 % on 90 %; the entire >10 % tail is already
  recovered by the deterministic `AutoRace`, and the one real deterministic gap
  (AMF absent from the race set, 3.9 % of families) closes with a one-line change,
  no ML. A learned ordering surrogate would only trade the race's amortized-away 9×
  symbolic cost — a weak motivation. The other candidate knobs (amalgamation,
  kernel gates) were not measured here and remain open, but ordering was the
  strongest a-priori case and it did not survive measurement.
- **Corpus completeness / distribution shift.** rslab needed 90 conv-diff problems
  to lift held-out R² 0.75→0.98 on the LU path. feral's corpus is KKT-heavy; a
  model must be validated as not regressing on non-KKT structure, and OOD fallback
  must be the default posture, not the exception.
- **Maintenance surface vs. a hard-real project.** An embedded weight artifact + an
  offline trainer + a sweep harness is real ongoing surface. It must be gated on a
  demonstrated, corpus-measured win, and shipped behind a deterministic fallback so
  a stale/bad model can never break correctness or regress the default.
- **Determinism.** Inference must be bit-reproducible (fixed weights, deterministic
  matmul order) to keep feral's byte-exact posture; straightforward with f64 and a
  fixed evaluation order.

## 7. Conclusion

The idea maps cleanly onto feral's architecture and constraints — pure-Rust
inference is trivial, the feature vector and a labeled corpus already largely
exist, and feral has independently built the deterministic-heuristic ancestor of
rslab's tuner. feral's correctness-first / exact-inertia guarantee means any learned
tuner may own only inertia-neutral performance knobs (ordering, amalgamation, kernel
gates — the latter still validated against the inertia gate), while scaling and
pivot-`u` stay under the existing deterministic escalation.

**But the gating measurement (§5.5) says do not build a learned ordering tuner
now — scored by fill.** On 568 KKT families the cheap `Auto` heuristic is
fill-optimal 86 % of the time and within 1.4 % on 90 %; the entire >10 % fill tail is
already recovered by the existing deterministic `AutoRace`; and the one genuine
deterministic gap (`AutoRace` doesn't race AMF, 3.9 % of families) closes with a
one-line change, not a model. A learned surrogate would only recover `AutoRace`'s
~9× symbolic cost, which is amortized to near-zero across the IPM's pattern-reuse — a
weak payoff for a weight-artifact + offline-trainer + sweep-harness maintenance
surface.

**Read that conclusion narrowly (see §5.5.1).** The study optimizes *fill*, a proxy
that provably diverges from factor wall-clock and peak memory — feral's own nql180
case has smaller fill under MetisND yet AMF is 2.05× faster. So "86 % fill-optimal"
does not imply "time-optimal."

**The time/memory measurement (§5.6) bears this out and materially changes the
picture.** On large / thin-large matrices, picking the min-fill ordering leaves
~14–16 % factor time on the table (geomean, up to 3.6× on one matrix) and up to 3.9×
peak memory; a perfect per-matrix time-oracle beats the default `Auto` by 1.43–1.52×
and the fill-racing `AutoRace` by 6–12 %; and — unlike selection cost — this
numeric-factor-time gain is paid on every IPM iteration, so it does **not** amortize
away. Neither deterministic path dominates (Auto and AutoRace split 18/22 on
mittelmann; both miss svanberg). So there *is* real, non-amortizable headroom — just
concentrated on large matrices and mostly reachable deterministically.

The §5.7 prototype then answers the "how" cleanly: **no cheap symbolic proxy
(fill/flop/max_front) reaches the time-oracle** (fill is the best of them, still ~13 %
short; flop and max_front worse), so the deterministic capture is an **actual-time
ordering race latched per pattern** — factor the candidates once, keep the fastest,
reuse across the up-to-3000 IPM iterations, amortizing the selection cost to ~0. That
reaches the oracle with no model. A learned tuner is the right tool only for callers
who cannot amortize even one extra factor (one-shot / non-reuse solves) — a real but
narrower audience than feral's IPM core.

**Recommendations, in priority order:**
1. ~~**Deterministic, ML-free, cheap (fill):** add `Amf` to the `AutoRace` race
   set.~~ **Withdrawn (see §5.7 corollary, reviewed 2026-07-02).** Filed as #111 on the
   fill evidence, but the time data reverses it: `AutoRace` selects by *fill*, and adding
   the low-fill-but-slow AMF regresses factor *time* on 8/51 large matrices (c-big 1.60×,
   dtoc2 1.43×) — the biggest problems where AutoRace matters — while its fill "wins" are
   on small matrices where fill barely affects time. Recommend **closing #111**; the
   limitation is the fill *selection criterion*, not the candidate set.
2. **Time/memory ordering selection is real but marginal — do NOT build it speculatively.**
   §5.6 found ~1.4–1.5×-over-`Auto` headroom, but §5.7 showed no cheap proxy captures it
   and §5.8 showed the actual-time race captures only ~9 % over `Auto` at steady state,
   amortizes only over a median ~206 same-pattern reuses (typical IPM is 20–50), mis-picks
   on the transient first iterate (38 % representativeness), and merely ties `AutoRace` on
   the stable tail. `Auto`/`AutoRace` are already within ~9 %. Pursue a time-aware selector
   **only** if a concrete high-repeat, large-matrix workload demands it — and if so, note
   §5.8's reversal: since the race *overhead* is the limiter, a **learned predictor** (zero
   inference cost, picks on the stable pattern) is the more viable vehicle than a
   deterministic race, not less.
3. **If a learned model is pursued,** keep it to inertia-neutral knobs (ordering,
   amalgamation, kernel gates — validated against the inertia gate); ship behind a
   deterministic fallback + the inertia gate, rslab-style. But weigh it against a ~9 %
   prize on a subset of large problems — likely below the maintenance bar unless the
   workload is specifically ordering-sensitive and high-repeat.
4. **Do not** put scaling or pivot-`u` under a model — inertia-sensitive, already
   handled by deterministic escalation.

The investigation deliverables are this note plus two reusable study binaries:
`crates/feral-diagnostics/src/bin/study_ordering_gap.rs` (fill) and
`study_ordering_timemem.rs` (numeric factor time + peak memory). No change to the
core solver is proposed.
