# Per-factor cost cluster — two mechanisms

**Status:** Pre-implementation research note.
**Date:** 2026-05-21
**Author:** session 2026-05-21-01
**Related:**
- Issues #44 (NARX_CFy timeout), #47 (explicit-zero fast-path defeat),
  #38 (closed — rocket_12800 residual perf gap)
- `external_benchmarks/mittelmann_ipopt/REPORT-vs-plato.md` (the sweep
  that surfaced the loss cluster)
- `dev/research/phase-2.11-small-front-amalgamation.md` (loop-side
  tiny-front overhead — related but distinct, see §5)
- `dev/research/cascade-break.md`,
  `dev/research/issue-15-cascade-break-symbolic-arm.md` (Mechanism A
  lever)
- `dev/journal/2026-05-21-01.org` (the profiling run log)

---

## 1. The question

The 2026-05-17 full Mittelmann sweep left a cluster of problems where
feral's whole-solve wall is 5–34× MA57, and two (`NARX_CFy`,
`steering_12800`) time out where MA57 finishes. Whole-solve ratios
(`REPORT-vs-plato.md`):

| problem      | feral/MA57 |
|--------------|-----------:|
| marine_1600  |      33.97 |
| pinene_3200  |      12.32 |
| robot_1600   |       7.02 |
| corkscrw     |       5.56 |
| arki0003     |       5.54 |
| rocket_12800 |       4.67 |
| NARX_CFy     |   TO (>600 s) |

Is this one mechanism or several? This note answers that with a
profiling pass over five of the cluster problems (`corkscrw`'s KKT
corpus is not present locally, so it is excluded).

## 2. Method

Each problem has a dumped IPM KKT corpus under
`data/matrices/kkt-mittelmann/<problem>/` (gitignored). All runs use
`probe_kkt_replay` (warm replay through one `Solver`) and
`probe_rocket_profile` / `probe_robot_profile` (sequential replay with
the per-supernode `Profiler` attached). Configurations swept: default
(`AUTO_CB=0.05`), `CB=on`, `FRESH=1` (rebuild `Solver` per call),
`SCALING ∈ {infnorm, identity, mc64}`.

## 3. Headline finding — two mechanisms

The `CB=on` experiment separates the cluster cleanly. Total replay
factor time, default vs `CB=on`:

| problem      | default | CB=on   | CB helps? |
|--------------|--------:|--------:|-----------|
| arki0003     |  1.441s |  1.502s | no        |
| robot_1600   |  0.633s |  0.199s | **yes ~3×** |
| NARX_CFy     |  1.641s |  1.733s | no        |
| rocket_12800 |  6.455s |  6.632s | no        |
| marine_1600  | 24.485s | 10.520s | **yes ~2.3×** |

- **Mechanism A — delayed-pivot cascade** (`robot_1600`, `marine_1600`):
  a small number of single-iteration factor spikes that `CB=on`
  collapses ~10×.
- **Mechanism B — CB-immune sustained cost** (`rocket_12800`;
  `arki0003` and `NARX_CFy` are also CB-immune): `CB`, `FRESH`, and
  scaling strategy all leave it unmoved.

## 4. Mechanism A — delayed-pivot cascade

Per-call detail (default `AUTO_CB=0.05`):

- `robot_1600`: iter 1 = 0.473 s; under `CB=on` → 0.054 s. Iters 2–6
  are already fast (auto-CB armed them from iter 1's delayed count).
- `marine_1600`: iter 9 = **14.651 s**; under `CB=on` → 0.588 s.

`probe_robot_profile` (sequential, profiler) on a cascading iter:

```
robot iter2: prologue=6.6 ms  loop=424.8 ms  overhead=1.5%  n_snodes=3958
  snode 3957 (last/root, nrow=28 ncol=28)  278.31 ms
  snode 3956 (nrow=105 ncol=77)             77.04 ms
```

The profile signature is **loop-dominated**: cost piles into a few
large supernodes, above all the root, where delayed pivots from the
whole elimination tree accumulate. A nominally 28×28 root front taking
278 ms is the delayed-column list ballooning the effective front.

**Why auto-CB does not already fix it.** `Solver::with_auto_cascade_break`
is *reactive*: it arms cascade-break for factor N+1 when factor N's
`max(n_delayed) ≥ β·n`. The first cascade in a trajectory is therefore
never prevented — `robot` iter 1 and `marine` iter 9 are exactly those
unanticipated first spikes. `marine` iter 8 (0.84 s) was mildly
elevated but did not cross `β=0.05`, so iter 9 cascaded unbraked.

## 5. Mechanism B — prologue-dominated factor cost

`probe_rocket_profile` (sequential, profiler):

```
rocket iter0: prologue=3408.5 ms  loop=10.8 ms  epilogue=0.0 ms
              total=3424.7 ms  overhead=99.5%  n_snodes=16406
rocket iter1: prologue=2287.4 ms  loop= 7.9 ms  n_snodes= 6283
```

**99.5% of the factor time is the prologue.** The per-supernode
numeric loop — the actual Bunch-Kaufman arithmetic — is ~10 ms and is
*not* the problem. The supernode bucket histogram confirms it: 16406
supernodes for n=89601 (avg 5.5 columns each), every front nrow ≤ 32,
total loop work ~10 ms.

The "prologue" is everything in `factorize_numeric`
(`src/numeric/factorize.rs:1668–1782`) before the supernode loop:
`compute_scaling_with_cache`, `permute_csc_values`,
`symmetric_pattern()`, the `from_triplets` rebuild, and workspace
setup.

Controls that rule out the obvious causes:

- **Not scaling / MC64.** `SCALING=infnorm` gives rocket 6.905 s vs
  6.455 s default — no improvement. (Contrast the #38-era *live*
  ipopt run where MC64 was 55% of wall; on this dumped corpus MC64 is
  cheap.)
- **Not warm-state.** `FRESH=1` gives rocket 6.374 s ≈ warm 6.455 s.
  Unlike #38 Failure A there is no warm-state leak; the cost is
  intrinsic to factoring each matrix.
- **Not the numeric kernel.** Loop = 10 ms.

A static read of lines 1668–1782 shows only O(n)/O(nnz) code —
`symmetric_pattern`, `permute_csc_values`, `from_triplets` are all
linear passes. 3.4 s for nnz≈333k is two-to-three orders of magnitude
slower than the apparent work. **The prologue cost is anomalous and
must be localized empirically** (plan step B1) — it is not explained
by reading the code.

This is **distinct from `phase-2.11` small-front amalgamation.** Phase
2.11 targets loop-side per-front arithmetic overhead on tiny IPM-tail
fronts (the ACOPR30/CRESC100 8× cases). Rocket's loop is already fast;
its gap is the prologue. Amalgamation may still help indirectly (fewer
supernodes → less prologue per-supernode work, *if* the prologue cost
turns out to scale with `n_snodes`), but that is a hypothesis B1 must
test, not an established fix.

## 6. arki0003 — a value-dependent sub-case

`arki0003` deserves a separate note. Per-call (default):

```
iter 0      0.025 s   (cold)
iter 1-4    0.30-0.42 s
iter 6-65   0.001 s
```

Iters 1–4 are ~300× slower than iters 6–65 **on the same IPM
pattern** (same symbolic factorization). The cost is therefore in the
numeric loop and is purely value-dependent: early-IPM iterates (large
barrier µ) factor slowly, late iterates factor instantly. `CB=on` does
not move it, so it is not the L-perturbation cascade of Mechanism A;
it is more likely heavy 2×2 / delayed pivoting that the specific
cascade-break perturbation does not relieve. Full classification needs
a delayed-pivot count trace (plan step A1 instrumentation will provide
it). For now: arki is a value-dependent pivoting-cost case, adjacent
to Mechanism A but not fixed by the current CB lever.

## 7. marine_1600 — a correctness side-finding

Independent of either perf mechanism, `marine_1600` returns
`WrongInertia` at iters 10, 14, 16, 17 under **both** CB modes
(e.g. got `38409/38392/6`, want `38415/38392/0`). The factor reports
spurious zero eigenvalues. This is an inertia-correctness drift, not a
speed issue, and `CB=on` does not fix it. It should be filed and
triaged separately from the per-factor cost work.

## 8. Summary classification

| problem      | mechanism | profile signature        | lever                         |
|--------------|-----------|--------------------------|-------------------------------|
| robot_1600   | A         | loop, root supernode     | proactive cascade-break       |
| marine_1600  | A (+ §7)  | loop, single iter spike  | proactive cascade-break       |
| rocket_12800 | B         | prologue 99.5%           | localize + fix prologue       |
| NARX_CFy     | B (likely)| CB/scaling-immune        | localize + fix prologue       |
| arki0003     | A-adjacent| loop, value-dependent    | needs delayed-pivot trace     |

`NARX_CFy` is classified B by exclusion (CB-immune, scaling-immune); a
direct `probe`-style supernode profile should confirm the prologue
signature before committing the fix.

## 9. Conclusion

It is **two mechanisms, not one.** They have opposite profile
signatures and need separate fixes:

- **Mechanism B is the larger and clearer prize.** A 3.4 s prologue
  doing ~333k nonzeros of nominally linear work is a localizable
  inefficiency, not a fundamental algorithmic gap. Fixing it plausibly
  closes the rocket residual and #44 (NARX_CFy) at once.
- **Mechanism A is already half-solved.** `CB=on` works; the gap is
  that auto-CB is reactive. Making cascade-break proactive (symbolic
  arm, or arm-from-iter-0) catches the first spike.

See `dev/plans/per-factor-cost-cluster.md` for the implementation
plan.
