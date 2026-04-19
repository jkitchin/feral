# Lever C — Adaptive MC64 Scaling for Delay-Prone KKTs

**Date:** 2026-04-19
**Supersedes (in part):** the lever-C definition in
`dev/research/dense-kernel-vesuvio-tail.md` §3, which described
lever C as "exploit the arrow-KKT structure during symbolic to
shrink the root frontal." The diagnostic in §1 below shows that
characterization was wrong. The actual lever is the scaling
strategy.
**Diagnostic source:** `src/bin/vesuvio_diag.rs` extended this
session with column-degree distribution and InfNorm-vs-MC64
toggling; output captured in `dev/journal/2026-04-18-09.org`
entry 3.
**Existing context:** `src/scaling/mod.rs` `ScalingStrategy`
docstring (the prior InfNorm-vs-MC64 decision); `dev/decisions.md`
2026-04-12 entry "Phase 2.2.1 MC64 default" and 2026-04-13 entry
"Phase 2.3 pivot-threshold split"; `dev/sessions/2026-04-12-02.md`
(MC64-as-default → InfNorm-as-default flip on the Phase 2.2.3
follow-up).

## 1. The original lever C was wrong

Yesterday's research note (`dense-kernel-vesuvio-tail.md`)
attributed the VESUVIO factor outliers to "single dense linking
column → unavoidable ~67%-of-n root frontal" and proposed
exploiting that arrow-KKT structure during symbolic. Today's
diagnostic refutes that:

```
== VESUVIOU_0000 ==
  shape: n=3083  max_col_nnz=1026  diag_only=1025
  col_deg: deg=1: 1025  deg=2-4: 2050  deg=5-32: 0  deg>32: 8
  Amd/infnorm: sym_max_nrow=10  actual_max_nrow=2059
               total_delays=949  root=2059x959 (67% of n)
               fac=234.7 ms
```

Two observations the original note missed:

1. **`sym_max_nrow = 10`.** Symbolic predicts the largest frontal
   has 10 rows. The actual max_nrow is 2059 — a 200× blow-up.
   The root frontal is not large by structural design; it is
   large because pivots got pushed up the etree.
2. **`total_delays = 949` out of 4032 attempted.** 23% of pivot
   attempts are rejected and delayed. The 949 delayed columns
   cascade up to the root and accumulate there — root_ncol = 10
   native + 949 delayed = 959. That, not the structure, is what
   makes the root dense.

Same pattern across the family:

| matrix       | sym_max | actual_max | delays | root | factor (ms) |
|--------------|--------:|-----------:|-------:|-----:|------------:|
| VESUVIOU_0000|      10 |       2059 |    949 |  959 |       234.7 |
| VESUVIOU_0005|      10 |       2059 |    580 |  590 |       150.1 |
| VESUVIO_0000 |      13 |       2062 |    142 |  155 |        54.0 |
| VESUVIO_0021 |      13 |       2062 |    697 |  710 |       179.1 |
| VESUVIA_0000 |      10 |       2059 |    264 |  274 |        84.0 |
| MUONSINE_0000|       3 |       1024 |    312 |  315 |        20.0 |
| CRESC132_0000|       8 |       5084 |   4846 | 4854 |      5316.3 |

Factor time tracks delay count, not structure. The
"dense-kernel-limited" attribution from session 08 was correct
about *what* was slow (the dense scalar BK kernel on a giant
root), but wrong about *why* the root was giant.

## 2. MC64 scaling eliminates the delays entirely

The scaling vector controls how well-conditioned the diagonal is
at the start of factorization. Knight-Ruiz ∞-norm balancing
(InfNorm, current default) doesn't help on matrices where the
slack rows have small diagonals and the linking columns are very
unbalanced. MC64 matching-based scaling does.

Same matrices, same ordering, same kernel — just changing
scaling from InfNorm to MC64Symmetric:

| matrix       | InfNorm fac | MC64 fac  | speedup | InfNorm delays | MC64 delays |
|--------------|------------:|----------:|--------:|---------------:|------------:|
| VESUVIOU_0000|    234.7 ms |   10.4 ms |     23× |            949 |           0 |
| VESUVIOU_0005|    150.1 ms |    9.4 ms |     16× |            580 |           0 |
| VESUVIO_0000 |     54.0 ms |    9.6 ms |    5.6× |            142 |           0 |
| VESUVIO_0021 |    179.1 ms |   10.5 ms |     17× |            697 |           0 |
| VESUVIA_0000 |     84.0 ms |    8.6 ms |    9.8× |            264 |           0 |
| MUONSINE_0000|     20.0 ms |    1.6 ms |   12.5× |            312 |           0 |
| CRESC132_0000|   5316.3 ms |   23.2 ms |    229× |           4846 |           0 |

Inertia matches between scalings on every matrix (no correctness
regression). Delays drop to **zero** in every case. Factor times
collapse by 6×–229×.

For VESUVIOU the new ratio against MUMPS is 10.4ms / ~2.5ms ≈
4× — down from 84×. For CRESC132 the new ratio is ~23ms / ~10ms
≈ 2× — down from 11× (note CRESC132 is already on MetisND from
the dispatcher; MetisND/MC64 is 16ms, even better).

This is *the* lever for the IPM tail. It is also single-digit
lines of production code (the default), modulo the corpus
validation question in §3.

## 3. Why InfNorm is the current default — what the prior decision found

The 2026-04-13 Phase 2.2.3 follow-up flipped the default from
MC64Symmetric back to InfNorm. The recorded reason
(`src/scaling/mod.rs:44-60`):

> MC64 was a silent no-op on matrices like HYDCAR20, METHANL8,
> SWOPF, and HATFLDG — matrices whose raw row norms span 4+
> orders of magnitude but whose MC64 matching-based scaling came
> out near-identity. Knight-Ruiz equilibration scales those
> matrices successfully and the sparse path then matches the
> MUMPS oracle.

The prior decision was symmetric: InfNorm helps a different set
of matrices, MC64 helps VESUVIA-family. The choice of InfNorm as
default reflected which class was bigger in the corpus at the
time, not that MC64 was strictly worse.

So the policy question for lever C is *not* "switch the default
to MC64"; it is "make scaling adaptive based on a cheap matrix
feature, instead of one-size-fits-all".

## 4. Three candidate policies

### Policy 1 — switch default to MC64 unconditionally

Smallest patch (one line in
`src/symbolic/supernode.rs::SupernodeParams::default`). Wins
big on the seven measured matrices and on the IPM tail
generally. Loses on HYDCAR20 / METHANL8 / SWOPF / HATFLDG and
unmeasured similar shapes — quantifying that loss requires the
full bench.

Risk: regressing the ~150 580 sparse-residual-pass count
(currently 154 241 of 154 588) by some unknown amount on the
matrices where InfNorm currently helps. May be small (those
matrices are 4 of ~155k); may be larger if there is a class of
similar shapes.

### Policy 2 — adaptive routing based on matrix shape

Rule: detect the "many-degree-1 rows + few high-degree linking
columns" signature in symbolic, route to MC64 for those; keep
InfNorm elsewhere. The signature is exactly what
`vesuvio_diag` already prints — `diag_only ≥ 0.3 · n` is a
candidate threshold. Implementation is one column-degree pass
(O(nnz)) followed by a `ScalingStrategy` selection.

Risk: the threshold is a tunable. Picking it on the seven
measured matrices alone is overfitting. Need a corpus sweep to
calibrate.

Upside: gets the VESUVIO / CRESC wins without the unmeasured
HYDCAR-class regression. Mirrors the existing `pick_default_method`
adaptive-ordering pattern (it already routes by `(n, nnz/n)`).

### Policy 3 — try-MC64-first-with-InfNorm-fallback

Run MC64 symbolic; if MC64 returns `PartialSingular` or if a
heuristic (e.g. `||DAD - I||_∞ < 0.1` on the scaled diagonal)
indicates MC64 was a no-op, fall back to InfNorm. The
`ScalingInfo` enum already supports `PartialSingular` reporting.

Risk: doubles the symbolic cost on the matrices where MC64
fails. MC64 is already 2–4× slower than InfNorm at symbolic
time (see CRESC132: 4.3 ms → 12.0 ms).

Upside: zero false negatives — every matrix gets the
better-conditioning scaling. No threshold to tune.

## 5. Recommendation

Run the full 154 588-matrix IPM bench with all three policies
(plus the current InfNorm baseline) before authoring a plan.
Specifically:

1. **Baseline** — current `default = InfNorm`. This is the
   session-08 number (sparse factor/MUMPS geomean 0.42).
2. **Policy 1** — flip default to `Mc64Symmetric`. Measures the
   "blanket switch" delta directly.
3. **Policy 2** — implement the `diag_only / n ≥ 0.3` heuristic
   in `pick_scaling_strategy(csc) -> ScalingStrategy`,
   mirroring `pick_default_method`. Measure.
4. **Policy 3** — implement the try-MC64-fallback-to-InfNorm
   path. Measure.

The decision criterion is the same as the prior 2.2.3 follow-up:
maximum sparse residual-pass count and inertia-match count, ties
broken by factor-geomean. CLAUDE.md's hard rule on tolerances
applies — no loosening to favor a policy.

Estimated cost: one session for policies 1+2 (the implementation
is small; the bench takes ~30–60 min wall time), one more
session for policy 3 if needed. The plan and validation report
follow once the data is in.

## 6. Why this is "lever C", not "lever D"

The parent note's lever C was named for "exploit the structure
during symbolic." The lever still lives in symbolic — it is the
choice of `ScalingStrategy` inside `SupernodeParams`, which is
consumed by symbolic factorization. The fact that the structure
itself is not the lever (delays are) does not change where the
patch goes. Renaming would be cosmetic; the implementation
boundary is the same.

## 7. What this note explicitly does not cover

- **The dense-kernel SIMD GEMM work (lever B / 2.4.1b).** Still
  the right path for non-VESUVIO factor outliers if any emerge
  after lever C lands. The 2.4.1b plan stays on the shelf.
- **The arrow-KKT structural fix.** Genuinely not the bottleneck
  on the measured corpus. If a future workload exhibits a
  structurally dense root *with* MC64 scaling and *with* zero
  delays, this option re-enters the menu.
- **K1 as generic preprocessor and HEMSR coarsening.** Both still
  deferred per session 08; orthogonal to scaling.

## 8. Files touched in this session

- `src/bin/vesuvio_diag.rs` — added column-degree distribution
  and the `(method, scaling)` cross-product. Diagnostic only;
  not a production change.

The scaling toggle and the corpus measurement that the §5
recommendation calls for are explicitly *not* in this session.
This note is the prerequisite; the next session executes the
measurement.
