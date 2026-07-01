# issue #91 — parallel dense-front factorization: diagnosis & design — 2026-06-30

The residual qap15 gap to faer (after the ordering fix, PR #92) is dense-kernel
throughput on ~36 large fronts, dominated by a 2955×2955 indefinite root
(42% of the sequential factor loop). This note records the measured cause and
the correctness-constrained design of the fix.

## Measured scaling (the crux)

qap15 factor, this machine (14 cores / 10 performance), `bench_qap15` steady ms:

| threads | time | speedup | note |
|---|---|---|---|
| 1  | 1923 | 1.00× | |
| 2  | 1422 | 1.35× | |
| 4  | 1037 | 1.85× | |
| 8  |  935 | 2.06× | flatlines 4→8 |
| 8, `FERAL_INTRAFRONT=off` | 1845 | ≈1.0× | tree-parallelism alone ≈ useless |

**Two facts.** (1) Tree-level (assembly-tree) parallelism does nothing here —
the work is concentrated in a few big fronts, not a bushy tree. (2) Within-front
(`intrafront`) parallelism carries everything but **tops out at ~2× on 10
cores**. That ceiling × the nofma per-core kernel ≈ the ~3× gap to faer.

**Amdahl fit.** Speedup 2.06× at 8 threads ⇒ effective serial fraction
`f = (1/S − 1/P)/(1 − 1/P) ≈ (0.485 − 0.125)/0.875 ≈ 0.41`. **~40% of the
parallel-mode work is effectively serial** — the exposed panel factorizations +
ramp-down tail. No amount of better per-tile scheduling breaks a 40% serial
floor; the fix must *overlap or remove the serial panel factorization* (Levers A
and D). This is the single number that rules out incremental scheduling tweaks.

Cheap levers already ruled out by measurement (`dev/tried-and-rejected.md`
2026-06-30): source-panel packing (net slowdown), larger `block_size` (no
change, capped at `MAX_N_ELIM=128`). FMA is +23–32% but a reproducibility-policy
change.

## Why intrafront scales only 2× — and the correctness constraint

Current model (`apply_blocked_schur_panel`, `factor_frontal_blocked_in_place_with_scratch`):
**fork-join per panel**. For each rank-`bs` (=64) panel of the front:

1. **serial** `lblt_panel_frontal`: Bunch–Kaufman factorization of the panel —
   dynamic 1×1/2×2 pivot search, growth tests, possible delayed pivots;
2. **parallel** `apply_blocked_schur_panel`: rank-`n_elim` trailing update,
   `par_chunks_mut` over trailing columns (bit-exact for any column partition —
   each column reduces over ascending `q` on one thread);
3. **join** (barrier) before the next panel.

Three structural caps:

- **Serial panel factorization (Amdahl).** For an *indefinite* KKT root this is
  not cheap — full BK pivoting is inherently serial and heavier than an SPD
  Cholesky panel. ~46 panels for the root, each a serial section with 9 idle
  cores.
- **No look-ahead:** panel *k+1* cannot begin until panel *k*'s entire trailing
  update joins. The serial panel-factor is fully exposed on the critical path.
- **Ramp-down + 1-D load imbalance:** late panels' trailing blocks fall below
  `INTRAFRONT_MIN_AREA` (serial tail); `par_chunks_mut` uses equal-width column
  chunks over a *triangular* trailing block (first chunks heavier).

### The hard constraint: dynamic BK pivoting is not statically tileable

A full 2-D tile-DAG factorization (PLASMA-style) assumes a **fixed elimination
order**. Bunch–Kaufman reorders rows/columns *during* factorization based on
values, and delayed pivots move work between panels — so a byte-exact tile-DAG
of the BK factorization is **not** cleanly achievable. This is the crucial
constraint that shapes the design: we cannot simply "tile it like faer's
Cholesky." Two consequences:

- The **trailing update** *is* statically structured once a panel's pivots are
  fixed — it is already parallel and byte-exact. Improving *its* parallelism is
  safe.
- The **panel factorization** is the dynamic, serial part. It cannot be tiled,
  but it *can* be overlapped (look-ahead) and it *can* be made cheaper
  (static/SQD pivoting — see "Orthogonal lever").

## Design — byte-exact, correctness-first, staged

### Lever A — look-ahead pipelining (primary; byte-exact, no numerics change)

Overlap the serial panel factorization with the parallel trailing update, the
standard fix for exactly this Amdahl cap (LAPACK `look-ahead`, faer):

- After factoring panel *k*, split its trailing update into (i) the **next
  panel's block-column** (rows for columns `k+bs .. k+2bs`) and (ii) the **rest**
  of the trailing block.
- Apply (i) first, then immediately start factoring panel *k+1* **while (ii)
  runs in parallel** on the remaining cores.
- Byte-exact: each trailing column still accumulates panel contributions in
  ascending panel/`q` order; look-ahead only reorders *independent* work
  (panel *k+1*'s factor depends only on region (i), which is completed first).
  Inertia and cross-arch bit-identity are preserved — the same invariant that
  already licenses `par_chunks_mut`.
- Interaction with delayed pivots: a delayed pivot from panel *k* defers a
  column into panel *k+1*; look-ahead must apply region (i) *including* any
  deferred columns before starting *k+1*. Handle by making the look-ahead
  block-column the exact input `lblt_panel_frontal` reads.

Expected: removes the serial-panel critical path → scaling from ~2× toward core
count on the big fronts.

### Lever B — 2-D trailing-update tiling + work-stealing (load balance)

Replace equal-width column chunks with 2-D tiles (e.g. 256×256) fed to rayon so
work-stealing balances the triangular front and the ramp-down. Byte-exact
(per-element order unchanged). Cheaper than A; do first as an independent A/B.

### Lever C — register-tiled bit-exact microkernel + opt-in FMA-on-large

Per-core throughput: a wider register tile (more independent accumulators) for
the per-tile nofma kernel, plus an opt-in FMA policy gated to large fronts (the
measured +23–32%, keeping the 4 small-front pivot-drift KKTs on nofma). FMA is a
reproducibility-policy decision, tracked separately.

### Orthogonal lever D — static/SQD pivoting for large well-regularized fronts

The deepest lever: factor large quasi-definite fronts with **static signed
pivoting** (Vanderbei/SQD) + static-pivot perturbation for tiny (1e-10)
regularization pivots, instead of dynamic BK. This (i) removes the serial BK
search entirely (uncapping Lever A), and (ii) makes the front statically
tileable (enabling a true tile-DAG). Cost: it perturbs, so inertia is of `A+Δ`
and needs iterative refinement — a numerics/policy change the maintainer must
accept. The existing `with_sqd_mode` trips `SqdContractViolated` on qap15's
1e-10 pivots; pairing it with `static_pivot_threshold` (issue #38) is the route.
Highest ceiling, highest risk — sequence after A/B/C prove out.

## Implementation plan (each stage: byte-exact parity + corpus inertia gate)

Parity oracle throughout = the current serial/`par_chunks_mut` path (itself
proven bit-exact vs the scalar reference). Benchmark on qap15 + corpus.

1. **Measure the serial fraction** — instrument the blocked factor to split
   per-front time into panel-factor vs trailing-update. Confirms A is the lever
   (expected: panel-factor is the exposed serial critical path on the root).
2. **Lever B** (2-D tiling / work-stealing) — smaller, independent, byte-exact.
   A/B on qap15.
3. **Lever A** (look-ahead) — the primary scaling fix. Byte-exact; handle
   delayed pivots in the look-ahead block-column.
4. **Lever C** (microkernel width; opt-in FMA-on-large as a separate PR).
5. **Lever D** (static/SQD large-front path) — only if A–C leave a gap and the
   perturbation/refinement policy is accepted.

## Harnesses

`examples/bench_qap15.rs` (configs incl. block-size A/B), `examples/profile_qap15.rs`
(per-front buckets, nofma vs fma). Fixture `tests/data/large/qap15_kkt.mtx`
(gitignored; `dev/scripts/regen_qap15_kkt.sh`).
