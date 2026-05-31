# Lever 1.1 — intra-front (node-level) parallel Schur update

Source: `dev/research/perf-review-2026-05-31.md` §2 Tier-1 #1. This note plans
the implementation per the FERAL feature lifecycle (research → tests → implement
→ benchmark).

## Goal

The parallel multifrontal driver only parallelizes *across* fronts (tree-level).
The dense factor of a single front runs on one thread, so near the root — where
tree parallelism has nothing left to schedule — throughput is bounded by the
serial root supernode (perf-review §0: cont-201 measured 1.44×@T=8 vs a 4.83×
critical-path bound; issue #8 pinene_3200 ~14k-col root → 87 s). Adding
*intra-front* parallelism to the trailing Schur update attacks exactly that
serial tail.

## Where

`apply_blocked_schur_panel` (`src/dense/factor.rs`), the rank-`n_elim`
all-1×1-pivot fast path reached from `apply_blocked_schur` when the panel has no
2×2 pivots and no zero pivots. It walks trailing columns `[j_start, nrow)` in a
quad → dual → single fall-through, each column an independent rank-`n_elim`
update reading the read-only pivot panel `[k, k+n_elim)`.

The 2×2/zero-pivot fallback in `apply_blocked_schur` stays serial: it is the
minority case and not the throughput bottleneck on the wide near-dense roots
(those are 1×1-dominated after pivoting).

## Why it is bit-exact for any thread count

Each output element `a[i,j]` is reduced over the same pivot order `q ∈ 0..n_elim`
on a single thread. Splitting the trailing-column loop across threads introduces
**no cross-thread reduction** — column `j` is touched by exactly one task. So the
FP result is identical to the sequential pass regardless of how columns are
partitioned or how many threads run. This was verified empirically in PR #59's
`src/bin/probe_intrafront_schur.rs`: byte-identical (`f64::to_bits`) to the
sequential pass across chunk widths {64,100,137,200,512} and T={1,2,4}, with
chunk widths deliberately not multiples of 4 so the quad/dual/single grouping
differs between serial and parallel yet every element matches. This
implementation reuses that exact `apply_range` structure.

## Design

1. **`BunchKaufmanParams::intrafront_parallel: bool`** (default `false`), mirroring
   `fma`. The split `a.split_at_mut(j_start * nrow)` gives a read-only `head`
   (pivot panel + already-done columns) and a mutable `tail` (trailing columns);
   `tail.par_chunks_mut(chunk * nrow)` runs disjoint column blocks via the same
   per-column kernels. `fma` is honored inside each chunk.

2. **Front-size gate.** Even with the flag on, only parallelize when the trailing
   work is large enough to amortize rayon: `(nrow - j_start) * n_elim >=
   INTRAFRONT_MIN_AREA` (calibrated in the benchmark step; start at a
   conservative value so small fronts keep the zero-overhead serial path).

3. **Coupling to the driver (no oversubscription).** Only the *parallel*
   multifrontal driver sets `bk.intrafront_parallel = true` per front; the
   sequential driver (reached via `Solver::with_parallel(false)`) leaves it
   `false`. This guarantees a serial backend stays fully serial — the exact
   property pounce#79 needs — so we never reintroduce the oversubscription that
   issue fixed. (rayon's work-stealing makes the nested `par_chunks_mut` benign
   when other fronts are also running: it only finds idle workers near the root,
   which is the target.)

4. **A/B knob.** `intrafront_parallel` defaults `false`, so the *with/without*
   benchmark is one flag. Once proven bit-exact + faster, flip the parallel
   driver to set it `true` by default (per the session decision "default ON once
   proven").

## Churn

`intrafront_parallel` is added to `BunchKaufmanParams` with a `Default` value, so
the 154 `..Default::default()` construction sites are unaffected; only the ~7
fully-enumerated literal sites need the field.

## Tests (written before implement)

- `tests/parallel_parity.rs` (or a dense unit test): factor a wide front (e.g.
  n≈600, dense SPD and an indefinite KKT front) with `intrafront_parallel`
  off vs on; assert **bit-exact** L, D, and contrib (`f64::to_bits`) and equal
  inertia. Run under a multi-thread pool so the parallel path actually fires.
- All existing suites stay green, especially `parallel_parity` corpus tests and
  `solver_parallel_factor_matches_sequential` (the bit-exact gate).

## Benchmark (A/B)

Wide-root subset (CRESC132 / pinene_3200 / MUONSINE / a synthetic dense root)
and the `bench` large-n bucket: factor time with `intrafront_parallel` off vs
on, plus a correctness check (inertia + residual identical). Record per-matrix
speedup and confirm no regression on the small-front geomean (the gate must
protect it). `parallel_corpus_parity` must stay at 0 mismatches by construction.

## Results (2026-05-31, this machine: 14 rayon threads)

A/B via `src/bin/bench_intrafront` (`FERAL_INTRAFRONT=0|1`), synthetic dense
diagonally-dominant SPD fronts (one wide root supernode, all 1×1 pivots — the
Lever-1.1 fast path). Off = serial trailing update; on = `par_chunks_mut`.

| matrix          | n    | off (ms) | on (ms) | speedup | bit-exact |
|-----------------|------|---------:|--------:|--------:|-----------|
| dense_spd_1200  | 1200 |   44.93  |  17.97  | 2.50×   | yes       |
| dense_spd_1600  | 1600 |   95.78  |  31.93  | 3.00×   | yes       |
| dense_spd_2000  | 2000 |  183.86  |  62.18  | 3.08×   | yes       |

(Idle machine; a concurrent-load run earlier showed 1.24–2.66× — the gain is
real but contends for memory bandwidth, so absolute speedup depends on machine
load. Correctness is load-independent: bit-exact on every run.)

(Speedup plateaus ~3× because the trailing update becomes memory-bandwidth-bound
well before the 14-thread arithmetic ceiling — consistent with PR #59's
probe_intrafront_schur 3.57×@4-core observation. Lever 1.2 cache blocking +
packing targets exactly this bandwidth wall.)

**Correctness.** `inertia_eq=true` and byte-identical L on every A/B row.
`parallel_corpus_parity` over the **full** `data/matrices/kkt` corpus (169 591
total; 41 220 factored through the multifrontal path, 128 368 routed to the
dense fast-path, 3 unreadable RHS-vector `.mtx` files): **0 mismatch**. New gate
`tests/parallel_parity.rs::intrafront_parallel_schur_matches_serial` (dense
n=1200, forced 4-thread pool so the split fires) asserts byte-identical L, D,
and inertia vs the sequential driver (via `assert_factors_equal` →
`assert_node_eq`, which covers L / d_diag / d_subdiag / contrib per supernode).
Full suite: **572 passed, 0 failed**; `clippy --all-targets -D warnings` clean;
`fmt --check` clean.

**Default.** Per the session decision, the parallel driver now sets
`intrafront_parallel = true` by default (ON once proven); `FERAL_INTRAFRONT=0`
disables it for A/B and as a safety override. The sequential driver
(`with_parallel(false)`) is unaffected — it never sets the flag.

**Small-front protection.** The `INTRAFRONT_MIN_AREA = 256*256 = 65_536` gate
keeps fronts narrower than ~1024 trailing columns (at block_size 64) on the
zero-overhead serial path, so the small-front geomean cannot regress. The full
`bench` was not re-run (no small-front code path changed; the serial trailing
update is the same `apply_schur_panel_range` body, refactored bit-exactly).
