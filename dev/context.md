# FERAL Context (auto-generated)

Generated: 2026-06-06T16:33:38Z

## Latest Session
File: dev/sessions/2026-06-06-02.md
```
# Session 2026-06-06-02

## Goal

Invest in a systematic super-linear (O(n²)) scaling audit across ordering,
scaling, and the numeric/symbolic prologue — including verifying the recent
MC64 heap-reuse fix (issue #80, commit 6699f09). Plan approved: full
systematic sweep + a deterministic (CI-noise-immune) MC64 regression guard.

## Accomplished

**Step 0 — MC64 Hungarian regression guard (committed 767b0d9).**
- Added `HungarianStats {heap_init_slots, augment_searches, touched_total,
  phase3_inner_iters}` to `src/scaling/hungarian.rs`, threaded `&mut stats`
  through `IndexHeap::new`/`reset` (counting `pos[]` slots zeroed) and the
  phase-3 length-2 augmentation loop. `hungarian_match_instrumented()` returns
  the counters; `hungarian_match()` is a thin wrapper (off the Dijkstra inner
  loop → negligible overhead, bit-identical matchings).
- Test `mc64_hungarian_no_quadratic_heap_realloc_regression`. Calibration
  (gen_random_sparse deg=3, n=1k..8k) showed the legitimate `touched_total` is
  itself ~quadratic on a hard matching (long augmenting paths), so a growth
  ratio on total heap work is the wrong guard. The correct, threshold-free
  guard is the exact structural invariant `heap_init_slots == n + touched_total`
  (heap allocated once + incremental resets), which fires on the realloc revert
  at any n. `phase3_inner_iters` IS linear (563→4609, 8.2× over 8× n) → a <16×
  ratio guards the O(nnz²) phase-3 suspect.
- Teeth verified: injecting a per-search `IndexHeap::new` → `heap_init` 254727
  vs expected 53727 → test FAILS as designed; reverted. Full feral lib suite
  317 passed/0 failed; clippy/fmt clean.

**Step 1 — `scaling_sweep` diagnostic binary (committed 1b2b21c).**
- `crates/feral-diagnostics/src/bin/scaling_sweep.rs`. Modes `--family` /
  `--manifest` / `--generated {spd,kkt} --sizes`. Per matrix: profiling-on
  Solver, `invalidate_symbolic_cache()` before each of K factors (forces a
  symbolic miss so the normally-cached symbolic phase is timed), per-field
  median over K, CSV with the full prologue breakdown + all 17 symbolic stages
  + `max_col_degree`/`sum_d_logd` control variates. `--scaling` pins the
  strategy. Rust = data collection only; α-fits run in Python over the CSV.
- Generators are constant-bandwidth (banded) so fill stays near-linear; on the
  banded SPD/KKT ladders all phases scale α≈1.0 (good fittable baseline).

**Step 2 — rocket_12800 localization + #80 (committed in 1b2b21c; CORRECTED
in journal 16:35).**
- rocket_12800_0000 (n=89601, nnz=332793, **max_col_degree=38401**) with MC64:
  `pb_scaling_us` = 4.3 ms (numeric), symbolic = 38.8 s of which
  `sym_ldlt_compress` = 38.3 s (98.9%). `permute_us` = 64 ms (exonerated, as the
  research note predicted).
- **Correction:** `ldlt_compress` = `compute_mc64_cache` → `compute_matching`
  → `hungarian_match`. So the 38.3 s IS the MC64 Hungarian, run in symbolic for
  the LdltCompress ordering-compression preprocessor. The cheap 4.3 ms numeric
```

## Git Status
```
1b2b21c feat(diagnostics): scaling_sweep binary; verify #80 + find ldlt_compress O(n^2)
767b0d9 test(scaling): deterministic MC64 Hungarian O(n^2) regression guard (#80)
bc8496a docs(session): addendum — MC64 heap fix + dead-code removal (#80)
10a3a1a refactor(ordering): remove dead amd_order, keep permute_pattern (#80)
6699f09 perf(scaling): reuse MC64 Hungarian heap across columns (#80)
```

## Test Status
```
