# Plan: logical-permutation Forrest–Tomlin row-elimination (issue #87)

Design & route selection: `dev/research/ft-row-elimination-design-2026-06-21.md`.
Goal: replace the `O(bump²)` full-bump re-triangularization in
`src/lu/sparse_update.rs::eliminate_bump` with an `O(bump)` Forrest–Tomlin
row-elimination that carries a logical permutation `uperm`, so wide dense-spike
updates (issue #87 / discopt#229) and their eta files scale `O(m)`, not `O(m²)`.

Correctness-first: every phase keeps the full suite + the casctanks trace green
and adds the differential oracle for that phase before the code.

## Invariant

`P A Q = L G U`, `G = E₁⁻¹…Eₜ⁻¹`. `U` is upper triangular **in `uperm` order**
(`uperm`/`uperm_inv`: pivot-position ↔ triangular rank). `uperm = identity` at
factor, so the pre-update world is byte-identical. Etas (`FtOp` axpy/swap), `L`,
`P`, `Q`, scaling, refinement stay in fixed pivot-position coordinates — never
relabeled.

## Phases (each: oracle/test first → code → `cargo test` + clippy → atomic commit)

### P0 — measurement harness (no behavior change) ✅ scaffolding
- `lu_wide_bump_probe` (done): dense-spike scaling table + exponent.
- Keep `casctanks_ft_update` bench as the realistic before/after.
- BEFORE captured: probe exp≈2.8, casctanks 144-chain 16.88 ms.

**Status (2026-06-21):** P1 ✅ (`a676aaf`), P2 ✅ (`ebaeca6`), P4 ✅ (probes +
casctanks bench + new differential/invariant tests). P3 (permute-when-possible)
and in-bump stability pivoting deferred — growth-monitor → `NeedsRefactor` covers
correctness. P5 checkpoint in progress.

### P1 — `uperm` plumbing, identity-only (pure refactor, zero behavior change) ✅
- Add `uperm`, `uperm_inv` fields (identity at factor; cloned in refactor).
- Rewrite `usolve`/`ut_solve` to walk rows in `uperm` order and take each row's
  `uperm`-diagonal as pivot, with a fast path when `uperm` is identity (the
  current loop verbatim). Diagonal-first invariant becomes "pivot-first in the
  row's `uperm`-rank sense."
- Test: all existing solve/parity tests green (identity `uperm` ⇒ no change);
  add `uperm_identity_soles_match` guard.

### P2 — `ft_update`: spike-insert + symmetric-permutation FT, no fill-perm opt
- New module path replacing `eliminate_bump`. Steps:
  1. `compute_spike` (reuse) → spike `ρ`, support, bump `[r,h]` in `uperm` rank.
  2. Insert `ρ` as the new last bump column (rank `h`); old ranks `r..h-1` are the
     old `r+1..h` (cyclic shift composed into `uperm`/`uperm_inv`, `O(bump)`).
  3. Single-row elimination of the old row `r` (now rank `h`) against the
     triangular part, recording axpys (`FtOp::Axpy`, fixed pivot coords). Optional
     2×2-style swap only for stability, bounded — not the `O(bump)` cyclic shift.
  4. Growth monitor over changed rows; `NeedsRefactor` + rollback unchanged.
- Oracle FIRST: `wide_bump_dense_update_matches_dense` — chain of dense-column
  updates on a sparse base; assert `‖Bx−a‖` and dense↔sparse parity after each,
  and `U` upper-triangular under `uperm`.
- Reuse the rollback/pool machinery (`saved_*`, `row_pool`, …).

### P3 — Schork–Gondzio "permute-when-possible" fast path
- Before eliminating, run the symbolic pass: if the spiked digraph over `[r,h]` is
  acyclic, re-triangularize by permutation **alone** (compose into `uperm`, no
  axpys, no fill). Fall back to P2's single-row elimination on the residual.
- Test: assert zero `eta_ops` added when a permutation-only update applies
  (construct such a case); residuals still exact.

### P4 — validation, benches, scaling
- AFTER table from `lu_wide_bump_probe` (target exp≈1.0, eta `O(m)`).
- `casctanks_ft_update` AFTER; `lu_update_trace` allocs probe unchanged-or-better.
- Full `lu_*` suites, clippy `-D warnings`, fmt.

### P5 — checkpoint
- Research note status → done; `decisions.md` entry; CHANGELOG; session file;
  journal entries throughout (real-time).

## Risks / watch items
- `uperm`-ordered `U`-solve correctness is the load-bearing change — guard with the
  identity fast path + the triangularity invariant test.
- Stability without in-bump magnitude pivoting: rely on `growth`/`max_growth` +
  refactor; verify on an ill-conditioned dense-spike chain.
- btran replays etas transposed-reverse *and* `Uᵀ` in `uperm` order — assert btran
  residuals in the same differential test, not just ftran.
