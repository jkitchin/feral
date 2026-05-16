# FERAL Context (auto-generated)

Generated: 2026-05-16T18:57:27Z

## Latest Session
File: dev/sessions/2026-05-16-08.md
```
# Session 2026-05-16-08

## Goal

Ship the M7 SQD (symmetric quasi-definite) fast-path through
phases (c)–(h) of the user-approved plan
(`~/.claude/plans/let-s-work-on-a-reflective-anchor.md`). Phase
(a) (research note + bib + decisions + GH issue #34) and phase
(b) (`NumericParams::sqd_mode` builder + `Solver::with_sqd_mode`)
landed in earlier sessions. This session covers (c) kernel, (d)
dispatch, (e) error variant + L-growth guard, (f) full test
coverage, (g) bench harness, (h) checkpoint.

## Accomplished

### Phase (c) — diagonal kernel landed (commit `58e7421`)
- `factor_diagonal(matrix, params) -> Result<(Factors, Inertia)>`
  in `src/dense/factor.rs`: top-level analog of `factor()`, applies
  equilibration, calls `factor_frontal_diagonal_in_place` at full
  dimension.
- `factor_frontal_diagonal_in_place(matrix, ncol, params)
  -> Result<FrontalFactors>`: per-pivot loop reading `a[k,k]`,
  reusing the shared `do_1x1_update` rank-1 kernel, counting
  inertia from `sign(d)`. Same return shape as the BK frontal
  factor so downstream consumers (solver, assembler) see no
  structural difference.
- 3 unit tests in `tests/sqd_fast_path.rs` (hand-check 2x2 pure
  diagonal, 2x2 with off-diagonal, zero-pivot rejected).

### Phase (d) — supernode dispatch wired (commit `05730a4`)
- `params.sqd_mode` dispatch at three FrontalFactors construction
  sites in `src/numeric/factorize.rs`:
  - `dense_fast_factor_with_workspace` (n ≤ 16 or ρ ≥ 1/4)
  - `factor_one_supernode` (multifrontal driver)
  - `factor_one_small_leaf` (small-leaf specialisation)
- Added 3 Solver-level dispatch tests (dense path, multifrontal
  path on n=24 banded SQD, contract-violation surfaces as
  non-success).

### Phase (e) — `SqdContractViolated` + L-growth guard (commit `b44b9d9`)
- New `FeralError::SqdContractViolated { column: usize, pivot: f64 }`
  variant in `src/error.rs`.
- Two contract bounds enforced per pivot:
  1. `|d_kk| > zero_tol` (near-zero guard, unchanged).
  2. `max |l_{ik}| <= 1/sqrt(EPS) ≈ 6.7e7` (Gill-Saunders-Shinnerl
     1996 column-growth bound, new).
- Both trips surface `SqdContractViolated` immediately —
  never silent BK fallback.
- 2 additional tests: `sqd_l_growth_bound_rejected` (column-growth
  trip even when `|d| > zero_tol`),
```

## Git Status
```
499e5de bench(#34): bench_sqd synthetic SQD-vs-BK harness (phase g)
4adef8c test(#34): phase (f) reference-parity + property + cache coverage
b44b9d9 feat(#34): SqdContractViolated + L-growth guard (phase e)
05730a4 feat(#34): wire factor_one_supernode SQD dispatch (phase d)
58e7421 feat(#34): factor_diagonal + factor_frontal_diagonal_in_place (gated)
```

## Test Status
```
