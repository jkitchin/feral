# Issue #221: 2×2 partner from the fully-summed argmax

## Symptom

pounce on brainpc0-9 (Vanderbei) and the largest Mittelmann QCQP/NQL
models stalls on feral 0.19/0.20. The first solve of a brainpc3 IPM
iteration (n = 13,805, μ ≈ 2.5e-9) factors in 0.04 s on 0.18.0 and in
9.4 s on 0.20.0. The reporter's `sample` puts most of the busy time in
`try_reject_1x1_with_rook_rescue`.

## Diagnosis (probes on the reporter's matrices)

Matrices were dumped with pounce `main` linked against local feral
(`--dump kkt:34-40`). The harness is
`crates/feral-diagnostics/src/bin/diag_221_rook_cost.rs`.

- iter 37/1 on HEAD: 9.08 s cold, 9.21 s warm, nnz(L) = 12.88M
  (iter 34: 162K). 428 fronts receive delays, Σ n_delayed_in = 399,871,
  and the largest fronts carry about 2000 delayed columns with nelim of
  17-80. 11,846 of 13,805 columns are delayed at least once, up to 235
  times.
- Profile (`sample`, 8 s): `rook_rescue` 3191, Schur kernels about 1370,
  `scalar_pivot_step` 600, `swap_rows_cols` 285. A cheaper rook alone
  would leave seconds, so the delay volume is the defect.
- Values: state-variable Hessian diagonals are about 1e-13 (about 1e-8 at
  iter 34). Constraint rows have a zero diagonal, weight ~1 to their state
  variables, and weight ~600-675 to two parameter hub columns (degree
  10,348, eliminated at the root).
- Origin: single-column leaf supernodes hold a state variable whose MC64
  partner (a constraint) is a ghost row. |d| ~ 1e-13 < u·1, so it delays.
  That delay is expected and costs one hop.
- Cascade: in the parent front the delayed state column sits at the end
  of the fully-summed range. The constraint column at `k` has
  - BK argmax `r` = a hub row (not fully summed),
  - `a[k,k+1] = 0` (the #46 fallback does not apply),
  - rook rescue starting at `k`, which walks to the hub, alternates among
    hub-dominated rows and fails.

  So the constraint delays too, and the pair never meets in a pivot test.
  Both columns ride up the chain to the front where the hubs become fully
  summed. That gives a quadratic Σ front² over a 428-front chain.
- The #215 budget (fronts may grow to 4096 rows) is what lets the cascade
  run. Under 0.18's 64-delay cap, cascade-break fired instead. That was
  fast, but its inertia came from floored pivots (6903 neg vs the eigvalsh
  oracle's 6901).

## Reference behaviour

SSIDS `ldlt_tpp.cxx` (ralna/spral 3d96a48):
- The 2×2 partner is `find_row_abs_max(nelim, p, ...)` (:207), restricted
  to uneliminated fully-summed columns. A non-fully-summed row is never a
  partner.
- The 1×1 and 2×2 tests use `find_rc_abs_max_exclude` (:75-86), whose
  maxima run over the full column to `m`. An entry in a ghost row only
  tightens the tests.

feral's partner logic took the global argmax as the partner. When that
row was a ghost it fell back only to `k+1`. A partner further down the
fully-summed range was never considered.

## Change

In `scalar_pivot_step`, record the fully-summed argmax (`gamma_fs`,
`r_fs`) during the existing γ₀ scan. When `r` is a ghost and `k+1` is
uncoupled, use `{k, r_fs}` as the 2×2 candidate, but only if
`rook::passes_2x2_gates(a, nrow, k, k, r_fs, u)` passes before any swap.
That function is the same Duff-Reid growth bound and SSIDS det floor as
the in-path test, evaluated in pre-swap coordinates. A rejected candidate
leaves the path byte-identical to before (rook-wrapped last-resort 1×1).
The stability gate is not relaxed.

## Evidence

| matrix | HEAD time | HEAD nnz(L) / neg | fix time | fix nnz(L) / neg | eigvalsh neg |
|---|---|---|---|---|---|
| brainpc3 i35/1 | 43.8 s (cb 1, tiny 277) | 23.29M / 6902 | 0.022 s | 443K / 6901 | 6900 † |
| brainpc3 i37/1 | 9.46 s | 12.88M / 6901 | 0.022 s | 402K / 6901 | 6901 |
| brainpc3 i38/1 | 2.92 s | 7.17M / 6900 | 0.023 s | 386K / 6901 | 6900 |
| brainpc3 i39/1 | 2.40 s | 6.90M / 6903 | 0.023 s | 378K / 6902 | 6902 |

† All four matrices are numerically singular. The fraction of
eigenvalues with |λ| < n·eps·‖A‖ is 129 (i37, i38), 7 (i39) and 13,795
(i35, ‖A‖ = 1.5e12). The project's exact-inertia constraint covers
non-singular matrices only, so no factorization's count is guaranteed on
these. The fix matches the oracle on 2 of 4 and HEAD matches on 2 of 4,
but HEAD's i35 count came from 277 floored pivots.

Normwise backward error, plain / refined:
- i37: HEAD 7.8e-17 / 7.8e-17, fix 7.3e-19 / 2.5e-25
- i38: HEAD 3.8e-15 / 2.6e-18, fix 7.6e-21 / 2.6e-26
- i39: HEAD 6.0e-16 / 5.5e-22, fix 8.7e-17 / 2.7e-23

Corpus (`data/matrices/kkt-pounce`, 370 matrices, pounce params): inertia
and n_tiny are identical on all 370. nnz(L) changes on 141 (corkscrw,
laptime) by −0.9 % to +0.03 %. Total factor time is 69.0 s on HEAD and
68.6 s with the fix.

Synthetic brainpc-shaped KKT (`tests/issue_221_fully_summed_2x2_partner.rs`):

| T | HEAD | fix |
|---|---|---|
| 500 | nnz(L) 256K | 19K |
| 2000 | 4.04M, 2.83 s | 78K, 0.0044 s |

The inertia is exact by the saddle-point theorem in both.
