# Plan — fix the zero-(2,2)-block KKT delayed-pivot cascade (issue #46)

Date: 2026-05-20
Research note: `dev/research/kkt-zero-2x2-block-cascade-2026-05-20.md`

## STATUS — original plan abandoned; actual fix shipped

> The original plan (below, struck through) was an **activation-predicate
> change** to `pick_ordering_preprocess`, on the three-agent diagnosis
> that #46 was an analysis-phase ordering failure. Ground-truth probes
> overturned that diagnosis: `LdltCompress` already activates on the CHO
> KKT and 96.6 % of MC64 pairs are already co-located. The original
> "Phase 1" was a no-op. See the research note for the full overturning.
>
> **The fix actually shipped** is the original plan's *Phase 2,
> sub-bullet 2* — the numeric 2×2 gate.

## Actual fix

`scalar_pivot_step` (`src/dense/factor.rs`) — widen the 2×2 partner
search. When BK's magnitude-argmax row `r` is not fully summed (an
out-of-front coupling), fall back to the literal next column `k+1` as
the 2×2 partner, guarded by `a[k,k+1] ≠ 0` (the columns are actually
coupled). `LdltCompress` co-locates the MC64-matched saddle partner at
`k+1`, so this is the numerically correct partner for a zero-diagonal
constraint column. The Duff–Reid growth bound and SSIDS det floor still
gate the 2×2 — the change widens the search, not the acceptance.

Diff: ~30 lines, one logical change. Bit-identical to the pre-#46 kernel
whenever `r` is fully summed or `k`/`k+1` are structurally uncoupled.

## Result

CHO KKT (production path): 11.7 s → 0.204 s (57×), `factor_nnz`
28.05M → 3.35M, inertia `(21672, 21660, 0)` unchanged. feral now ~3×
MA57 (was ~160×).

## Tests (committed)

`tests/issue_46_saddle_kkt_cascade.rs` — synthetic structurally-zero-
(2,2)-block saddle KKT, `n = 500`, constraints-first layout, `Identity`
scaling (isolates the kernel). Oracles: inertia `(nv, nc, 0)` by the
saddle-point inertia theorem; solve residual `‖Ax−b‖/‖b‖ ≤ 1e-8`;
`factor_nnz ≤ 5 ×` the symbolic estimate (cascade guard). Verified
against a temporarily-reverted kernel: pre-#46 → 61× blowup, test fails;
fixed → 0.83×, test passes.

## Out of scope (unchanged from original)

- A from-scratch 2×2 saddle kernel — feral's kernel already handles
  `[[0,b],[b,·]]`.
- `pinene_3200` end-to-end IPM-loop validation — follow-up.
- The MC64 scaling-spread issue (#45) — already fixed (`b017beb`).

---

## ~~Original plan (abandoned — diagnosis was wrong)~~

~~Phase 1 — activation. Broaden `pick_ordering_preprocess` with a
zero/absent-diagonal-fraction predicate (MUMPS-style, threshold ≈ 0.10
of `n`) so `LdltCompress` turns on. **Abandoned: `LdltCompress` was
already on for the CHO KKT — feral stores only the lower triangle, so
constraint columns are stored-degree 0/1 and the existing
`low_degree(≤2)` predicate already fires (frac 0.7505).**~~

~~Phase 2 — only if the probe shows the cascade persists.~~
~~- Supernode split guard in `find_supernodes`. **Not needed: 96.6 % of
  pairs already co-located.**~~
~~- Numeric 2×2 gate in `scalar_pivot_step`. **This was the real fix —
  see "Actual fix" above.**~~
