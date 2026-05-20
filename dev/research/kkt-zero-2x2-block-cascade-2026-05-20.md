# Research — delayed-pivot cascade on zero-(2,2)-block KKT (issue #46)

Date: 2026-05-20
Issue: #46 (LDLᵀ ~160× slower than MA57 on the CHO `parmest` IPM KKT;
also `pinene_3200`, issue #8).
Related: #45 (the correctness face — now fixed, `b017beb`).

## STATUS — diagnosis corrected, fix shipped (2026-05-20)

> **The original three-agent diagnosis in this note (an "analysis-phase
> ordering failure", fixed by an activation-predicate change) was wrong
> in every load-bearing claim.** Ground-truth probes on the real CHO
> KKT overturned it. The real bug is a *numeric-kernel* failure: the
> Bunch-Kaufman 2×2 partner selection is matching-unaware. The fix is
> in `scalar_pivot_step`, not in the ordering. The disproven sections
> are kept below under "Superseded diagnosis" for the record; the
> corrected diagnosis follows immediately.

## The bug

A saddle-point / interior-point KKT has the block structure

    K = [ H   Bᵀ ]
        [ B   0  ]

The `(2,2)` block is **structurally zero**: every constraint column has
a zero diagonal. On the CHO KKT (`n = 43332`, 205 512 nnz) ~21 660 of
the columns are such constraint columns.

A column with a zero diagonal cannot be a 1×1 pivot. feral's
Bunch-Kaufman kernel therefore **delays** it up the elimination tree.
The delays cascaded into a 28M-factor-nonzero, ~17 s factorization
where MA57 does ~70 ms — end-to-end the POUNCE NLP solve was ~160×
slower.

## Corrected diagnosis — a NUMERIC-kernel failure (gap 3)

Two ground-truth probes (`src/bin/probe_issue46_preprocess.rs`,
`src/bin/probe_issue46_supernode.rs`) on the real CHO KKT:

**The ordering machinery already works.** feral stores only the lower
triangle, so a KKT constraint column is stored-degree 0/1, *not* high
degree. `pick_ordering_preprocess`'s existing `low_degree(≤2)` predicate
fires (frac 0.7505) and **already** returns `LdltCompress` for the CHO
KKT. `build_supermap` **already** forms 21 660 MC64 pairs. Activation
is a no-op "fix" — there was nothing to activate.

**The pairs are co-located.** With `preprocess = LdltCompress` the
probe measured: symbolic `factor_nnz_estimate = 1.22M` (the no-delay
fill prediction), max supernode `ncol = 133` (no giant root
supernode), and **20 918 / 21 660 MC64 pairs land in the same
supernode, 20 794 of them at adjacent columns (96.6 % co-located)**.
The ordering places each saddle pair exactly where it should.

**Yet the numeric factor blows up 23×.** Same run: numeric
`factor_nnz = 28.05M`, `n_delayed = 103 110`. The symbolic ordering
predicts 1.22M fill; the numeric phase produces 28M. The cascade is
**not** an ordering failure and **not** a supernode-split failure — it
is purely a numeric delayed-pivot blowup.

**Root cause.** The numeric Bunch-Kaufman kernel re-derives every
pivot blind via magnitude-argmax and never consults the analysis-phase
MC64 pairing. `scalar_pivot_step` reached its 2×2 branch only when the
BK argmax row `r` was *fully summed* (`r < ncol`). For a zero-diagonal
constraint column whose largest coupling points at an **out-of-front
row** (`r ≥ ncol` — a trailing, not-yet-eliminated row), the kernel:
could not form a 2×2 (`r` not fully summed), could not 1×1 the zero
diagonal — so it **delayed the column**, even though the co-located
MC64-matched partner sat at the adjacent fully-summed column `k+1`.
panel_diag on the cascade run: `fallback_2x2_need_swap_or_bound =
14 584` is the dominant fallback — the panel inline-2×2 path declining
because `r ≠ k+1`.

**Why MUMPS/MA57 do not hit this.** They apply MC64 *scaling* (not
just matching): scaling makes every matched entry magnitude ≈ 1, so
BK's argmax always lands on the partner. feral's MC64 scaling is
degenerate on saddle systems (#45, spread ~3e82) and is correctly
rejected, falling back to InfNorm. So feral cannot lean on the scaling
mechanism — it has the matching *ordering* but lacked the matching-aware
*numeric pivot selection*. That is gap 3, and it is the whole bug.

**Control — `allow_delayed_pivots = false`.** Static pivoting breaks
the cascade (27 ms, 1.49M fill ≈ symbolic estimate) but gets the
inertia **wrong** — `(15422, 27910, 0)` vs the correct
`(21672, 21660, 0)` — because force-accept assigns arbitrary signs to
~6 250 saddle pivots. Not an admissible fix; it violates the hard
"inertia exactly correct on non-singular matrices" constraint. That is
exactly why it is an opt-in knob, not the default.

## The fix (shipped)

`scalar_pivot_step` (`src/dense/factor.rs`) now selects the 2×2 partner
explicitly:

    partner = if r_is_fully_summed && k + 1 < ncol { Some(r) }
              else if k + 1 < ncol && a[k, k+1] != 0.0 { Some(k + 1) }
              else { None };

When BK's magnitude-argmax `r` is fully summed it remains the textbook
2×2 partner. When it is *not* (an out-of-front coupling), the kernel
falls back to the literal next column `k+1` — provided `k` and `k+1`
are actually coupled (`a[k,k+1] ≠ 0`). The `LdltCompress` analysis
phase co-locates every MC64-matched saddle pair at adjacent
fully-summed columns, so for a zero-diagonal constraint column `k+1` is
the numerically correct partner.

This widens the 2×2 *search*, it does not relax the stability gate: the
candidate `{k, k+1}` is still subject to the Duff–Reid growth bound and
the SSIDS determinant floor below; an unsound candidate fails those and
falls through to the last-resort 1×1 exactly as before. The
`a[k,k+1] ≠ 0` guard keeps the path bit-identical to the pre-#46 kernel
for every structurally-uncoupled neighbour.

## Verification

CHO KKT, `LdltCompress` + default `allow_delayed_pivots = true` (the
production path), `probe_issue46_supernode`:

| metric        | before  | after   |
|---------------|---------|---------|
| factor time   | 11.7 s  | 0.204 s | 57× faster
| `factor_nnz`  | 28.05M  | 3.35M   |
| `n_delayed`   | 103 110 | 17 431  |
| `n_2x2`       | 11 736  | 16 850  |
| inertia       | (21672, 21660, 0) — correct, unchanged |

feral is now ~3× MA57 instead of ~160×.

Committed regression test: `tests/issue_46_saddle_kkt_cascade.rs`. A
synthetic structurally-zero-(2,2)-block saddle KKT (`n = 500`,
constraints-first layout, global-variable coupling, `Identity` scaling
to isolate the kernel). Verified against a temporarily-reverted kernel:
pre-#46 cascades it to a 61× fill blowup (`n_delayed = 398`,
`n_2x2 = 0`) — test fails; the fixed kernel holds it at 0.83×
(`n_delayed = 0`, `n_2x2 = 199`) — test passes. Inertia is
`(300, 200, 0)` in both: the cascade is slow-but-correct, so the fill
bound, not the inertia, is what catches the regression.

## Oracles (test policy)

- Residual `‖A·x−b‖ / ‖b‖` — a mathematical identity, admissible.
- Inertia on a non-singular synthetic saddle KKT — `(nv, nc, 0)` by the
  saddle-point inertia theorem (`H` SPD + `B` full row rank; external
  math, e.g. Benzi, Golub & Liesen, Acta Numerica 2005, §3.4).
- `factor_nnz ≤ 5 × factor_nnz_estimate` — a measurement oracle / cascade
  guard; the 5× bound separates the 0.83× healthy factor from the 61×
  cascade decisively.

## Key files

- `src/dense/factor.rs` — `scalar_pivot_step`, the 2×2 partner
  selection (THE FIX).
- `src/bin/probe_issue46_preprocess.rs` — refutes "gap 1 = activation".
- `src/bin/probe_issue46_supernode.rs` — refutes "gap 2 = supernode
  split"; the decisive co-location + numeric-blowup measurement.
- `tests/issue_46_saddle_kkt_cascade.rs` — committed regression test.
- `src/symbolic/ldlt_compress.rs` — match / compress / expand (already
  correct; not modified).

---

## Superseded diagnosis (the original three-agent analysis — WRONG)

The text below is the original research note, retained for the record.
**Every load-bearing claim in it is disproven by the probes above.**
Specifically:

- "This is an ANALYSIS-phase failure / ordering failure" — wrong; the
  ordering predicts 1.22M fill, the numeric phase produces 28M.
- "Gap 1 — activation is the load-bearing gap" — wrong;
  `LdltCompress` already activates on the CHO KKT (constraint columns
  are stored-degree 0/1, not high-degree).
- "Gap 2 — co-location may be lost at supernode boundaries" — wrong;
  96.6 % of pairs are co-located at adjacent columns.
- "Phase 1 (activation predicate) is the whole fix" — wrong; the fix
  is the numeric gap-3 kernel change.
- "matching-based ordering *is* the fix" (the MUMPS/SSIDS conclusion)
  — incomplete; MUMPS/SSIDS also rely on MC64 *scaling*, which feral
  cannot use on saddles. feral needs the matching-aware *kernel*.

The MUMPS/SSIDS reference reading below is itself accurate and remains
useful background; only the conclusion drawn from it ("the fix belongs
in the ordering, not the kernel") was wrong.

### [superseded] Diagnosis: this is an ANALYSIS-phase failure

A zero-diagonal column *can* be eliminated immediately as one half of a
2×2 pivot `[[0, b],[b, a]]` — provided its 2×2 partner is fully summed
in the same frontal matrix. MUMPS 5.8.2 (`DMUMPS_SYM_MWM` +
`DMUMPS_LDLT_COMPRESS`, auto-activated for `SYM=2` when the zero-diagonal
count exceeds `N/10`, `dana_aux.F:1886`) and SPRAL SSIDS
(`options%ordering = 2`, `src/match_order.f90`) both pair each
zero-diagonal constraint with a coupled variable, compress the graph
onto those pairs, run the fill-reducing ordering on the compressed
graph, and expand keeping each pair adjacent — so both members are
fully summed in the same front and the numeric kernel forms the 2×2
in place. The original note concluded the fix was to broaden feral's
`pick_ordering_preprocess` activation predicate (a MUMPS-style
zero-diagonal-fraction test, threshold ≈ 0.10 of `n`) so `LdltCompress`
would turn on. **This was wrong: `LdltCompress` was already on.**
