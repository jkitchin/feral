# Issues #45 / #46 — MC64 symmetric scaling blows up on saddle-point KKTs

Date: 2026-05-20
Author: agent session 2026-05-20-02
Issues: https://github.com/jkitchin/feral/issues/45 (correctness),
        https://github.com/jkitchin/feral/issues/46 (performance)

## Summary

On the CHO `parmest` KKT (n=43332, symmetric saddle point, cond₂≈1.4e15,
non-singular), feral's `ScalingStrategy::Mc64Symmetric` produces a
scaling vector spanning ~3e82. Applying it underflows the factorization;
Bunch-Kaufman accepts exact-`0.0` pivots; the solve is garbage (relative
residual ~7e11) while `factor()` returns `Success` with correct inertia.
That silent wrong answer is **#45**.

`#45` and `#46` are the same root cause seen through two scaling
strategies, not — as the journal initially hypothesised — a fill-reducing
ordering defect. The ordering, elimination tree, column counts and
supernode structure are all **diagonal-insensitive** (verified below).

## Decisive experiments

`src/bin/probe_issue45_ordering.rs` — cross numeric input (the dumped
stripped `.mtx` vs the same matrix with an explicit `0.0` diagonal
completed) with an explicit scaling strategy:

| input     | scaling        | factor   | factor_nnz | min pivot | rel res  |
|-----------|----------------|----------|-----------:|-----------|----------|
| stripped  | Identity       | 39838 ms |  58163992  | 3.54e-8   | 8.19e-10 |
| stripped  | InfNorm        | 11205 ms |  28054562  | 1.96e-9   | 2.46e-8  |
| stripped  | Mc64Symmetric  |   220 ms |   4436782  | **0.0**   | **7.15e11** |
| completed | Identity       | 39941 ms |  58163992  | 3.54e-8   | 8.19e-10 |
| completed | InfNorm        | 11104 ms |  28054562  | 1.96e-9   | 2.46e-8  |
| completed | Mc64Symmetric  |   224 ms |   4436782  | **0.0**   | **7.15e11** |

The stripped/completed rows are **bit-identical** for every scaling.
Diagonal completion has zero numeric effect. Its only effect is on the
`ScalingStrategy::Auto` router (see below).

Earlier crossed-symbolic run (same probe, prior revision): a symbolic
factorization built from the stripped pattern and one built from the
completed pattern produce **bit-identical** numeric factors on the same
input — confirming `EliminationTree::from_pattern` (skips `i>=j`) and
`column_counts_gnp` (skip `i<=j`) are provably diagonal-insensitive, and
so are the supernodes derived from them.

## Root cause

`src/scaling/mc64.rs::scaling_from_cache` forms, per index `i`:

    s[i] = exp((u[i] + v[i] - cmax[i]) / 2)

from the Hungarian assignment dual potentials `u`, `v`. The only guard
is a per-entry overflow clamp `arg.clamp(-LOG_HUGE, LOG_HUGE)` with
`LOG_HUGE = 709` (the `exp` overflow ceiling).

On the CHO KKT the potentials are large but finite — `arg ≈ ±95` — so
the clamp never fires. The resulting scaling vector:

    Mc64Symmetric:  min = 2.891e-42   max = 8.878e40   spread ≈ 3.07e82
    InfNorm:        min = 1.552e-3    max = 4.576e0    spread ≈ 2.95e3

A scaling whose **own spread exceeds `1/EPS ≈ 4.5e15`** is numerically
degenerate: `D = diag(s)` is singular to working precision, so the
un-scaling step `x = D x̂` annihilates every component scaled by `s_min`.
Worse, `D·A·D` has a dynamic range of `spread² ≈ 1e165`; its small
entries underflow during the Schur-complement updates, BK is handed
all-zero pivot columns, force-accepts them (`ZeroPivotAction::ForceAccept`
zeros the column and counts inertia by sign), and the solve is garbage.
`info` is `Applied` — MC64 fully matched the symmetric pattern
(`n_matched == n`), so there is no `PartialSingular` signal either.

Why MC64 blows up here specifically: the saddle point `[H Bᵀ; B 0]` has
a structurally-zero `(2,2)` block. The symmetric matching matches dual
variables to off-diagonal `B` couplings instead of to a diagonal; making
those off-diagonals unit-scale forces extreme, path-accumulated dual
potentials. This is inherent to MC64-*scaling* on saddle points — it is
not a coding bug in the Hungarian kernel.

## Corpus measurement — where the guard threshold goes

`src/bin/probe_mc64_spread.rs` measured MC64 vs InfNorm scaling spread
on all 38 `tests/data/parity` families. Every legitimate matrix has
MC64 spread well under `1/EPS`:

    ssine      3.27e15    hatfldbne  1.33e15    muonsine  1.35e12
    vesuviou   5.34e11    himmelbj   7.73e10    vesuvia   1.95e10

The CHO catastrophe (3e82) sits in a **67-order gap** above the entire
corpus. A guard at `1/EPS ≈ 4.503e15` catches CHO and clears every
corpus matrix (max 3.27e15 < 4.503e15). It is anchored to a hard
numerical invariant, not fitted to the corpus.

## Why the existing issue-#24 guard misses it

`compute_scaling_auto_with_cache` already has a Policy-4 fallback that
throws away an MC64 scaling "catastrophically worse than InfNorm". It
does not fire here because of a fast-path:

    if raw_diag_range(matrix) >= RAW_GUARD (1e6) {
        return mc64_from_cache(matrix);   // commits to MC64 unchecked
    }

The CHO KKT is genuinely ill-conditioned (`raw_diag_range ≥ 1e6`), so it
takes this fast-path and the `mc_off` catastrophe diagnostic downstream
is never reached. The fast-path assumes "wide raw range ⇒ MC64 has
genuine work to do ⇒ trust it" — CHO is the counter-example.

## Why the real POUNCE run hits #45 but the dumped `.mtx` does not

The dumped `cho_iter0_kkt.mtx` stores only 10 812 of 43 332 diagonal
entries; the explicit zero diagonals were dropped on the way to disk.
That stripped matrix routes through the `Auto` scaling router to
**InfNorm** (`pick_scaling_strategy` counts diagonal-only columns;
without the diagonal the saddle signature is invisible) — and InfNorm
factors it correctly (2.46e-8). POUNCE's live KKT assembly emits a
complete diagonal; that matrix routes to **Mc64Symmetric** → blow-up →
#45. This reconciles the 2026-05-20 14:05 "cannot reproduce on the
dumped file" finding with the live failure.

## Fix (this session — correctness, #45 only)

Add a catastrophic-spread guard to `compute_scaling_auto_with_cache`:
after computing the MC64 vector, if `scaling_spread(&mc_vec) >
MC64_SPREAD_GUARD` (= `1.0 / f64::EPSILON`), discard it and fall back to
the already-computed InfNorm vector, tagged
`ScalingInfo::Mc64FallbackToInfnorm { reason:
Mc64FallbackReason::Mc64ScalingDegenerate }`. The check is placed
**before** the `raw_diag_range` fast-path so it fires regardless of raw
conditioning. Cost is one O(n) pass over a vector already in hand.

Effect: the CHO KKT (in either diagonal form) routes `Auto → MC64
attempted → spread 3e82 caught → InfNorm`, giving the correct solve
(rel res 2.46e-8). #45 closed.

## Not fixed here — #46 (performance)

After the guard, the CHO KKT solves **correctly** but via InfNorm: 28M
factor nonzeros, ~11 s. The 50× speedup (4.4M nnz, ~220 ms — the fill
MA57 also achieves) is only reachable with a matching-based scaling, and
feral's `Mc64Symmetric` cannot deliver one on saddle points without the
blow-up. Genuinely fixing #46 requires either (a) a saddle-point-stable
matching scaling, or (b) the MC64 *permutation* (large entries onto the
diagonal) that MA57/MUMPS apply, which feral does not currently do. That
is a separate, larger effort and is intentionally out of scope for the
correctness fix. Constraint: "correctness before performance, always."
