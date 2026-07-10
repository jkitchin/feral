# Issue #133 (dynamic Markowitz LU factor) — measure-first — 2026-07-10

**Verdict: not justified as a fill lever.** feral's current sparse-LU column
order (AMD on the AᵀA pattern) is already competitive with SuperLU's COLAMD on
discopt's real set-covering bases; there is no large fill headroom for a
Markowitz rewrite to capture, and the cheap Stage-1 threshold flip does not
reduce fill. Recorded per the measure-first discipline (cf. #129, #132).

## Context

#133 proposes replacing feral's static AMD-on-AᵀA column order + strict
threshold-partial pivoting (`pivot_threshold = 1.0`) with dynamic Markowitz
pivoting (Markowitz count × threshold, `u ∈ [0.01, 0.1]`), staged:
- **Stage 1 (cheap):** lower `pivot_threshold` to ~0.1 so the factor prefers
  the sparser within-threshold (diagonal) pivot.
- **Stage 2 (big rewrite):** right-looking Markowitz factorization.

Since discopt's simplex bottleneck is `compute_spike`'s L-solve over the
factor (cost *tracks fill* — see the #132 note), less fill ⇒ cheaper updates.
So #133's payoff = the fill it removes.

## Stage 1 — pivot_threshold sweep (`src/bin/probe_ft_eta.rs`, `FERAL_PIVTOL`)

**casctanks (real, m=2169):** `factor_nnz` 19888 (t=1.0) → 19896 (0.1) →
19897 (0.01). **No change.** Residual stable ~1e-12. The AMD-on-AᵀA order is
already good; strict pivoting causes no excess fill there.

**Real sc2000x800 transient bases (discopt example, `FeralLU::params` patched
to read `FERAL_PIVTOL`):** noisy, no consistent reduction —
```
depth 400:  t=1.0 → 4.5   t=0.1 → 6.7   t=0.01 → 6.7   (worse)
depth 800:  t=1.0 → 18.5  t=0.1 → 14.4  t=0.01 → 17.1  (mixed)
depth 1600: t=1.0 → 31.1  t=0.1 → 35.2  t=0.01 → 27.6  (mixed)
```
Update time barely moves. **Stage 1 is not a win** — the pivot threshold only
affects within-column pivot choice, a second-order effect on fill here.

## Stage 2 ceiling — external oracle (SuperLU/COLAMD via scipy 1.17.1)

Dumped discopt's *real* transient covering bases and factored the identical
matrix with SuperLU under several column orders (`factor_nnz/m`):

```
depth 1600 (A_nnz/m=3.7):  NATURAL 138.7 | MMD_ATA 37.6 | MMD_AT+A 33.8 |
                           feral AMD-on-AtA 31.1 | COLAMD 25.5   (COLAMD +18%)
depth 800  (A_nnz/m=3.1):  feral 18.5 | COLAMD 17.6                (COLAMD  +5%)
depth 400  (A_nnz/m=2.2):  feral  4.5 | COLAMD  5.7                (COLAMD −27%)
```

feral's AMD-on-AᵀA is **far better than natural (138.7), on par with the MMD
variants (33.8–37.6), and within ~5–18% of COLAMD on the heaviest transient
bases while beating COLAMD on the light ones.** Net across depths it is roughly
a wash with SuperLU's default.

## Interpretation

1. The heavy transient fill (18–31 nnz/row from a 3–4 nnz/row input) is largely
   **inherent to the covering-basis structure**, not an artifact of a bad
   column order: the best static order feral could adopt (COLAMD) is only
   ~modestly better on the worst bases and worse on others.
2. Dynamic Markowitz (value-aware) could edge past COLAMD on the heaviest
   bases, but the COLAMD-≈-feral result bounds that headroom low (single-digit
   to ~18% on the worst bases, negative on light ones) — not the "substantially
   less fill" the issue speculated.
3. Stage 1 (threshold) delivers nothing measurable.

## Decision

Do not implement #133. feral's LU fill is already competitive with
SuperLU/COLAMD on discopt's covering bases; the Markowitz rewrite's ceiling is
a modest, inconsistent fill change for a large clean-room effort. Re-open only
with an LP class where feral's AMD-on-AᵀA fill is shown (via this same SuperLU
oracle) to be materially worse than COLAMD across the solve, not just on the
single heaviest transient basis.

## Caveats

- COLAMD is a strong *static* order, used here as the Markowitz proxy; a
  value-aware dynamic Markowitz could do slightly better, but the wash-vs-COLAMD
  result is a tight upper bound on how much.
- Set-covering is discopt's stated expensive case; other LP structures may
  differ — the SuperLU oracle probe makes re-checking any new class cheap.

## Combined discopt-simplex finding (#130 / #132 / #133)

All three LU-side SOTA items come back not-justified for discopt's simplex:
#130 (hyper-sparse solves) — solves are 200–476× cheaper than a refactor;
#132 (permute-only update) — eta is 0.1–5% of update work; #133 (Markowitz) —
fill already near COLAMD. The `compute_spike` L-solve over an already-well-
ordered factor is near the achievable floor; remaining wins are on the simplex
algorithm (pivots/pricing) or refactor cadence, not the LU kernels.
