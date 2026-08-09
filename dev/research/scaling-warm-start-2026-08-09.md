# Scaling warm-start (post-0.15.0 item 1) — hypothesis FALSIFIED, with a better diagnosis

Session 2026-08-09-04. Motivation: the PR #150 review profiled six real
KKT matrices and found the **warm** prologue at 15–39% of factorization
with ∞-norm equilibration at 63–81% of that — re-derived from scratch on
every call. On `rocket_12800` scaling alone is 34% of the whole factor.
The reviewer's clean observation: across calls 0→2 the prologue *does*
warm for everything else (`permute` collapses 4443→539 µs on clnlbeam,
25647→1584 µs on dtoc2) while `scaling` stays flat (3909 / 19850 µs).

Hypothesis: KKT values drift smoothly across IPM iterations, so seeding
the iteration with the previous factorization's converged `d` should cut
the iteration count and hence the cost.

**The hypothesis is false, and the reason is more useful than the fix
would have been.**

## Measurement

`examples/probe_kr_warmstart.rs` (checked in): run the exact iteration
from `scaling::infnorm::compute_infnorm`, perturb every value by ±5%
(standing in for one IPM step), then re-run cold vs warm-started from
the previous `d`, counting iterations.

| fixture | n | cold iters | **warm iters** | cold-vs-warm max rel diff |
|---|---:|---:|---:|---:|
| clnlbeam-like (tridiagonal) | 100,000 | 2 | **2** | 2.2e-16 |
| grid250 (Laplacian) | 62,500 | 2 | **2** | 0 (bit-identical) |
| chain12000 (hicks-like) | 12,000 | 10 | **10** | 5.1e-2 |
| sparseqpL (banded KKT) | 105,000 | 10 | **10** | 5.1e-2 |
| HYDCAR20 | 198 | 10 | **10** | 6.3e-2 |
| twirism1_kkt | 745 | 10 | **10** | 5.1e-2 |

Zero iteration reduction anywhere. Two regimes, neither helped:

- **Already minimal (2 iterations).** Nothing to save.
- **Capped at 10 without converging.** The 5e-2 cold-vs-warm spread is
  the tell: both runs are on the *same* matrix, so if either had
  converged they would agree to ~`tol`. They differ by 5%, so neither
  is near a fixed point.

## Why: the tolerance is unreachable by construction

Per-iteration `max_dev` on the capped matrices (`KR_TRACE=1`):

```
HYDCAR20            twirism1_kkt
iter  1: 7.28e+1    iter  1: 1.05e+2
iter  2: 9.82e-1    iter  2: 9.90e-1
iter  3: 8.52e-1    iter  3: 9.02e-1
iter  4: 6.06e-1    iter  4: 6.87e-1
iter  5: 3.73e-1    iter  5: 4.41e-1
iter  6: 2.08e-1    iter  6: 2.52e-1
iter  7: 1.10e-1    iter  7: 1.35e-1
iter  8: 5.66e-2    iter  8: 7.01e-2
iter  9: 2.87e-2    iter  9: 3.57e-2
iter 10: 1.45e-2    iter 10: 1.80e-2
```

Clean linear convergence at ratio ≈ 1/2 per iteration — the known rate
for Ruiz-style ∞-norm equilibration (`d ← d / sqrt(row_max)`), which is
what `compute_infnorm` implements regardless of the "Knight-Ruiz" label
in the module doc. From `max_dev ≈ 1e-2` at the cap, reaching
`tol = 1e-8` needs ~20 more iterations, i.e. **~30 total**.

So `max_iter = 10` with `tol = 1e-8` means: on any matrix not already
near-equilibrated, the tolerance is unreachable and the loop *always*
runs the full 10 passes. Warm-starting cannot reduce a fixed iteration
count. The in-code comment — "Most matrices converge in 2–4 iterations;
a few pathological ones need all 10" — understates it: on this fixture
set 4 of 6 hit the cap, and they do not "need" 10, they are *truncated*
at 10.

## What this means for the optimization

The cost is **10 fixed passes over the matrix** on the hard class. To
make scaling cheaper you must do fewer passes, which is a
conditioning/speed trade, not a free win:

| cap | max_dev reached (HYDCAR20) | relative equilibration quality |
|---:|---:|---|
| 10 (today) | 1.4e-2 | baseline |
| 5 | 3.7e-1 | **26× worse** |
| 3 | 8.5e-1 | 60× worse |

Halving the cap would buy roughly half of a 63–81% share of a 15–39%
prologue — call it 5–15% of factor time — at the cost of a 26× worse
equilibration on exactly the ill-conditioned matrices the scaling
exists to rescue. That is a bad trade on its face and would have to be
justified by corpus residual/inertia data, not by timing alone.

## Options, ranked

1. **Do nothing (recommended for now).** The measurement says the
   planned free win does not exist. Item (3) should be re-scoped or
   dropped rather than implemented as designed.
2. **Warm-start for *quality*, not speed.** Seeding from the previous
   `d` makes the truncated iteration a *continued* one across the IPM:
   same 10 passes, but each factorization starts further along, so the
   scaling improves monotonically over the solve. Free conditioning,
   zero speed change, and it could reduce downstream iterative-
   refinement work — which is where it would actually pay. Speculative
   until measured against refinement counts.
3. **A faster-converging algorithm.** Ruiz's 1/2-linear rate is the
   binding constraint. A Newton/accelerated variant (true Knight-Ruiz)
   would reach tolerance in far fewer passes. Real work, and it changes
   numerics substantially.
4. **Lower the cap.** Cheapest to implement, worst risk/benefit; needs
   corpus residual data before it could be considered.

## Constraint on all of them

Every option changes numerical output — different `d` → different
`D·A·D` → potentially different pivots. Unlike everything in 0.15.0,
the `golden_bits` / parity apparatus does **not** gate this. The gate is
a full corpus run (inertia 100%, Phase 2.8.1 partitions, residuals),
which the dev container does not have.

## Reproducer

```sh
cargo run --release --example probe_kr_warmstart -- <matrix.mtx> [drift]
KR_TRACE=1 cargo run --release --example probe_kr_warmstart -- <matrix.mtx>
```
