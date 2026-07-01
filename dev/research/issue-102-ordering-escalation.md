# issue #102 follow-up — ordering quality: escalate on pivot growth — 2026-07-01

The #102 deadlock fix (PR #104) unblocked cont5_2_4_l/dirichlet120, and revealed
a **second, distinct regression** on cont5_2_4_l: converged in 14 iters on feral
0.11.3, but on `main` the IPM oscillates near the solution (inf_du floor ~1e-5,
restoration) and hits the CPU cap. Isolated (by the reporter and confirmed here)
to PR #92's `OrderingPreprocess::Auto` verify — not #99, not the deadlock fix
(dirichlet120 is bit-identical to 0.11.3).

## Root cause — fill is the wrong axis for the LdltCompress decision

`OrderingPreprocess::Auto` keeps `LdltCompress` only if its symbolic fill ≤ 2×
`None`. But `LdltCompress`'s value is **numerical**, not fill: its MC64 matching
pairs near-singular ±ε diagonals into stable 2×2 pivots. cont5's KKT has ~half
its diagonals at ∓1e-8 (μ-regularization); as the IPM drives μ→0 the diagonals
approach zero. On the late KKTs, dropping `LdltCompress` for fill leaves `None`
making ~1e-16 1×1 pivots.

Measured (feral-level, dumped cont5 KKTs; `examples/cmp_ordering_accuracy`):

| cont5 KKT | ordering | pivot growth | refined resid |
|---|---|---|---|
| iter 0 | None | 7.5e19 | 1.5e-11 (recoverable) |
| **late** | **None** | **4.1e32** | **1.4e-2** (refinement can't recover) |
| late | LdltCompress | 1.4e15 | 3.2e-16 |

The late-KKT None factor is garbage (growth 4e32); refinement floors at 1.4e-2 →
inaccurate IPM step → no convergence. Exactly the observed inf_du ~1e-5 floor.

## Why no *symbolic* / cheap gate works

None of fill-ratio, first-KKT residual, or predicate cleanly separates "None is
fine" from "None is broken": both qap15 and cont5 fire the predicate (≈50 %
low-degree cols), both have huge pivot growth, and both first-KKTs recover under
refinement. **Only the late-iteration factor's pivot growth distinguishes them**,
and only per-factor (iter 0 is fine, iter 12 is broken). So the decision must be
made numerically, per factor.

| matrix (None) | growth | converges? |
|---|---|---|
| dirichlet120 | ~2 | ✅ |
| qap15 | 4.5e18 | ✅ |
| cont5 iter 0 | 7.5e19 | ✅ |
| cont5 late | **4.1e32** | ❌ |

12-order gap between "fine" (≤7.5e19) and "broken" (4.1e32).

## Fix — per-factor ordering escalation on pivot growth (feral Solver)

`Solver::factor`: after the numeric factor, if the caller requested `Auto`, the
resolved preprocess was `None`, the predicate wanted `LdltCompress`, and the
factor's `max|piv|/min|piv|` exceeds `ordering_escalation_growth` (default
`1e24`, in the gap), re-factor with `LdltCompress` and latch it for the pattern
(reset on pattern change). Growth is checked first (cheap, from `FactorStats`);
the O(nnz) predicate probe runs only in the rare high-growth case. Clears just
the symbolic cache (keeps the fingerprint, so the latch survives the recursive
re-factor).

Design choices:
- **feral, not pounce** (per maintainer): self-contained, no IPM feedback needed.
- **Only `Auto`**: an explicit `None`/`LdltCompress` request is respected.
- **Per-factor, latched**: cont5's early iters (growth ≤7.5e19) stay on fast
  `None`; the late iter escalates to `LdltCompress` and stays there. qap15 /
  dirichlet120 never escalate → keep the #92 fast path.
- Configurable/disableable via `Solver::with_ordering_escalation(Option<f64>)`.

Validated (`examples/cmp_ordering_accuracy`): late cont5 Auto now matches
LdltCompress (growth 1.4e15, refined 3.2e-16); qap15 / cont5-iter0 unchanged
(fast None). Byte-exact for the non-escalated path; full lib 394/0,
parallel_parity/issue91/issue65/ldlt_compress/symbolic_profiler all green.
Regression guard `tests/issue102_ordering_escalation.rs` (gitignored fixture
`tests/data/large/cont5_late_kkt.mtx`; dump via POUNCE `POUNCE_DBG_KKT_DUMP` +
`POUNCE_DBG_KKT_DUMP_SKIP=12` on cont5_2_4_l.nl).

## Residual note

The growth ceiling is a heuristic (a pivot-growth proxy for solve inaccuracy),
not a proof. It cleanly separates the known corpus with a 12-order margin; a
future refinement could gate on an actual refined-solve backward error if a
matrix is found where growth mis-predicts. Tracked as a possible follow-up.
