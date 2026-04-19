# POLAK6_0021 Triage — MC64 Inertia Regression Root Cause

**Date:** 2026-04-19
**Authorized by:** `dev/research/lever-c-corpus-bench-2026-04-19.md` §"What unblocks Policy 4".
**Diagnostic source:** `src/bin/polak6_diag.rs` (this session, throwaway).
**Sidecar:** `data/matrices/kkt/POLAK6/POLAK6_0021.{json,feral.json,mumps.json,verdict.json}`.

## TL;DR

POLAK6_0021 is a **mathematically ambiguous matrix** at floating-point
precision, not a "matched but bad" MC64 failure. The verdict file
classifies it as `excluded` from consensus inertia: rmumps, MUMPS,
SSIDS, and feral all give *different* answers to the same factor.
The corpus-bench note's framing of POLAK6 as a Policy-4 target was
based on the wrong assumption that the rmumps sidecar was ground
truth. It is not.

The lever-C measurement note's recommendation (do not flip default)
still stands, but for a different reason than originally stated. The
follow-up plan needs revision before Policy 4 is worth pursuing.

## 1. The matrix is fundamentally indeterminate

### Raw shape

```
n=9 stored_nnz=32
diag_only = 4 / n = 0.444 (≥ 0.30 → adaptive routes to MC64)
raw |diag|: min=1.000e-4  max=1.326e42  range=1.325e46
raw diag: [1.46e33, 2.69e4, 2.69e4, 1.32e42, 2.69e4,
           -1e-4, -1e-4, -1e-4, -1e-4]
```

This is an IPM iteration where `delta_w = 26843.5` (per the sidecar
`json` file) and the primal-dual bound multipliers have grown to
`1e33` and `1e42`. The diagonal spans **46 orders of magnitude.**
That is past double-precision dynamic range (~308 orders) but
past *useful* condition-number range (~16 digits). Any congruence
transform that touches 5+ matched diagonals can only preserve 16
digits of information; the rest is round-off noise.

### Oracle disagreement (per `.verdict.json`)

| oracle  | inertia      | residual   |
|---------|--------------|-----------:|
| rmumps  | (5, 4, 0)    | (n/a)      |
| feral   | (5, 1, 3)    | 1.90e-16   |
| MUMPS   | (5, 1, 3)    | 1.90e-16   |
| SSIDS   | (6, 3, 0)    | 3.82e-16   |

```
"verdict": "excluded",
"inertia_agreement": "none",
"feral_match_inertia": false,
"feral_residual_pass": true
```

feral's InfNorm baseline produces inertia `(5, 1, 3)` — *the same as
MUMPS*. The session-08 corpus bench's reported "expected (5, 4, 0)"
is the rmumps sidecar value, which the verdict file explicitly
flags as not consensus. The bench's residual-pass criterion still
counts InfNorm as a pass because the residual on the (5,1,3)
factorization is tiny (1.90e-16); inertia is a separate metric and
this matrix is in the 1567 baseline inertia mismatches.

## 2. Why MC64 makes the residual blow up

### InfNorm scaling (baseline)

```
scaling vector range (max/min) = 7.03e18
scaled |diag| range            = 1.95e45
scaled max(|off|/|diag|) col   = 1.95e45 (col 8)
```

InfNorm balances ∞-norms. With raw diagonal range 1e46, balancing
sends the small-diagonal (slack) columns to scaling factors of order
1e-21, which compresses their *scaled diagonals* down to 1e-45 —
effectively subnormal. The factorization sees them as zero pivots
(inertia zero count = 3) but the *information* in the columns is
preserved, just at a tiny absolute scale. Iterative refinement on the
unscaled solve recovers a 16-digit-clean residual.

### MC64 scaling (Policy 2 + adaptive on this matrix)

```
scaling vector range (max/min) = 2.37e42
scaled |diag| range            = 2.92e41
scaled max(|off|/|diag|) col   = 2.92e41 (col 1)
```

MC64's matching tries to make all matched diagonals 1.0 in absolute
value. With raw diagonals at 1e33, 1e4, 1e42, and 1e-4, the
matching produces scaling factors spanning 42 orders of magnitude.
The *scaled* matrix has near-1 diagonals on the columns MC64 picked
(cols 3, 5, 6–9) but produces scaled diagonals of order 1e-42 on
the *original-large* diagonals (cols 1, 2, 4). Worse, the scaled
off-diagonals on those columns are 1e41× bigger than the scaled
diagonals — completely catastrophic.

The bare-`solve_sparse` residual under MC64 is 2.35e28; with
iterative refinement (what the bench measures) it improves to
1.31e13 but is still off by 13 orders of magnitude. Refinement
cannot recover the lost information because the *scaled* matrix is
too far from anything resembling diagonal dominance.

### Why neither scaling is "right"

The matrix's condition number, computed in any consistent norm, is
≥ 10^46. No double-precision factorization can preserve all 9
inertia counts simultaneously. The four oracles disagree because
they round off the over-determined parts in different orders:

- rmumps holds 5 positive + 4 negative, zeroing nothing.
- feral + MUMPS hold 5 positive + 1 negative, zeroing 3.
- SSIDS holds 6 positive + 3 negative, zeroing nothing but
  promoting one negative to positive via threshold pivoting.

This is not a feral bug. It is the matrix.

## 3. Implications for Policy 4

The corpus-bench note proposed Policy 4 as
"try-MC64-fallback-to-InfNorm" gated on a post-scaling diagnostic.
Three candidate heuristics from the triage:

| heuristic                          | InfNorm value | MC64 value | distinguishes? |
|-----------------------------------|--------------:|-----------:|:--------------:|
| H1: scaled `min |diag| < 1e-12`   |       5.1e-46 |    3.4e-42 | NO (both fail) |
| H2: scaled `|diag| range > 1e8`   |       1.9e45  |    2.9e41  | NO (both fail) |
| H3: scaled `max(|off|/|diag|) > 1e3` | 1.9e45    |    2.9e41  | NO (both fail) |

**None of the cheap shape-only heuristics can distinguish "good"
InfNorm from "bad" MC64 on POLAK6_0021** — both scalings produce
ruinous diagonal/off-diagonal ratios because the underlying matrix
has 1e46 dynamic range. A single-shot heuristic on the scaled
matrix cannot make the call.

The viable Policy 4 designs reduce to:

### Option A — trial factorization with retry

Run MC64, factor, check the resulting inertia + a sample residual
on a random RHS. If `||r||/||b|| > 1e-6` (or some threshold) **and**
the matrix has a tractable raw dynamic range (e.g. `range(|diag|) <
1e10`), retry with InfNorm. Cost: doubled factor for the retry-set
matrices. Benefit: catches the "matched but bad" case empirically
without trying to predict it from shape.

### Option B — raw-range pre-filter

Compute `range(|raw diag|)` in O(n). If above some threshold (e.g.
1e10), skip MC64 entirely and use InfNorm — *or* skip both and use
identity. The hypothesis: matrices with raw range ≫ 1e16 are
floating-point-indeterminate regardless, and MC64 has no
information advantage to offer; InfNorm's ∞-balancing degrades
gracefully whereas MC64's matching can degrade catastrophically.

### Option C — accept that POLAK6-class matrices are uncategorizable

Treat the corpus-bench Policy 3 residual-loss of −9 as the cost of
buying the 8× tail compression on the VESUVIO/CRESC class, on the
basis that the lost matrices are oracle-disagreement matrices to
begin with. This is the **correctness-tradeoff** answer; CLAUDE.md's
"Inertia must be exactly correct" rule does not directly apply
because the lost matrices are not in the consensus-inertia set
either way.

## 4. Recommendation

Do not pursue Policy 4 in its originally-conceived form. The
prerequisite work is:

1. **Diff the residual-pass set** between Policies 1 and 3 in a
   precise way: dump the per-matrix residual under each policy and
   list every matrix where Policy 3 regresses below the bench's
   pass threshold while Policy 1 does not. The lever-C corpus-bench
   note reports aggregate counts (−9 for Policy 3) but doesn't
   enumerate which matrices.
2. **Cross-reference the regression set against `.verdict.json`** —
   how many of the −9 are "excluded" oracle-disagreement matrices?
   How many are consensus-inertia matrices that Policy 3 actually
   broke? The latter is the real Policy-4 target; the former is
   Option C territory.
3. Only after that, decide between Options A / B / C.

If the regression set is dominated by `excluded` matrices (the
hypothesis given POLAK6_0021's profile), Option C becomes the
defensible answer and Policy 3 (`Auto`) becomes the recommended
default with an updated correctness-tolerance note. If consensus
matrices are in the regression set, Option A or B is needed.

## 5. What this triage explicitly does not produce

- A code change to `pick_scaling_strategy` or `compute_scaling`.
  No production code change is justified by POLAK6_0021 alone.
- A revised threshold for `diag_only / n`. POLAK6_0021's
  ratio (0.444) is well above 0.30, but lowering the threshold to
  exclude it would also exclude the VESUVIO matrices that motivate
  lever C. Threshold tuning is the wrong knob.
- A "fix" for POLAK6_0021. It is not broken in feral; it is broken
  in IEEE-754 with this particular IPM iteration's barrier
  perturbation. The fix lives in the IPM solver (smaller
  `delta_w`, restart), not in the linear algebra.

## 6. Files this session

- `src/bin/polak6_diag.rs` (new, diagnostic only, no production code).
- `dev/research/polak6-triage-2026-04-19.md` (this file).

The next session that wants to revisit lever-C policy 4 should
start with the §4.1 residual-set diff binary. That binary does not
yet exist; the bench harness would need a flag to dump per-matrix
residuals to a file.
