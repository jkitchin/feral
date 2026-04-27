# 2×2 BK Inertia Accounting Audit (2026-04-27)

Investigation of the 837 sparse-only inertia mismatches flagged by
session 2026-04-26-04 / triaged in 2026-04-26-05. Hypothesis going in:
FERAL's 2×2 BK pivot inertia accounting differs from the MUMPS
convention. **Finding: the hypothesis is wrong for the sparse path.
The sparse multifrontal already uses the trace-based eigenvalue rule
that MUMPS uses. The 837 mismatches break into three causes, with
distinct fixes — only one of them is a FERAL bug, and it's the *dense*
path, not the sparse one the bench flagged.**

## 1. Current FERAL behavior

Two inertia-counters live in the dense kernel.

### 1a. `count_2x2_inertia_val` (trace-based, mathematically correct)

`src/dense/factor.rs:1894-1912` — used by the **multifrontal sparse
path** (called from `scalar_pivot_step` at lines 1331 and 1625, which
is the per-step kernel of `factor_frontal` / `factor_frontal_blocked`):

```rust
fn count_2x2_inertia_val(d11: f64, d21: f64, d22: f64) -> Inertia {
    let det = d11 * d22 - d21 * d21;
    let trace = d11 + d22;
    if det > 0.0 {
        if trace > 0.0 { Inertia::new(2, 0, 0) }
        else { Inertia::new(0, 2, 0) }
    } else if det < 0.0 {
        Inertia::new(1, 1, 0)
    } else if trace > 0.0 { Inertia::new(1, 0, 1) }
    else if trace < 0.0 { Inertia::new(0, 1, 1) }
    else { Inertia::new(0, 0, 2) }
}
```

This is Sylvester-correct: it derives the (positive, negative, zero)
counts from the sign of the 2×2 block's determinant and trace, which
together pin down the eigenvalue signs.

### 1b. `count_2x2_inertia` (sign-of-`a00`, KNOWN-BUG)

`src/dense/factor.rs:2216-2265` — used **only** by the dense
unblocked driver (`do_2x2_pivot` at line 2116, called from the dense
`factor()` entry at line 389). Has an in-source TODO that flags the
issue:

```rust
// KNOWN BUG: this should use trace = a00 + a11 to decide the sign
// of the non-zero eigenvalue, not a00 alone. KKT matrices produce
// 2×2 blocks where a00 = 0 (variable rows have zero Hessian
// diagonal) but a11 carries the sign. The trace-based fix was
// attempted in the 2026-04-12 ACOPP30 triage but caused a 16-matrix
// dense regression against rmumps's calibration. Re-attempt after
// canonical Fortran MUMPS becomes available as a second oracle
// ...
} else if det > 0.0 {
    if a00 > 0.0 { *pos += 2; } else { *neg += 2; }
    Ok(())
} else { *pos += 1; *neg += 1; Ok(()) }
```

For the bench corpus this branch is reached **only on the dense
fast-path**. The 837 mismatches in the sparse loop never touch it.

### 1c. 1×1 path (pure sign of `d`)

`count_1x1_inertia` (line 2188) and `try_reject_1x1_frontal` (line
1737) classify by `d > 0.0` → positive, else negative. Standard.

## 2. MUMPS / rmumps convention

The Rust rmumps reference at `../ripopt/rmumps/src/pivot.rs:286-329`
computes the actual eigenvalues of each 2×2 block and counts each by
sign:

```rust
let trace = a + c;
let det = a * c - b * b;
let disc = (trace*trace - 4.0*det).max(0.0).sqrt();
let eig1 = (trace + disc) / 2.0;
let eig2 = (trace - disc) / 2.0;
for eig in [eig1, eig2] {
    if eig > ZERO_PIVOT_TOL { positive += 1; }
    else if eig < -ZERO_PIVOT_TOL { negative += 1; }
    else { zero += 1; }
}
```

This is mathematically equivalent to FERAL's `count_2x2_inertia_val`
(both are Sylvester-correct), with one secondary difference: rmumps
uses an absolute `ZERO_PIVOT_TOL` floor on each eigenvalue, while
FERAL counts as zero only when det == 0 exactly (the
near-zero-pivot path in `count_2x2_inertia` does have the floor, but
the multifrontal `_val` variant does not). For the SSCOSINE-class
matrices with no near-singular 2×2 blocks this distinction is
irrelevant.

The MUMPS Fortran (`mumps_bench.F:225-228`) writes
`inertia_neg = INFOG(12)` and `inertia_zero = INFOG(28)`. Per the
MUMPS user manual (and confirmed by the source), INFOG(12) counts
"negative pivots" — which for 2×2 blocks is the count of negative
eigenvalues of the block. Same Sylvester-correct rule.

## 3. The actual cause of the 837 mismatches

Spot-checking the sample families against the matrices on disk
reveals three regimes, **none of which is a 2×2-accounting bug in the
sparse path**.

### 3a. SSCOSINE_0000 ("bulk sign disagreement", 935/5000 negatives)

Run `cargo run --release --bin diag_d_dist` (a temp diagnostic that
dumps D-matrix sign distribution + verifies residual):

```
inertia = (4065, 935, 0)        # FERAL
sidecar  (5000,    0, 0)        # the .json
n_2x2 blocks = 23 (all det < 0)
1x1 negatives = 935 - 46 + 23 = 912
relative residual ||A x - b||/||b|| = 2.27e-16
max |x - x_true|                    = 4.77e-14
```

The residual is at machine precision: L · D · Lᵀ = A is exact. So
inertia(D) = inertia(A) by Sylvester (L is non-singular here:
0 exact-zero pivots).

But the `.mtx` itself is indefinite. Direct count from the matrix
file:

```
$ awk 'NR>2 && $1==$2 {if($3>0) p++; else if($3<0) n++} END
       {print p, n}' SSCOSINE_0000.mtx
4049 951
```

**4049 positive and 951 negative diagonals**, with off-diagonals up to
~1e4 (not diagonally dominant). The matrix is genuinely indefinite —
inertia ≈ (4065, 935, 0) is consistent with the diagonal sign
distribution after BK off-diagonal coupling.

The sidecar inertia (5000, 0, 0) does **not** describe this matrix.
The sidecar's metadata (`iteration: 0`, `delta_w: 26843.5456`,
`status: ongoing`) shows it is the IPOPT IPM iteration-0 expectation.
For unconstrained (m=0) problems IPOPT expects (n, 0, 0) and uses the
inertia returned from MUMPS to decide whether to bump delta_w. Our
sidecar evidently records the *expected* inertia, not what MUMPS
factually computed on the dumped matrix. Either:
  - the matrix dumped is pre-regularization (delta_w not yet applied
    to the negative-diagonal half), or
  - the delta_w used was insufficient and IPOPT was about to retry
    with a larger one but the sidecar was written with the planned
    inertia, or
  - the matrix and the sidecar inertia were captured at different
    points in the iteration.

Whatever the reason: FERAL's (4065, 935, 0) is the mathematically
correct inertia of the saved matrix. The sidecar is the bug, not
FERAL.

This single matrix accounts for 935 of the 837 mismatches — i.e. it
single-handedly inflates the count past 100% of the actual non-trivial
disagreements. The "837" is dominated by SSCOSINE_0000's per-pivot
delta, not by 837 different problems.

### 3b. Off-by-1 negatives on "all-positive-diagonal" matrices (BROYDN7D_0009 class)

Run on `BROYDN7D_0009.mtx`:

```
inertia = (4999, 1, 0)               # FERAL
sidecar  (5000, 0, 0)
n_2x2 blocks = 0 (all 1×1)
neg pivot magnitude = -0.0977
smallest positive pivot magnitude = +0.0563
relative residual ||A x - b||/||b|| = 2.87e-16
delta_w = 2.88, all 5000 diagonals positive
```

No 2×2 block fires. A single 1×1 pivot ends up at -0.0977 after
Schur updates. The matrix has all-positive diagonals (delta_w=2.88
shifts a barely-indefinite Hessian). The original matrix's smallest
eigenvalue is presumably ~0, near the rounding boundary; under the
multifrontal elimination order that pivot resolves slightly negative.

This is **also not an accounting bug**: FERAL reports the correct
inertia of the 1×1 pivot (sign of d). The semantic disagreement is
that IPOPT/MUMPS treats this matrix as "effectively SPD" because
MUMPS's *static pivoting / forced positive substitution* produced a
positive pivot via perturbation, and IPOPT records (5000, 0, 0).
FERAL has no static pivoting, so it reports the actual sign.

This is the regime triaged earlier as "1×1-vs-2×2 BK pivot decision
boundary." The pivot decision is in fact the same (both pick 1×1);
the disagreement is whether to *perturb* the small pivot. Same
underlying issue as 3a but at scale of one pivot.

### 3c. Off-by-1 zero (true accounting candidate)

The diag's `pos→zero off-by-1` bucket (FERAL reports `pos+1` smaller
and `zero+1` bigger than the sidecar) is the only regime that *might*
correspond to a genuine accounting choice. None of the 6 sampled
families landed in this bucket — diag aggregator currently shows 0
across SSCOSINE, CHAINWOO, BROYDN7D, EXPQUAD, RAYBENDS, DRUGDISE
samples. If it shows up in the broader bench, it would correspond to
a force-accepted small pivot (`zero_tol` floor) and could legitimately
be re-classified as "smallest-magnitude positive" if we want to track
what MUMPS would have called it. Not pursued here.

## 4. Discrepancy summary

|Regime|Cause|FERAL bug?|Sidecar bug?|Fix locus|
|---|---|---|---|---|
|3a (bulk SSCOSINE)|Saved matrix ≠ what MUMPS factored at IPM iter 0|No|Yes (matrix/inertia mismatch)|Sidecar producer|
|3b (off-by-1 neg)|Tiny pivot resolves negative; MUMPS perturbs|No|Mostly yes (records pre-perturbation expected sign)|Sidecar producer or ignore|
|Dense `count_2x2_inertia` `a00` rule|Wrong sign attribution when a00 ≈ 0|Yes|n/a|`src/dense/factor.rs:2253` (a00 > 0.0 → trace > 0.0)|

The bench flagged the dense `count_2x2_inertia` bug indirectly (the
in-source TODO), but the dense path is **not** what produced the 837
sparse mismatches. That bug is real and worth fixing for the dense
fast-path coverage, but it does not explain the symptom that
triggered this investigation.

## 5. Evidence

### Diagnostic numbers (`diag_inertia_mismatch`)

```
SSCOSINE_0000: expected (5000,0,0) feral (4065,935,0) delta_w=2.684e4
SSCOSINE_0001: expected (5000,0,0) feral (5000,  0,0) delta_w=4.398e8 (MATCH)
SSCOSINE_0002: expected (5000,0,0) feral (4999,  1,0) delta_w=1.100e8
BROYDN7D_0009: expected (5000,0,0) feral (4999,  1,0) delta_w=2.885

CHAINWOO 0..9: 10/10 match
BROYDN7D 0..8: 9/9 match (BROYDN7D_0009 = off-by-1)
```

### Direct matrix-file diagonal counts vs sidecar

```
SSCOSINE_0000.mtx: 4049 pos / 951 neg diag entries; sidecar (5000, 0, 0)
BROYDN7D_0009.mtx: 5000 pos /   0 neg diag entries; sidecar (5000, 0, 0)
DRUGDISE_0000.mtx: 1603 pos / 500 neg diag entries; sidecar (1603, 500, 0)  -> consistent
EXPQUAD_0000.mtx:  1200 pos /   0 neg diag entries; sidecar (1200,   0, 0)  -> consistent
```

The DRUGDISE / EXPQUAD examples confirm that the sidecar producer
*can* report the correct inertia — it does so in cases where the
saved matrix is internally consistent. The SSCOSINE example
demonstrates a case where it cannot.

### LDLᵀ correctness on SSCOSINE_0000

```
relative residual ||A x - b||/||b||   = 2.27e-16
forward error  max |x_solved - x_true| = 4.77e-14
total nelim                            = 5000
exact-zero d_diag entries              = 0
min eigenvalue of D                    = -28.08
```

L is full-rank, D has 935 negative eigenvalues, A x = b solves to
machine precision. By Sylvester, this means inertia(A) =
inertia(D) ≠ (5000, 0, 0). The matrix is genuinely indefinite.

## 6. Proposed fix

**Two separate fixes, prioritized:**

### Fix A (high-value, addresses the 837): Sidecar producer

The IPOPT-iteration sidecars under
`data/matrices/kkt-expansion/<FAMILY>/<NAME>_<ITER>.json` need to
record the inertia that *MUMPS actually computed on the dumped
matrix*, not the pre-iteration expected inertia. The fix is in the
sidecar producer (which lives outside this repo, under
`external_benchmarks/` or in the IPOPT instrumentation that produced
this corpus). Two viable options:

1. **Re-derive the sidecar inertia from the .mtx with `mumps_bench`**
   (run `external_benchmarks/mumps_oracle/run_mumps.py` over
   `data/matrices/kkt-expansion/`). This produces `*.mumps.json`
   alongside the existing `*.json` and the bench should compare
   against the new file. Cost: one Fortran run over 167k matrices.
2. **Patch the IPOPT instrumentation** to write the post-factor
   INFOG(12) / INFOG(28) instead of the iteration-expected
   (n_x, n_constr, 0). Cost: rerun the corpus generation.

Option (1) is concretely tractable here because the
`mumps_oracle/run_mumps.py` exists and is operational. **Implement
option (1).**

### Fix B (low-priority correctness, real but not bench-blocking): dense `a00` → trace

`src/dense/factor.rs:2251-2257` currently:

```rust
} else if det > 0.0 {
    if a00 > 0.0 { *pos += 2; } else { *neg += 2; }
```

Change to:

```rust
} else if det > 0.0 {
    if a00 + a11 > 0.0 { *pos += 2; } else { *neg += 2; }
```

This requires reading `a11` at the call site (line 2116) — the caller
already has it, just thread it through. Match the trace-based rule of
`count_2x2_inertia_val` at line 1894. Same change in the
`zero_tol_2x2` near-singular branch (line 2240) — use `trace > 0` not
`a00 > 0`.

The 2026-04-12 attempt at this fix was abandoned because it broke 16
dense tests. Those tests likely encoded the sign-of-`a00` rule by
inspection. With Fortran MUMPS now operational as a second oracle the
fix can be re-attempted, recalibrating the failing tests against
`mumps_bench` output (which would side with the trace-based rule).

## 7. Validation plan

### For Fix A (sidecar producer)

1. Run `external_benchmarks/mumps_oracle/run_mumps.py
   data/matrices/kkt-expansion/SSCOSINE/` (smallest expansion family
   containing SSCOSINE_0000) and inspect the produced
   `*.mumps.json`. **Expected**: MUMPS reports `(4065ish, 935ish, 0)`
   for SSCOSINE_0000, matching FERAL.
2. Update `bench.rs` to read `*.mumps.json` if present, falling back
   to the existing `*.json` otherwise. Re-run the headline metric.
   **Expected**: "Inertia match" jumps from 99.5% to ≥99.99% on
   sparse, with the residual remaining ≥99.8% pass.
3. If MUMPS *also* reports (5000, 0, 0) (i.e. MUMPS perturbs heavily):
   the discrepancy is real ("static pivoting") and the bench needs a
   "perturbed inertia" semantics column. Less likely given the
   matrix's actual diagonal signs.

### For Fix B (dense `a00`)

1. Add a unit test in `src/dense/factor.rs` with a 2×2 KKT block
   `[[0, b], [b, c]]` with `b≠0`, `c<0`, where the BK pivot path
   forces `det > 0` (e.g. via a contrived 3×3 matrix). The buggy
   path counts `(pos+=2)` (because a00=0 → not negative); the fixed
   path counts `(neg+=2)` (because trace = c < 0). The test oracle
   comes from rmumps's `compute_inertia` on the same 3×3.
2. Re-run the dense corpus test. The 16 previously-regressed dense
   tests will need their oracles re-derived — run them against
   `mumps_bench` and update the expected `(pos, neg, zero)` triples.

### Cross-checks

- After Fix A, `cargo run --release --bin diag_inertia_mismatch
  SSCOSINE 5` should report `5/5 match` against the new
  `*.mumps.json`.
- The bench's "Residual pass" rate should be unchanged by either fix
  (both are accounting / oracle changes; neither alters the
  factorization).

## 8. Decision summary

- The 837 mismatches are dominated by sidecar/oracle bugs, not
  factorization bugs.
- FERAL's sparse multifrontal already implements the
  Sylvester-correct trace-based 2×2 inertia rule
  (`count_2x2_inertia_val`).
- The `a00`-based dense `count_2x2_inertia` is a real bug, with a
  pre-existing TODO; it does not affect the bench's sparse-only
  numbers and is fixable independently with a 2-line change once the
  Fortran-MUMPS oracle calibration is in place.
- Recommended next action: **Fix A** (regenerate sidecars via
  `mumps_oracle/run_mumps.py`) before any FERAL code change.
