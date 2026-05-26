# Issue #54 — LP-shaped KKT inertia, oracle cross-check

Status: investigation, no fix proposed.
Matrix: `dev/repros/issue-54/nuffield2_trap_iter1.mtx`
  (n=26649, nnz=72813 in lower triangle, 8404 positive / 0 negative
   / 18245 zero diag entries).
Reporter's claim: `num_negative_eigenvalues() = 13148`,
  expected 13202, diff −54; MA57 returns 13202.

## What the oracles actually say

Probe binary: `cargo run --release --bin probe_issue54`.
Oracle runs: see commands below.

| Solver | pos    | neg    | zero | residual (‖Ax−b‖/‖b‖) |
|--------|--------|--------|------|------------------------|
| MA57   | 13399  | 12553  | 697  | 1.40e+05               |
| SSIDS  | 13648  | 12707  | 294  | 2.23e+05               |
| MUMPS  | FAIL (INFOG(1)=−9, workspace) |  |   |                        |
| Feral  | 13501  | 13148  | 0    | 1e+03 … 1e+19          |

Reproducing the oracles:

```
# RHS: deterministic pattern matching probe_issue54.rs.
python3 -c "n=26649; import sys
with open('/tmp/issue54_rhs.txt','w') as f:
    f.write(f'{n}\n')
    for i in range(n):
        f.write(f'{(((i*2654435761)%9999)/9999.0 - 0.5):.17g}\n')"

# SSIDS
echo "$(pwd)/dev/repros/issue-54/nuffield2_trap_iter1.mtx /tmp/issue54_rhs.txt /tmp/issue54_ssids.out" \
  > /tmp/m.txt
OMP_CANCELLATION=true OMP_PROC_BIND=false \
  external_benchmarks/ssids_oracle/ssids_bench /tmp/m.txt

# MA57
external_benchmarks/ma57_oracle/ma57_bench /tmp/m.txt   # uses same path
```

## What the numbers mean

The matrix is **genuinely rank-deficient**, not nearly-singular:

  * SSIDS reports `matrix_rank = 26355` (294 zero pivots) and a forward
    residual of 2.2e5 — orders of magnitude above any reasonable
    consistency tolerance even though `status = ok`.
  * MA57 reports 697 zero pivots. Larger zero count than SSIDS,
    consistent with its different (more aggressive) zero threshold;
    relative residual is also 1.4e5.
  * No two oracles agree on (pos, neg, zero). The three triples differ
    by hundreds in every component.

The issue's "expected_neg = 13202" is **pounce's structural target**
(n_c + n_d, the row count of the −I-signed blocks in the 4-block IPM
KKT). On a non-singular IPM iterate this matches the algebraic answer
by construction. On this pre-perturbation iterate the matrix is
algebraically singular, so the structural target and the algebraic
inertia diverge — and the direct solvers themselves disagree by ±500.
**MA57 does not in fact return 13202 negatives** on this matrix; it
returns 12553 + 697 zeros.

## Where feral lands

Probe results (`cargo run --release --bin probe_issue54`):

  * Every config (default, no-delayed-pivots, pivtol ∈ {1e-2, 0.0},
    scaling ∈ {Identity, InfNorm, Mc64Symmetric}, static_pivot_floor ∈
    {1e-12, 1e-8, 1e-6}) returns a different inertia, none matching any
    oracle.
  * `static_pivot_floor > 0` makes it worse, not better: pushes the
    positive count up by ~700 by perturbing every zero diagonal to a
    positive sign.
  * Default (13501, 13148, 0) — note `zero = 0`, which is the *force-
    accept-with-sign* policy: when a 1×1 pivot is strict zero, its sign
    bucket is chosen by the IEEE sign of `d` (and `+0.0` routes to
    positive). The SSIDS/MA57 numbers say a non-trivial number of
    those pivots are *genuinely* zero pivots that should be counted
    into `zero`, not split into pos/neg by floating-point noise.
  * All residuals are huge (1e3 → 1e19), so feral's factor is also
    *not* a similarity transform of A on this matrix. Sylvester's law
    does not constrain feral's reported inertia to match the algebraic
    inertia here, just as it doesn't constrain SSIDS or MA57.

## Why MA57 unblocks pounce and feral doesn't

Even though MA57's algebraic inertia (12553) differs from pounce's
structural target (13202) by *more* than feral's (13148, also differs
by 54), MA57 lets pounce converge in 1.8s while feral stalls at 600s.
The crash-out isn't really about who reports the right number on the
exact pre-perturbation iterate. Two candidate hypotheses:

  1. **Inertia stability across pounce's perturbation cascade.** Pounce
     responds to a `WrongInertia` by escalating `δ_x` (1, 3e-3, 2.6e-2,
     …, 6.99e19). With each δ, the KKT becomes less singular and the
     algebraic inertia should monotonically approach the structural
     target. The issue body reports feral's count varies *non-
     monotonically* (12615…13256) across the cascade. MA57's likely
     does converge to 13202 monotonically as δ grows (worth checking).
     Whatever feral does on near-zero pivots — force-accept + sign-of-
     `d`, plus the post-Schur cancellation that produces `±0`s — breaks
     this monotonicity property in a way MA57 doesn't.

  2. **The δ → ∞ limit.** As δ_x → 6.99e19, the matrix becomes
     diag-dominated by ±δ on each block, and the inertia is set
     structurally. If feral still does not return 13202 at the max δ,
     that is the actual feral bug — it indicates a structural pivot-
     sign error (likely in the 2×2 trace-vs-discriminant logic on
     blocks whose off-diagonals are tiny relative to the perturbed
     diagonals).

I did not collect the δ-sweep data this session.

## Pivot-bucket instrumentation (run before reverting)

With `PANEL_DIAG_ENABLED = true` I added (then removed) atomic
counters tracking the 1×1 force-accept sign and the 2×2 inertia
classification. Default-config run produced:

  * 2×2 blocks: 35 (2,0,0), 55 (0,2,0), 6251 (1,1,0), 0 zero-bucket
  * 2×2 blocks with `d11 == 0` exactly: 0
  * 2×2 blocks with `d22 == 0` exactly: 0
  * 1×1 strict-zero force-accept: 60 pos, 39 forced neg
  * 1×1 gamma0 == 0 path: 0 pos, 170 neg

The 60 forced-pos + 39 forced-neg = 99 events would be the main
candidates to redirect to `zero`. That alone would not change the gap
to 13202 in any clean way (`13148 + 39 = 13187`, still off by 15).
But it would move feral noticeably toward SSIDS (12707 neg) and MA57
(12553 neg) — both of which place these strict-zero pivots into
`zero`, not `neg`.

## Direction (not implementing this session)

A productive next step is *not* "match 13202." It is:

  * Pick a sound zero-pivot policy. The current `ForceAccept`/sign-of-
    `d` rule routes `+0.0` to `pos` and `−0.0` to `neg`, which is not
    a numerical decision — it's whatever the previous Schur update's
    rounding produced. MA57 and SSIDS instead count these into `zero`
    and report `flag = 4/7` (rank deficient). Adopting that convention
    on the strict-zero branch (only) would make feral closer to both
    oracles and would let pounce drive δ exactly as it does with MA57.
  * Re-run the δ cascade with whatever new policy and check
    monotonicity. The actionable signal is the *trend* across δ, not
    one number on one matrix.

Before changing the policy, capture two more datapoints:

  1. δ-sweep: for δ ∈ {0, 3e-3, 2.6e-2, …, 6.99e19} construct
     A + δ·diag(sign pattern) and tabulate (MA57, SSIDS, feral)
     inertia. Confirm MA57 monotonically reaches 13202 and identify
     where feral diverges.
  2. SSIDS-aligned policy: prototype `on_zero_pivot = CountAsZero` (a
     new variant that adds to `zero`, marks `needs_refinement`, and
     reports `Singular`) and re-run probe_issue54 + the corpus to
     confirm no regression on non-singular KKTs.

## Constraint cross-check (CLAUDE.md)

> Inertia must be exactly correct on non-singular matrices. On
> matrices where the canonical Fortran direct solvers (MUMPS 5.8.2
> and SPRAL SSIDS) disagree on inertia, feral must agree with at least
> one of them.

This matrix is singular by both oracles' admission. MUMPS does not
factor it. MA57 ≠ SSIDS by 154 negatives and 403 zeros. The "agree
with at least one of them" floor is not currently met (feral neg=13148
vs SSIDS 12707 vs MA57 12553), but the consensus framework would
likely tag this matrix as `excluded`. Worth checking against
`external_benchmarks/consensus/compute_consensus.py` output.

## Artifacts

  * `dev/repros/issue-54/nuffield2_trap_iter1.mtx` — the matrix.
  * `src/bin/probe_issue54.rs` — multi-config repro/probe binary.
  * `/tmp/issue54_{ssids,ma57,mumps}.out` — oracle outputs from this
    session (transient; regenerate with the commands above).
