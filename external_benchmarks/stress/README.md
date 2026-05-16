# Stress suite

Targeted hard-matrix harness for feral, sourced from the SuiteSparse
`GHS_indef` group plus a synthetic generator for rank-deficient,
near-singular, ill-conditioned, and deep-null-cascade matrices.

This sibling of `comparison/` and `scaling/` exists because the curated
comparison sample skews toward KKT systems where feral already does
well; the stress suite is meant to *fail* on purpose and surface
weak spots ahead of release.

## What's in the manifest

`manifest.tsv` — one row per matrix, with `category` tagging the
pathology being probed:

| category    | what it stresses                                            |
| ----------- | ----------------------------------------------------------- |
| `saddle`    | KKT / augmented-system shape, large null contributions      |
| `pde`       | indefinite Jacobians from PDE discretizations (Bratu, Helm) |
| `opt`       | non-convex Hessian blocks from CUTEr                        |
| `mech`      | structural-mechanics indefinite stiffness                   |
| `dense`     | high fill-in, dense supernodes                              |
| `rankdef`   | exact known null space (synthetic; oracle checks `inertia.zero`) |
| `near_sing` | one pivot at scale `1e-9` or `1e-12`                        |
| `illcond`   | geometric eigenvalue spread, κ ≈ `1e10` or `1e14`           |
| `cascade`   | long chain of zero pivots, deep null cascades               |

## Quick start

```bash
# Build the bench driver (once)
cargo build --release --bin bench_one_matrix

# Generate the synthetic matrices (rankdef, ill_cond, etc.)
python3 external_benchmarks/stress/synth.py

# Download SuiteSparse matrices listed in the manifest
python3 external_benchmarks/stress/fetch.py

# Run feral over the whole stress sample
python3 external_benchmarks/stress/run.py

# Report + acceptance gate (exit 1 on any flagged matrix)
python3 external_benchmarks/stress/report.py
```

## Smoke run

```bash
# Synth-only, fastest
python3 external_benchmarks/stress/synth.py
python3 external_benchmarks/stress/run.py --category rankdef,illcond,near_sing,cascade
python3 external_benchmarks/stress/report.py
```

## Acceptance rules

`report.py` flags a matrix if any of:

1. `status != ok` (factor failure, symbolic failure, read failure)
2. `rel_res > 1e-6` (configurable via `--rel-res`)
3. For `rankdef_n_k` synthetics: `inertia.zero != k` (oracle mismatch)
4. `inertia.pos + inertia.neg + inertia.zero != n` (impossible sum)

Exit code is `0` iff no matrix is flagged — wire `report.py` into CI
to gate against regressions.

## Layout

```
external_benchmarks/stress/
├── README.md           this file
├── manifest.tsv        curated matrix list
├── fetch.py            SuiteSparse downloader (skips synth/)
├── synth.py            synthetic-matrix generator
├── run.py              feral driver (synthesizes RHS + invokes bench)
├── report.py           per-matrix + per-category report + flags
├── matrices/           downloaded + generated .mtx files (gitignored)
├── rhs/                synthesized RHS files (gitignored)
└── out/feral/          per-matrix sidecars (gitignored)
```

## Notes

- The synthetic rank-deficient matrices are dense by construction
  (Q D Q^T with Q orthogonal). They're small (n ≤ 200) and used purely
  for inertia-oracle gating, not for performance measurement.
- SuiteSparse downloads pull from
  https://suitesparse-collection-website.herokuapp.com/MM/<group>/<name>.tar.gz
  and may take several minutes for the larger entries
  (`d_pretok`, `turon_m`, `ncvxqp3`).
- The manifest sorts by `n` at runtime so smoke runs hit small
  matrices first; growing the manifest later is just appending rows.
