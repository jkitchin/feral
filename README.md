# FERAL

![FERAL banner](assets/feral-banner.png)

**F**actored **E**rror-**R**esistant **A**lgebra **L**ibrary — a pure-Rust
sparse symmetric indefinite linear solver with certified inertia counts,
built as a clean-room replacement for MUMPS in interior-point
optimization workloads.

The name is a pun. **Fe** is iron's chemical symbol (atomic number 26,
atomic mass 55.85). Iron rusts. Rust is the language this is written in.
The creatures that survive in the wild are *feral*. The solver is both
wild and exact.

## Status

Phase 1 is complete (2026-04-12). On the 153,151-matrix KKT corpus
collected from CUTEst runs of IPOPT, feral's inertia matches **canonical
Fortran MUMPS 5.8.2** on **99.97%** of matrices, and matches SPRAL/SSIDS
on 99.76%. Under a 3-oracle consensus framework (feral vs MUMPS vs SSIDS),
there are **zero feral failures on the Definitive subset**.

Full retrospective: [`dev/phase1-retrospective.org`](dev/phase1-retrospective.org).
Phase 1 exit session: [`dev/sessions/2026-04-12-01.md`](dev/sessions/2026-04-12-01.md).

Detailed numbers:

| Pair                | Match rate | Matches / Total   |
|---------------------|------------|-------------------|
| feral vs MUMPS      | **99.97%** | 152,094 / 152,145 |
| feral vs SSIDS      | 99.76%     | 152,779 / 153,151 |
| MUMPS vs SSIDS      | 98.25%     | 153,172 / 155,899 |

Feral agrees with canonical MUMPS *more* often than canonical MUMPS and
SSIDS agree with each other on this corpus. The two Fortran solvers
disagree on boundary-pivot classifications that feral happens to match
MUMPS's reading of.

## What's in the box (Phase 1)

- **Dense Bunch-Kaufman kernel** (`src/dense/factor.rs`,
  `src/dense/solve.rs`): scalar unblocked LDL^T with the classical
  `(1+√17)/8` pivot threshold, Knight-Ruiz infinity-norm equilibration,
  and iterative refinement via a best-iterate strategy.
- **Sparse multifrontal solver** (`src/symbolic/`, `src/numeric/`):
  CHOLMOD-style analysis pipeline (AMD → postorder → column counts →
  supernode amalgamation with SSIDS nemin merge rule) feeding a
  postorder multifrontal factorization that wraps the dense BK kernel.
- **External benchmark oracles** (`external_benchmarks/`): native
  Fortran MUMPS 5.8.2 and SPRAL/SSIDS drivers that run on the same KKT
  corpus and produce per-matrix sidecar JSONs. The consensus framework
  (`external_benchmarks/consensus/compute_consensus.py`) votes across
  feral + MUMPS + SSIDS to classify each matrix as Definitive,
  Borderline, Numerically Intractable, or Excluded.

None of the external Fortran oracles are linked into the Rust crate.
`cargo build` works on a machine with nothing but a Rust toolchain.

## Architecture constraints

These are hard rules, recorded in `dev/decisions.md`:

1. Pure Rust on the stable toolchain.
2. Zero non-Rust dependencies in the core solver. No BLAS, no LAPACK, no
   Fortran. The Fortran MUMPS and SSIDS trees in `external_benchmarks/`
   are test infrastructure only, built manually and never linked from
   `cargo`.
3. MIT license.
4. Clean-room implementation from published papers and BSD-licensed
   references. Canonical references are cited in `references.bib`.
5. Inertia must be exactly correct — no tolerance on inertia counts for
   Definitive matrices in the consensus framework.
6. Correctness before performance, always.

## Building

```sh
cargo build --release
cargo test
cargo clippy -- -D warnings
```

Pre-commit hooks for `cargo fmt` and `cargo clippy` are wired up via
`.pre-commit-config.yaml`. Install once per clone:

```sh
pre-commit install
```

CI runs the exact same hooks via `pre-commit/action@v3.0.1` so local
and CI cannot drift.

## Using the solver

```rust
use feral::{factor, solve_refined, BunchKaufmanParams, SymmetricMatrix};

// Dense path
let mat = SymmetricMatrix::zeros(3);
// ... populate lower triangle with mat.set(i, j, v) ...
let (factors, inertia) = factor(&mat, &BunchKaufmanParams::default())?;
let x = solve_refined(&mat, &factors, &rhs)?;

// Sparse path
use feral::{CscMatrix, symbolic::{symbolic_factorize, SupernodeParams}};
use feral::numeric::{factorize::factorize_multifrontal, solve::solve_sparse_refined};

let csc = CscMatrix::from_triplets(n, &rows, &cols, &vals)?;
let sym = symbolic_factorize(&csc, &SupernodeParams::default())?;
let (sp_factors, sp_inertia) = factorize_multifrontal(&csc, &sym, &params)?;
let sp_x = solve_sparse_refined(&csc, &sp_factors, &rhs)?;
```

Both refined solvers use a *best-iterate* iterative refinement strategy:
on rank-deficient matrices where `ZeroPivotAction::ForceAccept` produced
a wrong `A⁻¹`, the refinement guarantees the returned `x` is no worse
than the unrefined solve, even when individual refinement steps would
have amplified the error.

## Running the KKT benchmark

```sh
# Core bench (dense + sparse, against rmumps sidecars)
cargo run --release --bin bench

# Emit per-matrix .feral.json sidecars for the consensus framework
FERAL_EMIT_SIDECARS=1 cargo run --release --bin bench
```

The bench reads `data/matrices/kkt/<problem>/<id>.mtx` + `<id>.json`
and reports inertia agreement and residual pass counts along with a
family-grouped failure analysis and dense ∩ sparse cross-comparison.
The KKT matrices are not committed — generate them via ripopt's
`collect_kkt` tool.

## Running the multi-oracle consensus

Requires `gfortran`, `OpenBLAS`, `METIS`, and the `ref/mumps` and
`ref/spral` source trees.

```sh
# Build the Fortran oracles (one-time)
make -C external_benchmarks/mumps_oracle all
make -C external_benchmarks/ssids_oracle all

# Run them over the corpus (writes .mumps.json and .ssids.json sidecars)
python3 external_benchmarks/mumps_oracle/run_mumps.py data/matrices/kkt --skip-existing
python3 external_benchmarks/ssids_oracle/run_ssids.py data/matrices/kkt --skip-existing

# Emit feral sidecars
FERAL_EMIT_SIDECARS=1 cargo run --release --bin bench

# Compute verdicts
python3 external_benchmarks/consensus/compute_consensus.py data/matrices/kkt
```

See [`dev/plans/phase-1b-consensus-exit.md`](dev/plans/phase-1b-consensus-exit.md)
for the architecture of the consensus framework and
[`dev/phase1-retrospective.org`](dev/phase1-retrospective.org) for the
story of how and why it was built.

## Known limitations (going into Phase 2)

- **ACOPP30 family residuals**: on ACOPP30_0000, feral and canonical
  MUMPS agree on the factorization inertia but feral's final residual is
  12 orders of magnitude worse (3.15e-2 vs 5.0e-14). This is a global
  equilibration gap, not a pivoting issue — MUMPS applies matching-based
  (MC64-style) scaling across the whole matrix before factorization,
  feral only applies Knight-Ruiz equilibration locally per frontal.
  Phase 2 target.
- **Delayed pivoting not implemented**: `ZeroPivotAction::ForceAccept`
  is the Phase 1 default. For matrices with genuinely rank-deficient
  blocks, this produces a wrong `A⁻¹` that iterative refinement cannot
  recover. Delayed pivoting (SSIDS-style) is Phase 2.
- **Corpus is mostly small**: the KKT corpus used for Phase 1 validation
  is dominated by CUTEst test problems with n < 500. Larger and more
  adversarial problems from AC optimal power flow, PDE-constrained
  optimization, and gas/water network design will be added in Phase 2.

## References

The bibliography in `references.bib` is cited throughout the retrospective
and code. Canonical references:

- Bunch & Kaufman 1977 (BK pivoting)
- Duff & Reid 1983 (multifrontal method)
- Amestoy, Davis & Duff 1996 (approximate minimum degree)
- George & Liu 1981 (elimination trees)
- Davis 2006 (CHOLMOD, direct methods textbook)
- Hogg, Reid & Scott 2010 (SSIDS)
- Wächter & Biegler 2006 (IPOPT interior-point method)

## License

MIT. See [`LICENSE`](LICENSE).
