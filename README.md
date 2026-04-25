# FERAL

![FERAL banner](assets/feral-banner.png)

[![CI](https://github.com/jkitchin/feral/actions/workflows/ci.yml/badge.svg)](https://github.com/jkitchin/feral/actions/workflows/ci.yml)

**F**actored **E**rror-**R**esistant **A**lgebra **L**ibrary — a pure-Rust
sparse symmetric indefinite linear solver with certified inertia counts,
built as a clean-room replacement for MUMPS in interior-point
optimization workloads.

The name is a pun. **Fe** is iron's chemical symbol (atomic number 26,
atomic mass 55.85). Iron rusts. Rust is the language this is written in.
The creatures that survive in the wild are *feral*. The solver is both
wild and exact.

## Status

Feral is research-grade, pre-1.0. **Phase 1 closed via a multi-source
consensus oracle (feral + rmumps + canonical Fortran MUMPS 5.8.2 +
SPRAL/SSIDS); Phase 2 is the current work.** The corpus-wide sidecar
migration to MUMPS+SSIDS consensus inertia
(`consensus_mumps_ssids_feralsparse_2026-04-25`, see
[`dev/decisions.md`](dev/decisions.md)) replaced the prior rmumps-only
ground truth and is the basis for the numbers below.

The sparse multifrontal path now runs end-to-end on the **full
~153k-matrix KKT corpus** with no n-size filter (matrices up to
n = 5314 are present and produce residuals at machine precision on
the well-conditioned majority). The Phase 1 "broken at scale" picture
no longer applies: MC64 matching-based scaling shipped in Phase 2.2.1,
and subsequent phases have layered on:

- **Phase 2.6.5** — LDLᵀ-aware ordering (Duff–Pralet symmetric matching
  + quotient-graph compression, port of MUMPS `ICNTL(12) = 2`). Opt-in
  via `SupernodeParams::preprocess = OrderingPreprocess::LdltCompress`.
- **Phase 2.9** — small-leaf supernode grouping (default-on).
- **Phase 2.10** — per-supernode profiler that surfaced the tiny-IPM
  amalgamation gap in the long tail.
- **Phase 2.12 (just landed)** — SSIDS-style column renumbering
  (`AmalgamationStrategy::Renumber`) is now the **default**
  amalgamation strategy. Cuts factor time 30–67% on IPM-KKT tail
  matrices (ACOPR30 / CRESC100 / LAKES / NELSON / SWOPF) at the cost
  of ~10% on the corpus median for small CUTEst-Hessian matrices.
  Net win for feral's spec-stated mission. Revert per-call with
  `AmalgamationStrategy::Adjacency`.

Sparse factor-time ratio vs canonical MUMPS 5.8.2 on the corpus
worst-10 tail is now ~10× max (down from substantially worse), with
**KIRBY2_*** and **MUONSINE_*** the dominant remaining offenders.
That tail is the active work item; the headline is "much closer, not
done." Performance work continues in subsequent phases.

The full test suite is **315 tests passing, 0 failed** across lib +
integration; CI runs the same `pre-commit` hook set used locally so
local and CI cannot drift.

The full Phase 2 plan lives in
[`dev/plans/phase-2-planning.md`](dev/plans/phase-2-planning.md);
phase-by-phase decisions are in
[`dev/decisions.md`](dev/decisions.md); the Phase 1 story is in the
[Phase 1 retrospective](dev/phase1-retrospective.org).

## What's in the box

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
- **The sparse path has never run at scale.** Every matrix in the
  Phase 1 validation corpus has n ≤ 500 because the bench enforces a
  Phase 1a hold-over filter. The sparse multifrontal pipeline, which
  is the point of Phase 1b, has literally never been run on a matrix
  where the dense path wasn't also applicable. We do not know whether
  `column_counts` has a latent O(n²) that manifests at larger sizes
  (the Phase 1b plan explicitly noted "O(n²) worst case but fine for
  dim ≤ 500"), whether the frontal allocation pattern holds up under
  deep assembly trees, or whether the per-supernode vec allocations in
  the sparse solve become dominant at large n. The first Phase 2 task
  is lifting this limit and measuring feral against canonical MUMPS
  and SSIDS on moderate-scale (n = 10³ – 10⁴) problems from
  `ripopt/benchmarks/{large_scale, grid, gas, water}`.

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
