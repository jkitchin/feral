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

**Phase 1 is complete on the tested subset (matrices with n ≤ 500).
Feral is NOT yet correct on larger matrices.** The Phase 1 validation
ran through a benchmark harness that skipped matrices with more than
500 rows — a hold-over from the dense-only Phase 1a. When that filter
was lifted in Phase 2's first task and the sparse path was finally run
on larger matrices (up to n = 5314) already present in the corpus, it
produced residuals between **10⁴ and 10¹⁴** — 10 to 26 orders of
magnitude worse than canonical MUMPS and SSIDS on the same inputs.
The root cause is almost certainly missing global MC64 matching-based
scaling, which MUMPS and SSIDS apply by default for symmetric
indefinite matrices and feral does not. **Phase 2.2.1 (MC64 scaling)
is the current work in progress.** Until it lands, feral should not
be used on any matrix the dense path cannot handle.

### What Phase 1 did establish

On matrices with **n ≤ 500**, under a 3-oracle consensus framework
(feral vs canonical Fortran MUMPS 5.8.2 vs SPRAL/SSIDS), feral is
correct: zero Definitive failures on 153,117 consensus matrices, and
the following pairwise inertia agreement:

| Pair                | Match rate | Matches / Total   |
|---------------------|------------|-------------------|
| feral vs MUMPS      | **99.97%** | 152,094 / 152,145 |
| feral vs SSIDS      | 99.76%     | 152,779 / 153,151 |
| MUMPS vs SSIDS      | 98.25%     | 153,172 / 155,899 |

Feral agrees with canonical MUMPS *more* often than canonical MUMPS
and SSIDS agree with each other on this subset. This is real, and it
validates the pivot strategy and multifrontal structure. It does **not**
generalize beyond n ≤ 500. The bench tolerance on Phase 1 residual
checks (`n · ε · 10⁶ ≈ 10⁻⁷` on small matrices) was loose enough to
accept feral residuals that were already 6–8 orders of magnitude worse
than canonical solvers — the absolute bar was small, so the underlying
scaling bug hid in the noise. At larger n the same bug produces
residuals no reasonable tolerance can accept.

### What Phase 1 did NOT establish (and Phase 2 is fixing)

- Feral's sparse path has never produced a correct residual on a
  matrix where the dense path was not also applicable. The sparse
  multifrontal pipeline — the main deliverable of Phase 1b — is
  numerically broken at scale in a specific, understood way: no
  global scaling is applied before factorization.
- Performance against canonical solvers has never been measured
  (unknown, assumed slower).
- METIS ordering, delayed pivoting, blocked dense kernels, and SIMD
  are all Phase 2 work.

The Phase 2 plan ([`dev/plans/phase-2-planning.md`](dev/plans/phase-2-planning.md))
opens with measurement infrastructure and then proceeds directly to
MC64 global scaling as the first and most urgent correctness fix.
The scope caveat above is discussed in full in the
[Phase 1 retrospective](dev/phase1-retrospective.org) and the
decision records in [`dev/decisions.md`](dev/decisions.md).

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
