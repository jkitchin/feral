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
consensus oracle (feral + canonical Fortran MUMPS 5.8.2 + SPRAL/SSIDS);
Phase 2 has shipped scaling, ordering, pivoting, and amalgamation
work, and is ongoing.** The corpus-wide sidecar migration to
MUMPS+SSIDS consensus inertia
(`consensus_mumps_ssids_feralsparse_2026-04-25`, see
[`dev/decisions.md`](dev/decisions.md)) replaced the prior rmumps-only
ground truth and is the basis for current validation.

The sparse multifrontal path runs end-to-end on the **full
~183k-matrix KKT corpus** with no n-size filter (matrices range up to
n ≈ 5×10⁵, with ~14k above n = 500 and ~10k above n = 1000;
residuals are at machine precision on the well-conditioned majority).
Major Phase 2 capabilities now in `main`:

- **MC64 symmetric scaling** (`ScalingStrategy::Mc64Symmetric`) with
  an `Auto` strategy that picks MC64 only when its predicates fire.
- **LDLᵀ-aware ordering** (Duff–Pralet symmetric matching + quotient-
  graph compression, port of MUMPS `ICNTL(12) = 2`) with an `Auto`
  default that resolves to `LdltCompress` only on arrow-KKT-shaped
  inputs.
- **SSIDS-style delayed pivoting** in the sparse path
  (`may_delay = true` on non-root supernodes; rejected pivots are
  carried forward to the parent front).
- **Rook-rescue fallback** for pivots rejected by the column-relative
  threshold test, splicing into `try_reject_1x1_frontal` before
  delaying.
- **SSIDS-style column renumbering** (`AmalgamationStrategy::Renumber`,
  default-on) — cuts factor time 30–67% on IPM-KKT tail matrices
  (ACOPR30 / CRESC100 / LAKES / NELSON / SWOPF) at ~10% cost on the
  small-CUTEst-Hessian median.

**Reference-solver positioning** (per
[`dev/research/reference-solver-comparison.md`](dev/research/reference-solver-comparison.md),
which supersedes the earlier "10× vs MUMPS" framing): on the
archetype tail slice, FERAL matches or beats SPRAL/SSIDS on every
matrix where both ran (BATCH 0.14×, HAHN1 0.13×, HAIFAM_0082 0.47×,
ACOPR30_0067 1.11×, CRESC100 1.22×, VESUVIO 1.41×) — and SSIDS links
vendor BLAS while FERAL does not. Versus canonical MUMPS, FERAL
matches on most matrices (BATCH 0.84×, HAIFAM_0000 1.33×) and trails
by 5–8× on a tiny-IPM-KKT class where SSIDS itself trails MUMPS by
4–8×; that gap is acknowledged and deferred, with the proper
investigation queued in the reference-solver note.

The full test suite is **332 tests passing, 0 failed** (lib +
integration, 20 ignored); CI runs the same `pre-commit` hook set
used locally so local and CI cannot drift.

The Phase 2 plan lives in
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
use feral::{CscMatrix, NumericParams};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::numeric::{factorize::factorize_multifrontal, solve::solve_sparse_refined};

let csc = CscMatrix::from_triplets(n, &rows, &cols, &vals)?;
let sym = symbolic_factorize(&csc, &SupernodeParams::default())?;
let num_params = NumericParams::default();
let (sp_factors, sp_inertia) = factorize_multifrontal(&csc, &sym, &num_params)?;
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

## Known limitations

- **Rank-deficient-KKT inertia mismatches.** On the ACOPP14 / ACOPP30
  / CERI* / FBRAIN3LS panel matrices, FERAL and canonical MUMPS
  disagree on the inertia of constraint blocks that are exactly
  rank-deficient (typical signature: FERAL reports `(p, q, 0)` where
  MUMPS reports `(p, q-1, 1)` or vice versa). Phase 2.2.2's column-
  relative pivot rejection cut ACOPP30_0000's residual by 47 orders
  of magnitude (`2.27e+46 → 1.076e-1`) but does not flip the inertia
  count. Closure likely requires further work on rank-revealing
  behavior at the root supernode where `may_delay = false` forces an
  in-place `ForceAccept`. These cases are `#[ignore]`'d in
  `tests/parity.rs` with the panel-time failure mode documented in
  the test comment.
- **Tiny-IPM-KKT factor-time gap vs MUMPS.** On a class of small KKT
  matrices (BATCH, HAHN1, ACOPR30 at n ≈ 100–600), canonical MUMPS is
  5–8× faster than FERAL — a gap SPRAL/SSIDS also pays. The proper
  investigation is queued (measure MUMPS amalgamation / front-size
  distribution; bucket FERAL `factor_us` by front size; compare front
  counts) but not currently scheduled. See
  [`dev/research/reference-solver-comparison.md`](dev/research/reference-solver-comparison.md).
- **Dense kernel has no delayed pivoting.** The sparse multifrontal
  path implements SSIDS-style delayed pivoting on non-root supernodes
  (`may_delay = true`, `n_delayed_in/out` plumbed through
  `factorize_multifrontal`). The standalone dense `factor` entry
  point still falls back to `ZeroPivotAction::ForceAccept` on
  rank-deficient blocks; the sparse-path root supernode runs under
  `may_delay = false` for the same reason.

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
