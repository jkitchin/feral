# FERAL

![FERAL banner](https://raw.githubusercontent.com/jkitchin/feral/main/assets/feral-banner.png)

[![CI](https://github.com/jkitchin/feral/actions/workflows/ci.yml/badge.svg)](https://github.com/jkitchin/feral/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20162687.svg)](https://doi.org/10.5281/zenodo.20162687)

Feral is a pure-Rust sparse symmetric indefinite linear solver with certified inertia counts for use in interior-point optimization algorithms.

The name is a pun. **Fe** is iron's chemical symbol. Iron rusts. Rust is the language this is written in. A *feral* animal is one that was domesticated but now lives in the wild. 

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

## Python bindings

The `feral-solver` package on PyPI provides Python bindings built with
[maturin](https://www.maturin.rs/) + [pyo3](https://pyo3.rs/). Wheels
are published for CPython 3.10+ on Linux x86_64/aarch64, macOS
universal2, and Windows x86_64 — no Rust toolchain required for users.

```bash
pip install feral-solver           # plain
pip install 'feral-solver[scipy]'  # scipy.sparse adapters
pip install 'feral-solver[jax]'    # JAX interop
```

Quickstart:

```python
import numpy as np
import feral

A = feral.CscMatrix.from_dense(np.array([
    [4.0, 1.0, 0.0],
    [1.0, 3.0, 2.0],
    [0.0, 2.0, 5.0],
]))

solver = feral.Solver()
status, inertia = solver.factor(A)
assert status == feral.FactorStatus.SUCCESS
x = solver.solve(np.array([1.0, 2.0, 3.0]))
```

For interior-point KKT solves, `feral.ipm.KktSolver` wraps
`feral.Solver` with the Wächter–Biegler 2006 §3.1 perturbation-
escalation loop; symbolic analysis is cached across an entire Newton
run. `feral.from_scipy(...)` / `feral.to_scipy(...)` round-trip with
`scipy.sparse` matrices. See [`python/README.md`](python/README.md)
for the full API and IPM usage, and `python/examples/` for an end-to-
end Newton step.

Build the bindings from source:

```bash
cd python
pip install maturin
maturin develop --release
pytest tests/
```

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

## Using FERAL inside Ipopt

FERAL ships with everything needed to build [Ipopt
3.14](https://github.com/coin-or/Ipopt) with `linear_solver=feral`
as a selectable option. The vendored Ipopt source lives in
`ref/Ipopt/`; the integration glue (a `LinearSolverInterface`
shim against FERAL's C ABI, plus the autotools patch) lives in
`feral-ipopt-shim/`.

### Quickstart

```sh
# One-shot: builds libferal.a, patches Ipopt, configures, builds,
# and links Ipopt against the static FERAL archive.
make ipopt

# Smoke test: run the bundled hs071 sample NLP with linear_solver=feral.
make hs071
```

Under the hood `make ipopt` delegates to `feral-ipopt-shim/Makefile`,
which:

1. `cargo build --release` to produce `target/release/libferal.a`
2. Copies `IpFeralSolverInterface.{hpp,cpp}` and `feral_capi.h` into
   `ref/Ipopt/src/Algorithm/LinearSolvers/` and applies
   `patches/ipopt-feral.patch`
3. Configures Ipopt with `--disable-shared --enable-static
   --without-hsl --without-spral --without-pardiso --without-asl`
4. `make -j` in `ref/Ipopt/build-feral/`

To rebuild after editing FERAL source: `cargo build --release && make
-C ref/Ipopt/build-feral` (the relink picks up the fresh `.a`).

### Runtime env knobs

FERAL exposes its tuning options through environment variables that
the C ABI reads on `feral_new()`:

| variable                  | default | effect |
|---------------------------|---------|--------|
| `FERAL_CASCADE_BREAK`     | off     | `on` arms the static-pivot cascade-break perturbation unconditionally |
| `FERAL_AUTO_CB_BETA`      | `0.05`  | warm cascade-break auto-arm threshold (fraction of `n`); `0` disables |
| `FERAL_SCALING`           | auto    | `auto` \| `infnorm` \| `mc64` \| `identity` |
| `FERAL_PIVTOL`            | `1e-8`  | Bunch-Kaufman partial-pivot threshold |
| `FERAL_PARALLEL`          | off     | `on` enables the rayon-based parallel multifrontal driver |
| `FERAL_FACTOR_TRACE`      | off     | `on` streams per-factor wall + delayed-pivot counts to stderr |
| `FERAL_MC64_TRACE`        | off     | `on` streams per-call MC64 wall to stderr |

The defaults are the ones validated in the v0.4.0 Mittelmann sweep
(see `CHANGELOG.md`).

### Mittelmann NLP benchmark

`external_benchmarks/mittelmann_ipopt/` runs Ipopt with both MA57 and
FERAL on the 47-problem [Mittelmann NLP
panel](https://plato.asu.edu/ftp/ampl-nlp.html). The harness, the
aggregator, and the per-problem rescue dictionary are committed; the
`.nl` problem files are **not** (~1.5 GiB total, single file up to
~290 MiB).

To reproduce the benchmark:

1. Fetch the AMPL `.nl` files from Mittelmann's public archive
   (https://plato.asu.edu/ftp/ampl-nlp.html — the problem list is in
   `external_benchmarks/mittelmann_ipopt/run.py::PROBLEMS`).
2. Edit `NL_DIR` and `PROBLEMS` at the top of `run.py` to point at
   your local checkout.
3. Build an Ipopt binary that has both MA57 (HSL) and FERAL linked
   in. The shim Makefile above produces a FERAL-only Ipopt; for the
   dual-solver comparison binary you also need a licensed HSL source
   tree and an Ipopt configure step that links `libcoinhsl`.
4. `python run.py --solvers feral,ma57 --timeout 600 && python
   aggregate.py` produces `REPORT.md` (gitignored, regenerates from
   `results/{ma57,feral}.jsonl`).

See `external_benchmarks/mittelmann_ipopt/README.md` for the per-
problem rescue table and finer-grained invocation modes.

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

- **Rank-deficient-KKT inertia outlier (FBRAIN3LS_0839).** FERAL
  reports `(5, 0, 1)` where both MUMPS and SSIDS agree on `(6, 0, 0)`
  — feral is the outlier. The other historically-failing rank-
  deficient panel matrices (ACOPP14×2, ACOPP30 × 2, CERI651CLS) all
  agree with SSIDS against MUMPS and now pass the oracle-consensus
  gate per the CLAUDE.md correctness contract. ACOPP30_0005 is a
  three-way oracle disagreement (excluded). Closure likely requires
  further work on rank-revealing behavior at the root supernode
  where `may_delay = false` forces an in-place `ForceAccept`.
- **Tiny residual gap on a few panel matrices.** CERI651CLS_0487 and
  three SSI matrices (`SSI_1685`, `SSI_2412`, `SSI_2597`) produce
  feral residuals 1.6×–1600× larger than MUMPS — all still tiny in
  absolute terms (~1e-8 to ~1e-13) but outside the K=10 residual
  gate. Inertia is correct in every case. These are `#[ignore]`'d
  in `tests/parity.rs`.
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
