# external_benchmarks

Native solver oracles for the feral consensus benchmark.

These are **not built by `cargo`** and **not in CI**. They are run
manually on a developer machine to generate per-matrix sidecar JSONs
that the feral bench can compare against. Feral itself remains pure
Rust with zero non-Rust dependencies in the core solver — this
directory is test infrastructure only.

See `dev/plans/phase-1b-consensus-exit.md` for the full plan and
`dev/decisions.md` (entry 2026-04-12) for the architectural decision
that introduced consensus benchmarking.

## Layout

```
external_benchmarks/
├── README.md             ← this file
├── mumps_oracle/         ← canonical Fortran MUMPS 5.8.2
│   ├── Makefile          ← build mumps_bench (and MUMPS libs on first run)
│   ├── Makefile.inc.mumps ← MUMPS build configuration (copied into ref/mumps/)
│   ├── mumps_bench.F     ← Fortran 77 driver, reads manifest, writes per-matrix output
│   └── run_mumps.py      ← Python wrapper: walks data/, manages JSON I/O
└── ssids_oracle/         ← canonical Fortran SSIDS (SPRAL) — TBD
```

## Per-matrix sidecar schema

Each oracle writes one JSON file per matrix, alongside the existing
ipopt sidecar (`<id>.json`):

```
data/matrices/kkt/<problem>/<id>.mtx           # input matrix
data/matrices/kkt/<problem>/<id>.json          # ipopt sidecar (rmumps inertia + RHS)
data/matrices/kkt/<problem>/<id>.mumps.json    # canonical Fortran MUMPS output
data/matrices/kkt/<problem>/<id>.ssids.json    # canonical Fortran SSIDS output (TBD)
```

Schema:

```json
{
  "solver": "mumps-5.8.2",
  "version": "5.8.2",
  "matrix": "MGH10S_0000",
  "n": 51,
  "nnz": 83,
  "factor_us": 1055,
  "solve_us": 309,
  "inertia": {"positive": 35, "negative": 16, "zero": 0},
  "rhs_source": "sidecar",
  "residual_2norm_relative": 2.18e-16,
  "factorization_status": "ok",
  "solver_info": {"infog_1": 0, "infog_28": 0}
}
```

## Building the MUMPS oracle

Prerequisites:
- gfortran (Homebrew GCC 15+ on macOS, or any GCC ≥ 7)
- gcc (Homebrew GCC 15+ on macOS — keep aligned with gfortran)
- OpenBLAS (Homebrew, default keg path or edit `Makefile.inc.mumps`)
- The MUMPS 5.8.2 source tree at `ref/mumps` (gitignored, must be
  populated separately)

```sh
cd external_benchmarks/mumps_oracle
make all
```

This copies `Makefile.inc.mumps` into `ref/mumps/Makefile.inc`, builds
the MUMPS sequential double-precision libraries, then builds the
`mumps_bench` Fortran driver.

## Running the MUMPS oracle

From the repository root:

```sh
# Sanity check on a small subdirectory:
python3 external_benchmarks/mumps_oracle/run_mumps.py \
    data/matrices/kkt/MGH10S --limit 10

# Full corpus, skipping matrices that already have a .mumps.json:
python3 external_benchmarks/mumps_oracle/run_mumps.py \
    data/matrices/kkt --skip-existing
```

The Python wrapper:
1. Walks the input tree for `*.mtx` files.
2. For each matrix, reads the existing `<id>.json` sidecar and
   extracts the RHS vector.
3. Writes a temporary `<id>.rhs.txt` (one f64 per line).
4. Builds a manifest file passed to `mumps_bench`.
5. `mumps_bench` initializes DMUMPS once, then loops over the
   manifest, factorizing each matrix with SYM=2 (symmetric
   indefinite) and writing a plain-text result to a temp file.
6. The wrapper translates each plain-text result into the canonical
   `<id>.mumps.json` schema above.

`mumps_bench` reuses one DMUMPS instance across the whole manifest to
avoid per-matrix MPI/struct overhead. ICNTL(24)=1 is set so MUMPS
detects null pivots; INFOG(12) gives the negative pivot count and
INFOG(28) gives the null pivot count.

## Why this layout

`ref/` is gitignored — the MUMPS and SPRAL source trees are large
upstream repos that don't belong in the feral history. The
`Makefile.inc.mumps` template lives in this directory (which IS
tracked) so the build configuration is reproducible without copying
ref/ contents into the repo.

The `mumps_bench` driver is a Fortran 77 program because that's the
language MUMPS exports. The Python wrapper handles JSON I/O, which
is painful in Fortran.
