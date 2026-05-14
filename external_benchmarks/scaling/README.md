# Scaling benchmark — feral vs MUMPS vs MA57

Synthetic-matrix scaling sweep. Generates four matrix families at
multiple sizes, runs each through the three solver bench binaries,
and reports log-log slope of factor time vs. n per (solver, family).

Complements `external_benchmarks/comparison/`, which compares solvers
on a curated real-world sample. This harness isolates the asymptotic
constant and exponent so we can answer: "as n grows, do feral and
MUMPS track the same algorithmic complexity, or does the gap widen?"

## Families

| Family       | Description                                   | Expected factor scaling |
| ------------ | --------------------------------------------- | ----------------------- |
| `dense_si`   | Dense symmetric indefinite (random + shift)   | O(n³)                   |
| `banded_spd` | Banded SPD, bandwidth 10                      | O(n · bw²)              |
| `laplace2d`  | 5-point Laplacian on k×k grid, n = k²         | O(n^1.5) (ND ordering)  |
| `saddle_kkt` | [H A^T; A 0] saddle-point KKT, n = 2·n_H      | O(n^1.5)–O(n³)          |

## Usage

```bash
# Build the three solver binaries (one-time)
cargo build --release --bin bench_one_matrix
make -C external_benchmarks/mumps_oracle
make -C external_benchmarks/ma57_oracle

# Run the full sweep (3 solvers × 4 families × ~5 sizes each)
python3 external_benchmarks/scaling/run.py

# Subsets
python3 run.py --families laplace2d,banded_spd
python3 run.py --solvers feral,mumps
python3 run.py --max-n 4096

# Re-aggregate existing sidecars and print slope report only
python3 run.py --report
```

## Output

| Path                                  | Contents                                  |
| ------------------------------------- | ----------------------------------------- |
| `matrices/<family>/<tag>.mtx`         | Generated MatrixMarket matrix (symmetric) |
| `rhs/<family>/<tag>.rhs`              | RHS = A · x_true, x_true[i] = 1 + i/n     |
| `out/<solver>/<family>__<tag>.txt`    | Per-matrix sidecar (factor_us, solve_us, rel_res, …) |
| `scaling.tsv`                         | Aggregated TSV across all (solver, job) pairs |

`scaling.tsv` columns: `solver, family, tag, n, nnz, analyse_us,
factor_us, solve_us, rel_res, status`.

The slope report at end of run estimates `factor_us ~ C · n^slope`
from a least-squares fit on log-log axes. Compare slopes across
solvers within a family; compare per-family vs. theoretical exponent
to detect implementation regressions.

## Default size sweeps

Chosen to keep total wall time on the order of a few minutes.

| Family       | Sizes                                       | n range            |
| ------------ | ------------------------------------------- | ------------------ |
| `dense_si`   | n ∈ {64, 128, 256, 512, 1024}               | 64 – 1024          |
| `banded_spd` | n ∈ {1024, 4096, 16384, 65536, 262144}      | 1024 – 262144      |
| `laplace2d`  | k ∈ {16, 32, 64, 96, 128, 192}              | 256 – 36864 (n=k²) |
| `saddle_kkt` | n_H ∈ {64, 128, 256, 512, 1024}             | 128 – 2048 (n=2·n_H) |

`--max-n N` caps the total system size after the family-specific
expansion (so `--max-n 4096` includes only laplace2d k≤64 etc.).

## Notes

- All matrices are deterministic — seeded RNGs ensure reproducibility.
- Matrices and RHS are cached to disk; re-running is fast unless you
  delete `matrices/`.
- If a solver binary is missing, it is skipped with a warning rather
  than aborting the sweep.
- The harness writes lower-triangle symmetric MatrixMarket, matching
  the convention already used by the comparison harness, so the same
  bench binaries work without modification.
