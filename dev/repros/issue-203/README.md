# issue-203 — collocation KKT ordering probe

Supporting scripts for
`dev/research/issue-203-collocation-kkt-ordering-2026-09-17.md`.

Not included here: `kkt_pattern_repro.py`, the pattern generator. It is the
`kkt_pattern_repro.py` details block in [jkitchin/pounce#947][p947] and is
~11 KB of numpy with the per-time-point patterns embedded as base64. Copy it
out of the issue; `gen_perm.py` imports it by name from the same directory.

[p947]: https://github.com/jkitchin/pounce/issues/947

## Files

| file | what it does |
|---|---|
| `gen_perm.py` | builds the front-to-back **chain** permutations (`blocked` and `interleaved`) in `OrderingMethod::External` convention |
| `ma57_analyse.F` | MA57ID + MA57AD analysis only: forecast factor size, max front, elimination flops, and optionally a dump of `KEEP(1:N)` (the pivot order) |
| `value_kkt.py` | rewrites the pattern with saddle-point values so it can be *factorized*, not just analysed |

The Rust side is `crates/feral-diagnostics/src/bin/issue203_fill_probe.rs`
(symbolic) and `issue203_ab.rs` (paired wall-clock A/B across orderings).

## Wall-clock A/B

`kkt_pattern_repro.py` writes diag 4 / off-diag 1. That is fine for symbolic
analysis and useless for timing: it is not a saddle point, so none of the
pivoting a real KKT drives ever happens. `value_kkt.py` keeps the pattern and
assigns block-aware values — `H` strictly diagonally dominant positive (so
SPD), dual block `-1e-2 I` — which makes the inertia analytically
`(n_vars, m, 0)` with no rank assumption, since the Schur complement
`-dI - J H^-1 J^T` is negative definite whatever `J` is. `issue203_ab` gates
on that oracle before it prints a timing.

```sh
# n_vars = npt*NS + NC*nk from the reproducer; 111894 at nfe=48
python3 value_kkt.py mtx/gaslib40T_nseg10_nfe048.mtx valued/nfe048.mtx 111894
cargo run --release -p feral-diagnostics --bin issue203_ab \
    -- valued/nfe048.mtx 111894 7
```

Paired alternating arms, `min` over pairs, exact two-sided sign test, per
`dev/decisions.md` (2026-08-09). Report `min_factor`: pounce reuses the
symbolic across IPM iterates, so analysis is paid once.

## Running

```sh
python3 kkt_pattern_repro.py mtx 6 12 24 48 96
python3 gen_perm.py perms 6 12 24 48

cargo build --release -p feral-diagnostics --bin issue203_fill_probe
./target/release/issue203_fill_probe mtx/gaslib40T_nseg10_nfe048.mtx \
    perms/perm_blocked_nfe048.txt
```

MA57 needs the CoinHSL bundle; `HSL_ROOT` is the same path
`external_benchmarks/ma57_oracle/Makefile` expects.

```sh
HSL=$HOME/Dropbox/projects/CoinHSL.v2023.11.17.aarch64-apple-darwin-libgfortran5
gfortran -O2 -ffixed-line-length-none -fallow-argument-mismatch \
    -o ma57_analyse ma57_analyse.F \
    -L$HSL/lib -Wl,-rpath,$HSL/lib -lhsl -lopenblas -lm

# ICNTL(6): 2 = AMD, 4 = METIS, 5 = automatic (default)
./ma57_analyse mtx/gaslib40T_nseg10_nfe048.mtx 4 keep048.txt
```

To replay MA57's ordering through feral, invert `KEEP` into feral's
new-to-old convention (`perm[k]` = original index that became column `k`):

```python
import numpy as np
k = np.loadtxt("keep048.txt", dtype=np.int64)      # 1-based pivot order
inv = np.empty_like(k)
inv[k - 1] = np.arange(k.size)
np.savetxt("perms/perm_ma57metisinv_nfe048.txt", inv, fmt="%d")
```

Feeding `k - 1` directly instead of `inv` is the wrong direction and gives
`nnz_L = 1.17e10` at nfe=48 — a useful check that the convention matters.
