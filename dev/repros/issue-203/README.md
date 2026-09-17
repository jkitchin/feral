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

The Rust side is `crates/feral-diagnostics/src/bin/issue203_fill_probe.rs`.

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
