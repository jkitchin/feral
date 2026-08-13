# Real discopt simplex bases for feral #161 / PR #162

PR #162 says: *"The fixture is synthetic. It reproduces the structural signature
issue #161 reports (~2 nnz/col, near-triangular, non-triangular bump), but the
QPLIB `.npz` bases live in the discopt tree, not here, so the transfer is
**unverified**."*

These are those bases. Both formats are the same matrix; `.mtx` is 1-based
Matrix Market coordinate, readable by the `read_mtx` helper already in
`examples/basis_refactor.rs`.

| file | m | nnz | nnz/col | density |
|---|---:|---:|---:|---:|
| `QPLIB_3852_basis` | 1760 | 4,003 | **2.27** | 0.129% |
| `QPLIB_1157_basis` | 3937 | 29,376 | **7.46** | 0.190% |

`QPLIB_3852` is the instance PR #162's `lp_basis()` docstring names as the source
of its ~2.3 nnz/col target, so it is the direct check on the synthetic fixture.
`QPLIB_1157` is the instance discopt issue #1008 is written around, and it is
**4x denser per column** — the fixture has no analogue of it.

Provenance: written by feral itself during a discopt root-LP solve
(`FERAL_DUMP_BASIS=<n>:<path>`), i.e. the 20th basis feral factorized while the
in-house dual simplex solved the RLT-on root relaxation. They are real simplex
bases, not synthesized.

## Transfer test result

`real_basis_hyper.rs` (included) is PR #162's own A/B against a real basis, with
the vacuity guard kept: it asserts `hyper_sparse_sweeps()` and exits non-zero if
the route never fired.

```
cargo run --release -p feral --example real_basis_hyper -- QPLIB_1157_basis.mtx 0.25 4096
```

**The route transfers — it fires and is bit-exact.** 2370–2550 of 2560 solves
took the reach route; `max |off − on| = 0.000e0` on both bases.

```
QPLIB_3852  m=1760  2.27/col  fill 1.01x
  ftran(e_i) solution nnz: p50=1 p90=26 max=1141 mean=27.2  (1.5% of m)
  dense ftran 1.74x   dense btran 1.02x   ftran_sparse 92x vs reach-on

QPLIB_1157  m=3937  7.46/col  fill 6.76x
  ftran(e_i) solution nnz: p50=1 p90=3593 max=3621 mean=580  (14.7% of m)
  dense ftran 10.60x  dense btran 0.67x   ftran_sparse 103x vs reach-on
```

### The shipped default `hyper_sparse_max_density = 0.25` costs 1.45x on btran

Sweeping the cap on `QPLIB_1157`, everything else identical:

| cap | dense ftran | dense btran | route fired |
|---:|---:|---:|---:|
| 0.02 | 10.24x | 0.98x | 1195 |
| 0.05 | 10.59x | 0.98x | 1195 |
| 0.10 | 10.81x | 0.97x | 1195 |
| **0.25 (shipped)** | 11.01x | **0.69x** | 2370 |
| 1.00 | 10.75x | **0.69x** | 2560 |

`ftran` is flat from 0.02 to 1.00 — consistent with PR #162's own sweep, which
found "the win is flat from 0.05 to 1.00". What that sweep did not surface is
that **`btran` is not flat**: it falls off a cliff between 0.10 and 0.25, and the
fired count nearly doubles (1195 → 2370) across the same step. The extra 1175
solves are `btran` reaches lying between 10% and 25% of `m` that take the route
and lose to it — PR #162's own "the reach is sorted, not merged — `O(r log r)`
where a dense sweep is `O(r)`" limitation, hitting a population the synthetic
fixture does not contain.

This is visible only on a basis with `QPLIB_1157`'s solution-density profile
(mean 14.7% of `m`, straddling the cap). On `QPLIB_3852` — the density the
fixture was tuned to — `btran` is 1.02x and the cap never bites. **The synthetic
fixture cannot see this class of regression**, which is the concrete argument for
carrying these two files in-repo.

Suggested: default `0.10`. It keeps `ftran` at 10.81x (statistically the same as
0.25's 11.01x) and moves `btran` from 0.69x to 0.97x.

## Downstream: measured through discopt

discopt was held fixed at `main` (bce881ff) and only feral varied — three
prebuilt `.so`s, arms interleaved within each rep, arm identity asserted by md5
of the loaded binary, HiGHS on the same LP as an untouched control (its time held
to sd ≤ 0.004s across every arm). Root LP of `QPLIB_1157`, RLT on, 3937 rows:

| feral arm | in-house wall | vs crates.io |
|---|---:|---:|
| 0.15.1 (crates.io) | 6.088s | 1.00x |
| main (#160 merged, `dense_bump_max_dim` **default 0**) | 5.871s | 1.04x |
| main + `dense_bump_max_dim = 4096` | 3.567s | **1.71x** |
| PR #162 + bump + cap 0.10 | 3.607s | 1.69x |

Two things for feral from this:

1. **#160's dense-bump route is off by default and is the large win here.** Its
   `LuParams` default is `dense_bump_max_dim: 0`, so a caller that takes
   `..LuParams::default()` — as discopt does — gets none of it. Worth considering
   whether 0 is the right default, or at least saying loudly in the changelog
   that the route ships inert.
2. **PR #162's solve-side gain is largely invisible end-to-end on this
   instance** (1.69x vs 1.71x, i.e. nothing) *despite* a measured 10.8x on
   `ftran` in isolation. Triangular solves are ~1% of this LP's wall. On three
   other QPLIB instances it was worth 4–13% over the same baseline, so it is not
   nothing — but issue #161's "93.1% of wall in the LU layer" does not survive
   contact with this measurement, and the 17.2x/476x framing should not be read
   as an end-to-end prediction.

## Reproducing the sweeps

The two defaults were changed in the worktree, not parameterized:

```
sed -i '' 's/hyper_sparse_max_density: 0.25,/hyper_sparse_max_density: 0.10,/' src/lu/mod.rs
sed -i '' 's/dense_bump_max_dim: 0,/dense_bump_max_dim: 4096,/'               src/lu/mod.rs
```

`real_basis_hyper.rs` needs a `[[example]]\nname = "real_basis_hyper"` stanza in
`Cargo.toml` — examples in this repo are registered explicitly.

## Caveats

- Wall-clock figures come from a machine with load 4.6–6.4 (a macOS indexer).
  Interleaving and the flat HiGHS control argue the comparisons are sound, but
  treat absolute microsecond values as indicative. The `work` counters and the
  fired counts are deterministic and are not subject to this.
- `QPLIB_1157` at RLT-on is one LP. The 6-instance discopt sweep behind point 2
  above is in discopt, not here.

---

## What is carried here, and what is not

Only the two `.mtx` files. The `.npz` originals were verified to describe the
same matrices (`max |npz − mtx| = 0.0`, dimensions and nnz as tabulated above)
and are omitted: nothing in this repository reads `.npz`, and scipy reads
Matrix Market directly (`scipy.io.mmread`).

Consumed by `tests/lu_real_bases.rs` (deterministic regression guards) and
`examples/real_basis_hyper.rs` (the A/B harness these numbers come from).
