# FERAL kernel profile on CHAINWOO_0000 (METIS-ND)

Status: profiling note, 2026-04-27.
Driver: `src/bin/diag_chainwoo_profile.rs` (kept as a long-term diag).
Methodology: existing Phase-2.10 `Profiler` for driver-level breakdown,
plus direct calls to `factor_frontal_with_profile` on synthetic frontals
matching observed sizes. Median of 7 reps; warm-up call discarded.

Reference (CHAINWOO_0000, n=4000, nnz=7999, METIS-ND):

| solver | factor µs | nnz_L     |
|--------|-----------|-----------|
| MUMPS  |       726 |    51,964 |
| SSIDS  |     3,564 |   123,447 |
| feral  |    24,117 |   281,526 |

## 1. Time breakdown (median of 7 runs)

| bucket                                                                                      |    µs | % of total |
|---------------------------------------------------------------------------------------------|------:|-----------:|
| **Per-supernode loop** (`factor_one_supernode`, sum over 1934 fronts)                       | 23,477 |       97.4 |
| Prologue (MC64/scaling, `permute_csc_values`, `symmetric_pattern`, `is_root`, `contrib_blocks` alloc) |   244 |        1.0 |
| Driver-level unaccounted (sample sites + `Instant::now()` overhead)                         |   394 |        1.6 |
| Epilogue                                                                                    |     2 |        0.0 |
| **Total**                                                                                   | 24,117 |       100. |

Per-supernode loop further decomposed by `ncol`:

| ncol | count | sum µs | avg µs |
|-----:|------:|-------:|-------:|
|   32 |    64 | 23,485 | 366.95 |
|    1 | 1,860 |     3  |   0.00 |
|   12 |     1 |     3  |   3.00 |
|   18 |     1 |     3  |   3.00 |
| ≤ 22 |  rest |     ~12|     —  |

**99.95 % of the loop time is spent in just 64 fronts** (the 32×32 supernodes).
The remaining 1,870 supernodes — overwhelmingly leaves at `ncol=nrow=2`,
`ncol=1` — collectively cost **3 µs**. The bushy-leaf tail is not the
problem on CHAINWOO_0000; the wide-front trunk is.

### 1a. The "32×32" upper bound is misleading

Symbolic reports 64 supernodes at `nrow=32, ncol=32`. The *actual*
factored frontals (`NodeFactors::nrow`) are much larger because
`build_row_indices` (`src/numeric/factorize.rs:1779`) unions every
child's trailing-row set into the parent. The actual frontal-size
histogram from the live numeric phase:

| actual_nrow | count |
|------------:|------:|
|           2 | 1,860 |
|          67 |    10 |
|          64 |     5 |
|         100 |     3 |
|          97 |     3 |
|          60 |     3 |
|          52 |     3 |
| **largest** | **1,984** (one front, snode 1933, root) |

The slowest frontals from the per-supernode profiler:

| rank | snode | µs    | n_children |
|-----:|------:|------:|-----------:|
|    0 |  1933 | 14,994 |        38 |
|    1 |   945 |  3,699 |        37 |
|    2 |  1453 |  1,068 |        36 |
|    3 |   470 |    929 |        36 |

Snode 1933 alone is **62 % of total factor time**. It is a 1984-row,
32-column rectangular front. A direct call to
`factor_frontal_with_profile` on a synthetic 1984×32 frontal (Phase 8
in the diag binary) takes 13.0 ms — matching the 15 ms measured under
the live driver. The remaining ~2 ms is `extend_add` over 38 children.

So the relevant cost model is **not** `1934 × (32×32 work)`; it is
`64 × (n_actual × 32 dense work)` with `n_actual` varying by a factor of
~60×.

## 2. Top 3 hot loops

### Hot loop #1 — `do_1x1_update` rank-1 trailing update

`src/dense/factor.rs:1841-1860`. Called per pivot step; for the root
front (1984×32) it runs 32 times, each doing 1984 axpys.

```rust
fn do_1x1_update(a: &mut [f64], n: usize, k: usize) {
    let d = a[k * n + k];
    if d.abs() == 0.0 { return; }
    let inv_d = 1.0 / d;
    for i in (k + 1)..n {
        a[k * n + i] *= inv_d;
    }
    for j in (k + 1)..n {
        let l_jk = a[k * n + j];
        let alpha = l_jk * d;
        let (before, rest) = a.split_at_mut(j * n);
        let src = &before[k * n + j..k * n + n];
        let dst = &mut rest[j..n];
        schur_kernel::axpy_minus_unroll4_nofma(dst, src, alpha);
    }
}
```

The inner kernel is SIMD-vectorized (`pulp` axpy). The structural cost
is unavoidable per rank-1 step; the issue is that this is a **BLAS-1
formulation**: 32 separate rank-1 updates of an n-row trailing block,
each pass touching the trailing block end-to-end. MUMPS and SSIDS
formulate the panel of 32 pivots as a **single rank-32 DSYRK** which
gets ~10× more arithmetic intensity (data reuse in cache).

For CHAINWOO with the 1984×32 root, the dense kernel does ≈
`32 × 1952 × ½(1984)` ≈ 62 MFLOP at ~5–8 GFLOPS = 8–13 ms. Measured
13 ms. **This is the dominant cost.**

### Hot loop #2 — column-max pivot search inside `scalar_pivot_step`

`src/dense/factor.rs:1406-1427`. Two passes over the column for each
pivot: fully-summed rows and trailing rows.

```rust
let mut max_val = 0.0f64;
let mut max_row = k + 1;
for i in (k + 1)..ncol {
    let v = a[k * nrow + i].abs();
    if v > max_val { max_val = v; max_row = i; }
}
for i in ncol..nrow {
    let v = a[k * nrow + i].abs();
    if v > max_val { max_val = v; max_row = i; }
}
```

For the root front this scans 1983 rows per pivot × 32 pivots = ~63 k
abs-and-compare ops, plus follow-up `column_offdiag_max` and
`symmetric_row_offdiag_max` calls in the rook-rescue path. Conservative
estimate: ~10 % of the per-pivot loop. Not vectorized (data-dependent
branch).

### Hot loop #3 — `factor_frontal_blocked` alloc-and-copy + `validate()`

`src/dense/factor.rs:903` and `:954-959`. Every call:

```rust
matrix.validate()?;                          // O(n²/2) NaN/Inf scan
…
let mut a = vec![0.0; nrow * nrow];          // n×n zeroed
for j in 0..nrow {
    for i in j..nrow {
        a[j * nrow + i] = matrix.data[j * nrow + i];   // triangle copy
    }
}
```

Synthetic measurement (from Phase 8 of the diag binary):

| nrow | ncol | total ns | alloc_copy ns | pivot_loop ns | extract ns |
|-----:|-----:|---------:|--------------:|--------------:|-----------:|
|   32 |   32 |     2,875 |          167 |        1,792 |        541 |
|   67 |   32 |    11,708 |        1,375 |        7,625 |      1,375 |
|  100 |   32 |    24,333 |        2,333 |       16,500 |      3,333 |
|  256 |   32 |   196,333 |        9,125 |      162,333 |     11,916 |
| 1024 |   32 | 3,218,000 |      174,750 |    2,685,416 |    180,500 |
| 1984 |   32 | 13,013,167 |     866,750 |   10,400,500 |  1,052,000 |

`alloc_copy` is ~7 % at the worst front and ~9 % overall. Not the
top item — but eliminating it is cheap (workspace pooling) and combined
with the matching `extract` cost (same magnitude), 15–20 % of dense
kernel time goes to allocation and one-shot data movement.

`extract_ns` on the 1984 front is 1.05 ms — extracting `L`, `D`, and
the 1952×1952 contribution block, each into freshly allocated `Vec`s.
The contribution block alone is `(1984−32)² × 8B ≈ 30 MB` allocated and
copied for one supernode.

## 3. Per-front overhead (non-flop)

For a 32×32 supernode with synthetic SPD data the dense kernel runs in
~3 µs end-to-end. For a real CHAINWOO 32×32 supernode the *driver-level*
median time is 367 µs — but this is misleading because actual_nrow
varies. Subtracting the dense kernel cost predicted by Phase 8
measurements from the driver per-supernode time gives a residual that
is small (<10 %) for the root, growing larger fractionally on the
shallower fronts where more time is spent in `build_row_indices` and
`extend_add`.

`extend_add` (`src/numeric/factorize.rs:1897-1921`) uses
`SymmetricMatrix::set/get`, each of which branches on `i >= j`
(`src/dense/matrix.rs:48-65`). For a 1984-row frontal with 38 children
contributing dense Schur blocks, the extend-add cost is roughly
`∑ child_dim²` ≈ a few hundred K of branch-laden writes. Not a top-3
hotspot but a candidate quick win.

The `SymmetricMatrix::validate()` walk (`src/dense/matrix.rs:69`) at
every entry to `factor_frontal_blocked` rescans the entire lower
triangle for NaN/Inf, redundantly with the assembly that just wrote
those values from a value-checked source. Amortized this is a few
percent of dense-kernel time; on tiny leaves (1860 of them) it is the
dominant cost per-call but each call costs <2 ns so the absolute
contribution is negligible (3 µs total over the whole tail).

## 4. Mismatch with MUMPS expectations

Three independent gaps stack:

1. **Fill ratio (5.4× more nnz_L)** — orthogonal to this profile;
   investigated separately. Reduces both work and active frontal
   width.

2. **BLAS-3 vs BLAS-1 trailing update.** MUMPS panels 32 pivots and
   issues a single `DSYRK`/`DGEMM` for the rank-32 trailing update,
   reusing `L_panel` across the inner contraction. FERAL's
   `do_1x1_update` does 32 separate axpy passes, each reading and
   writing the trailing block end-to-end. At nrow=1984, ncol=32 the
   trailing block is `1952 × 1952` = 30 MB — well outside L2. Each pass
   pays the full memory bandwidth bill again. Estimated speedup from a
   blocked DSYRK-style update: 4–8× on this front-shape. The Phase-2.4.1
   `factor_frontal_blocked` panel exists but **only handles `ncol > bs`
   and falls back to scalar otherwise**; with default `bs=64` and
   CHAINWOO `ncol=32`, the panel never engages on this matrix.

3. **Symbolic frontal-width inflation.** Even with MUMPS' blocking,
   FERAL would still be slower because actual_nrow runs to 1984 vs
   MUMPS's reduced fill keeping fronts smaller. The two effects
   compound.

The largest immediately-actionable gap is (2): it is structural to the
inner kernel formulation and decoupled from the ordering issue.

## 5. Quick wins

Estimates are order-of-magnitude; all assume a clean experiment on
CHAINWOO_0000.

### QW-1. Engage the blocked panel for `ncol ≤ block_size`

`src/dense/factor.rs:949` short-circuits to scalar when
`ncol <= params.block_size` (default 64). On CHAINWOO every 32×32
supernode hits this early-out. Lower the panel-engagement threshold
(e.g. allow panels at `ncol >= 8` and let `bs = min(ncol, 64)`) so the
deferred-Schur formulation actually runs on these fronts. Even without
upgrading to DSYRK, the deferred-update formulation reduces memory
traffic on the trailing block by a factor of `bs`.

- **Expected speedup:** 2–3× on the dense pivot loop, maybe 1.8× on
  total CHAINWOO factor time.
- **Implementation cost:** ~30 LoC + parity test. The panel kernel
  exists; the gating threshold needs widening and one branch in
  `lblt_panel_frontal` for the small-`bs` case.

### QW-2. Replace the BLAS-1 trailing update with a rank-`bs` DSYRK-style accumulator

`src/dense/factor.rs:1280-1305` (`apply_blocked_schur`) currently
applies pivots `q ∈ 0..n_elim` as `n_elim` separate axpy passes over
each trailing column. Reformulate as a single `dst[i,j] -=
∑_q L[i,q] * d_q * L[j,q]` accumulated in a register tile (e.g. 4×4
or 8×4) so the trailing block is touched once per panel rather than
`bs` times. Pulp-friendly; no new deps.

- **Expected speedup:** 3–5× on the trailing update; **2–3×** on
  CHAINWOO total factor time.
- **Implementation cost:** ~150 LoC for the tiled kernel + parity tests
  (existing `tests/blocked_ldlt.rs` enforces bit-exact L,D,perm vs
  scalar — so the new kernel must be parity-safe; this constraint is
  the main implementation risk).

### QW-3. Pool the per-frontal `a` buffer and skip `validate()` on inputs from `factor_one_supernode`

`src/dense/factor.rs:903` and `:954` allocate `a = vec![0.0; nrow*nrow]`
per call. With `FactorWorkspace` already pooling `frontal_values`, an
analogous pool for the dense kernel's working `a` buffer (or just
factoring in-place into `frontal.data`) eliminates 1–2 ms of allocator
traffic per CHAINWOO factorization. The `validate()` call is redundant
when the assembled frontal came from a value-checked CSC; gate it
behind `debug_assertions`.

Additionally pool the `subdiag`, `perm`, `perm_inv`, `l`, `d_diag`,
`contrib` `Vec`s, all of which are allocated per call inside
`factor_frontal_blocked`. The `contrib` Vec for the 1984 root front is
30 MB by itself.

- **Expected speedup:** 5–10 % on CHAINWOO total time; bigger on
  IPM-style hot loops where the caller factors many similar matrices
  in sequence and the allocator can't predict the pattern.
- **Implementation cost:** ~80 LoC; mostly mechanical refactor of
  `factor_frontal_blocked`'s working-buffer ownership.

---

## Notes for follow-up

- The CHAINWOO numbers above are bounded by the fill issue; once
  ordering is fixed, the actual_nrow=1984 root will collapse and QW-2
  may matter less in absolute terms (but more in relative terms,
  because MUMPS will still beat us per nnz_L).
- The diag binary stays at `src/bin/diag_chainwoo_profile.rs` so we can
  re-run it after any kernel change and see the effect.
- `cargo bench --bench dense_factor` exists and would be the right
  place to validate QW-1/QW-2 with controlled inputs.
