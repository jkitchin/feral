# MUMPS small-frontal numeric kernel: how it stays fast for tiny fronts

Source: MUMPS 5.8.2 at `/Users/jkitchin/Dropbox/projects/ripopt/ref/mumps/`.
Question: with chain-like KKTs (CHAINWOO_0000, max front 32, ~thousands of
fronts), MUMPS factors at ~14 ns/nnz_L while FERAL factors at ~89 ns/nnz_L.
Gap is ~5×. None of it is BLAS3.

## 1. The pivot kernel — `DMUMPS_FAC_I_LDLT`

File: `src/dfac_front_aux.F:1147–1676`. Symmetric LDLᵀ pivot search for one
column inside a "block" (panel) of fully-summed columns. Pure Fortran
scalar code: no BLAS-2 dispatch for the pivot-search step itself.

What it does for each candidate `IPIV`:
- `:1317–1320` compute `APOS, POSPV1`, read pivot, `ABS_PIVOT`.
- `:1369–1399` scan column `IPIV` *inside* the block (above- and below-diagonal)
  to find `AMAX = max |off-diag|` and `JMAX`.
- `:1400–1418` scan the same column *outside* the block up to `LIM` rows,
  reducing into `RMAX` (this is the only loop with an `!$OMP PARALLEL DO`,
  guarded by `J1_end >= KEEP(360)` — `KEEP(360)` is the OpenMP threshold,
  defaulted around 1024 elements, so for fronts of size 32 every loop here
  runs serial).
- `:1494–1504` 1×1 acceptance: `|pivot| >= UULOC * max(RMAX,AMAX)`.
- `:1519–1606` 2×2 candidate: scan a second column for `TMAX` (`KEEP(360)`
  guarded again), compute Duff–Reid `(|a22|*RMAX + AMAX*TMAX)*UULOC <=
  |det|`. `:1599–1606`.

Critical optimization: `KEEP(206)` ("Inextpiv") — `:1273–1307`. After a
pivot succeeds, `Inextpiv = max(NPIVP1+PIVSIZ, IPIV+1)`. The next
pivot search **starts from `Inextpiv` rather than `NPIVP1`** and wraps
around. This avoids redoing the same `AMAX/RMAX` scan repeatedly on
identical-state columns when many pivots in a row are trivially
acceptable (sparse-KKT tiny fronts where most diagonals are huge
relative to off-diagonals — exactly CHAINWOO).

**Fused reduction**: `:1283–1303` — when `MAXFROMM` (the trailing maximum
captured *during the previous Schur update*) is available
(`IS_MAXFROMM_AVAIL`), MUMPS skips the AMAX/RMAX scan entirely if
`|pivot| >= UULOC * MAXFROMM_UPDATED`. The previous rank-1 update
already streamed through column `NPIV+2` and recorded the max as a
side-effect (see FAC_MQ_LDLT below). One memory pass, two outputs.

## 2. The rank-1 update — `DMUMPS_FAC_MQ_LDLT`

File: `dfac_front_aux.F:1677–1989`. This is the actual hot kernel.
For 1×1 pivots (`PIVSIZ.EQ.1`), the non-`__ve__` path is `:1813–1879`:

```fortran
DO I=1, NEL2
  K1POS = LPOS + int(I-1,8)*LDA8
  A(APOS+int(I,8)) = A(K1POS)            ! save off-diag (D·L^T row)
  A(K1POS) = A(K1POS) * VALPIV           ! scale to L column
  A(K1POS+1_8) = A(K1POS+1_8) - A(K1POS) * A(APOS+1_8)   ! fused: update col k+1
  MAXFROMM = max(MAXFROMM, abs(A(K1POS+1_8)))            ! and capture argmax
  DO JJ = 2_8, int(I,8)
    A(K1POS+JJ) = A(K1POS+JJ) - A(K1POS) * A(APOS+JJ)
  ENDDO
ENDDO
```

Three remarkable design choices:

(a) **Outer loop is over the rows `i` of L, inner loop over columns `j`**.
This is "right-looking by column scan": each row of L is touched once,
its scaled value `A(K1POS) = a_ik / d_kk` is computed once, and that
single scalar drives the rank-1 update across all column heads
`A(K1POS+JJ)`. Memory: writing column k of L (one position per i),
reading off-diags of columns `j > k` from row i (consecutive in column-major,
**unit stride**). The trailing-update inner loop `JJ=1,I` walks the
same row in i, hitting `A(K1POS+1), A(K1POS+2), ...` — strictly unit
stride, perfect for the autovectorizer.

(b) **Column k+1 is fused with the argmax of MAXFROMM** at `:1830–1831`
(and `:1860–1861` for the trailing block). The pivot search for column
k+1 is going to need `max |off-diag|` on column k+1 — which is exactly
what was just computed by this update. MUMPS captures it in the same
memory pass and hands it to FAC_I_LDLT via `IS_MAXFROMM_AVAIL = .TRUE.`
This eliminates one full O(nrow-k) scan per pivot.

(c) **The save-then-scale-then-update pattern**: `A(APOS+i) = A(K1POS)`
saves the *original* off-diagonal `a_ik` into the diagonal-block row
that becomes `D · L^T`, *then* `A(K1POS) = A(K1POS) * VALPIV` overwrites
it with `L_ik`. The trailing update `A(K1POS+JJ) - A(K1POS) * A(APOS+JJ)`
multiplies `L_ik * (D·L^T)_kj` — i.e., the kernel computes
`L D L^T` directly without reading L and the saved row from disjoint
buffers. The single column-major panel doubles as both L and D·L^T storage.

For 2×2 pivots `:1880–1979`, the structure is the same: scale once
into both `(POSPV1+2,...)` and `(POSPV2+1,...)`, then drive a
double-source axpy across the trailing rows.

## 3. The blocked update — `DMUMPS_FAC_SQ_LDLT` and BLAS-3 dispatch

File: `dfac_front_aux.F:1990–2089`. After a *block* of pivots completes
(panel size `NBKJIB`), MUMPS calls `DMUMPS_FAC_SQ_LDLT` to do the
deferred update of trailing columns. This is where DTRSM/DGEMM kick in:

- `:2036–2037` `dtrsm('L','U','T','U', NPIV_BLOCK, NRHS_TRSM, ...)` —
  block-triangular solve to scale the panel.
- `:2049–2052` `dgemmt` (when available, `KEEP(421)` controls) for the
  symmetric trailing update.
- `:2068–2070` falls back to `dgemm` over inner blocks of size `KEEP(8)`
  (default 120) when the trailing dim exceeds `KEEP(7)` (default 150).

**The crossover**: `DMUMPS_SET_INNERBLOCKSIZE` (`src/dtools.F:2398–2410`).
The "inner block" panel width `NBKJIB`:
- if `NASS < KEEP(4)` (default 24 for SYM=2): `NBKJIB = NASS` — i.e.
  **the entire fully-summed region is one panel**. No DGEMM ever runs.
- if `NASS > KEEP(3)` (default 96): `NBKJIB = min(KEEP(6), NASS) = 32`.
- otherwise: `NBKJIB = min(KEEP(5), NASS) = 16`.

`KEEP(420) = 4*KEEP(6) = 128` is the BLR block size, irrelevant here
since BLR is off by default.

For CHAINWOO with max front 32 and NASS ~ ncol_per_supernode ≪ 24,
**MUMPS never calls DGEMM**. The full factorization runs entirely in
the FAC_I + FAC_MQ scalar kernel above. The factor cost on this corpus
is dominated by:
- argmax scans (FAC_I_LDLT inner loops at `:1386–1399`, `:1414–1417`)
- rank-1 axpy (FAC_MQ_LDLT inner loop at `:1820–1822`)

## 4. Frontal matrix layout

`POSELT` indexes a contiguous `LDA × LDA = NFRONT × NFRONT` block in `S`,
column-major, with **`LDA = NFRONT`** (`dfac_front_LDLT_type1.F:138`).
**No padding.** Diagonal at `POSELT + (LDA8+1)*int(NPIV,8)`; column k
heads at `POSELT + LDA8*int(k,8) + int(NPIV,8)`. Row i of column k is at
offset `i - NPIV` from the column head.

For a 32×32 front this is 32 × 32 × 8 B = 8 KB — fits in L1 (typically 32 KB).
Once the front is in L1 the ~32-element axpy and ~32-element argmax are
cache-resident. There is no explicit prefetch — MUMPS relies on the L1
fitting the whole front and on the unit-stride loops being hardware-prefetched
trivially.

## 5. Per-front overhead

In `DMUMPS_FAC1_LDLT` (`dfac_front_LDLT_type1.F`):
- `:115–169` ~50 lines of setup that's mostly conditionals (BLR,
  parallel-pivot, OOC, Schur). For a fully-real, no-BLR, no-OOC, no-Schur
  invocation almost all of this collapses to a few integer assignments.
- `:206–229` OOC block — `IF (OOC_EFFECTIVE_ON_FRONT)` short-circuits.
- `:243–325` BLR setup — gated by `LR_ACTIVATED`; bypassed entirely.
- `:326` enters the main blocking loop.

So for the small-front fast path, prologue cost is bounded by O(1)
integer work. The IS_MAXFROMM thread of the loop avoids a full scan
on every other pivot. **There is no per-pivot allocation, no Vec
churn, no recomputation of nrow/ncol**. The frontal matrix
`A(POSELT:POSELT+NFRONT*NFRONT-1)` is a *view* into a pre-allocated
arena `S(LA)` — assembly writes into it, factorization mutates it
in place, contribution-block is a sub-view at `POSELT +
NPIV*(LDA+1)`, and the parent's assembly consumes that same memory
view with no copy.

**Delayed pivots and swap**: `DMUMPS_SWAP_LDLT` (`dfac_front_aux.F:2090–2151`)
swaps two rows/columns in-place via `A(...)` index reads. No buffer
alloc. The cost is bounded by `LIM_SWAP = NFRONT` — so for tiny fronts,
maybe ~64 element reads/writes per swap. This is amortized over all
pivots in the panel.

**Index bookkeeping**: `IW(IOLDPS+1+XSIZE)` is the running pivot count;
incremented in place. `IW(IOLDPS+...)` holds the global row indices
of the front; on swap, two integer entries are exchanged. No
permutation array allocation per front.

## 6. Threading

The pivot search loops at `:1414` and `:1569` use
`!$OMP PARALLEL DO ... IF(OMP_FLAG)` where
```fortran
OMP_FLAG = (J1_end >= KEEP(360))
```
`KEEP(360)` is set elsewhere (the analysis phase). For typical SMP
defaults it ranges 256–1024. **For front size 32, J1_end < KEEP(360)
always — the IF clause disables OpenMP**. The OMP overhead is one
runtime branch on a logical, not a thread fork.

`DMUMPS_FAC_MQ_LDLT` rank-1 trailing update has
`!$OMP PARALLEL DO ... IF (NCB1 > 300)` at `:1840` and `:1853`. Again,
for front 32 this is single-threaded.

Multifrontal sibling parallelism (across fronts) is the L0OMP
mechanism (`KEEP(405).EQ.1`), which is set up once at start of factor
phase and runs the assembly tree's leaf cluster in parallel. Per-front
overhead in the parallel path is ~1 atomic increment (`KEEP(80)`).

## 7. Specific FERAL gaps

### Gap 1: `factor_one_supernode` allocates a **fresh `nrow × nrow`** dense buffer per front.

`/Users/jkitchin/Dropbox/projects/feral/src/numeric/factorize.rs:1114–1120`:
```rust
let mut frontal_buf = std::mem::take(&mut ws.frontal_values);
frontal_buf.clear();
frontal_buf.resize(actual_nrow * actual_nrow, 0.0);
```
The `Vec::resize` zero-fills `nrow*nrow` doubles. For a 32×32 front
that's 8 KB; for thousands of fronts that's tens of MB of write
bandwidth purely on zero-fill.

MUMPS uses one fixed arena `S(LA)` allocated once at factor start
and indexed by `POSELT`. The arena was already zero (or the
assembly explicitly sets every entry it cares about). Zero-fill of
unused entries is unnecessary because the *column-major lower
triangle* is all that the factor reads, and assembly populates it.

`factor_frontal_blocked` then **allocates again**:
`/Users/jkitchin/Dropbox/projects/feral/src/dense/factor.rs:954`:
```rust
let mut a = vec![0.0; nrow * nrow];
for j in 0..nrow {
    for i in j..nrow {
        a[j * nrow + i] = matrix.data[j * nrow + i];
    }
}
```
**Two `nrow*nrow` allocations and one full copy per front**. For
CHAINWOO with ~1000 fronts of size 32, that's 16 MB of pure overhead
that MUMPS does not pay.

### Gap 2: `do_1x1_pivot` re-reads the L column, doesn't fuse with argmax of *next* column the right way.

`/Users/jkitchin/Dropbox/projects/feral/src/dense/factor.rs:2050–2087`:

```rust
let d_inv = 1.0 / d;
for i in (k + 1)..n {
    a[k * n + i] *= d_inv;       // pass 1: scale L column
}
// ...
if k + 1 < n {
    let j = k + 1;
    let l_jk = a[k * n + j];
    let l_jk_d = l_jk * d;        // re-multiply by d to recover a_jk
    a[j * n + j] -= a[k * n + j] * l_jk_d;
    for i in (j + 1)..n {
        a[j * n + i] -= a[k * n + i] * l_jk_d;   // pass 2: update col k+1
        // ... track argmax
    }
}
for j in (k + 2)..n {
    let l_jk = a[k * n + j];
    let l_jk_d = l_jk * d;        // re-multiply by d for each j
    for i in j..n {
        a[j * n + i] -= a[k * n + i] * l_jk_d;
    }
}
```

Three issues vs MUMPS:
- **Two passes over the L column**: scale-only then update. MUMPS
  does these together: read `A(K1POS)`, save it as `D·L^T` row to
  `A(APOS+I)`, scale in place, then drive the row's contribution to
  the trailing columns — one read of each row's entry, single pass.
  In FERAL the L column is read once per scale, then again per `j`
  via `a[k*n+i]` to drive the update. Two L1 traffic cycles per
  trailing element.
- **Recompute `l_jk * d`**: FERAL multiplies `L_jk` back by `d` to
  recover the original `a_jk` for the rank-1 step. MUMPS just stores
  the saved `a_jk` directly in `A(APOS+i)` and uses it as-is. One
  flop and one memory read saved per (i,j) pair.
- **Argmax fusion is partial**: only column k+1 captures the next
  pivot's `gamma0`. MUMPS captures `MAXFROMM` for column k+1 *and*
  passes it through — same effect — but the `Inextpiv` mechanism
  also lets it skip pivot search on column k+1 entirely if the
  saved max is small enough. FERAL has no equivalent skip.

### Gap 3: Row-major argmax interleaved with column-major update — bad for pulp dispatch.

The argmax tracking inside the rank-1 inner loop
(`/Users/jkitchin/Dropbox/projects/feral/src/dense/factor.rs:2070–2076`) is
a scalar reduction *inside* the SIMD-able update loop. The compiler
cannot vectorize this loop — every iteration carries a comparison and
a branch on `next_gamma0` and `next_r`. MUMPS *also* does this fusion
but only to capture `MAXFROMM` (one `max` reduction, no index), which
the autovectorizer handles well. FERAL's tracking of both `next_gamma0`
**and `next_r`** (the index) is harder to vectorize.

The pulp-dispatched `axpy_minus` kernel at
`/Users/jkitchin/Dropbox/projects/feral/src/dense/schur_kernel.rs:44–90`
is *only* called from the blocked path's `apply_blocked_schur`, not from
the per-pivot `do_1x1_pivot` shown above. For fronts ≤ `params.block_size`
(default 64), `factor_frontal_blocked` falls back to scalar
(`/Users/jkitchin/Dropbox/projects/feral/src/dense/factor.rs:949`):
```rust
if bs < 2 || ncol <= bs {
    return factor_frontal(matrix, ncol, may_delay, params);
}
```
**For CHAINWOO max-front=32 < 64, every front routes to the unblocked
scalar `factor_frontal`. The pulp SIMD kernel never runs.** This means
FERAL on small fronts is paying for: (a) two `Vec::resize`/allocate per
front, (b) scalar argmax-fused inner loops that the compiler can't
vectorize, (c) a separate L-extraction pass at `:798–821` that re-reads
the working `a` and writes a fresh L buffer.

