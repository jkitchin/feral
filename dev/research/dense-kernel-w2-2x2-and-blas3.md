# Dense kernel — W-2 2×2 and BLAS-3 followups

Date: 2026-04-27
Status: research note for the post-CHAINWOO kernel work
Related:
- `dev/plans/dense-kernel-speedup.md` (W-1, W-2 1×1, W-3, W-4 — landed)
- `dev/research/feral-kernel-profile-chainwoo.md` (24 ms profile, 2026-04-27)
- `dev/research/faer-dense-speed-reference.md` (faer reference read, 2026-04-27)
- `dev/tried-and-rejected.md` 2026-04-14 entries (deferred-update without
  SIMD kernel regressed; FMA breaks bit-exactness)

## 1. Why this note exists

The post-CHAINWOO profile of `qcqp1500-1c_0000` (n=12,008) exposes a
shape the kernel-speedup plan only partially addresses:

| ncol bucket | count | %factor time |
|---:|---:|---:|
| 1   | 9,055 |  **0.0%** |
| 17–32 | 31  | 43.2% |
| 33–64 | 14  | 27.7% |
| >128 | 2   | 20.8% (one is the 1061×1061 root — 1.20 s alone) |

Cumulative: 76% of factor time is in ncol≤64 panels; 24% is in ncol>64
where the rank-bs accumulator never engages a true GEMM and the dense
root supernode is treated as if it were a tall-narrow front.

Three outstanding levers from `dense-kernel-speedup.md` directly hit
these buckets:

1. **W-2 for 2×2 pivots.** `lblt_panel_frontal:1242-1246` bails to one
   scalar step on every 2×2 trigger. Panel state and the deferred
   `apply_blocked_schur` rank-bs accumulator are 1×1-only.
2. **True BLAS-3 DSYRK** (or strict-triangular GEMM) for `ncol ≥ 64`.
   The current rank-bs SIMD axpy in `schur_panel_minus_nofma_strided`
   is closer to BLAS-2.5; per-column it issues one pulp dispatch
   covering all `n_elim` contributions, which beats per-q dispatch but
   still walks `n_elim · trailing` flops with no register reuse across
   trailing columns.
3. **Cache-blocked dense factor** for very-wide root supernodes. A
   single-shot `cholesky_in_place`-style call on an n=1061 dense
   sym-indef block would issue O(1) blocked panel passes with one
   triangular-GEMM trailing flush each — the path faer takes at
   `bunch_kaufman/factor.rs:491` (`lblt_blocked_step`).

## 2. Lessons from prior failures (`tried-and-rejected.md`)

Two constraints lock down the design:

**L1 — Loop reordering without a SIMD micro-kernel is a no-op on
throughput.** The 2026-04-14 attempt at deferred trailing updates
regressed sparse p90 11% because the deferred path paid `Vec::new`
allocation + strided access + a second pass over the contribution
block, with the *same arithmetic count* as the eager rank-1 baseline.
Faer's blocked BK speedup lives in the SIMD GEMM micro-kernel
(`Ukr<MR, NR, T>` register-blocked, masked-tail loads), not in the
panel structure. **Implication:** any new BLAS-3 path must call a
new SIMD primitive, not a restructured loop over the existing
`axpy_minus`.

**L2 — FMA changes the bit pattern and breaks the inertia gate.**
The 2026-04-14 unroll4 FMA wiring regressed sparse inertia by 4
matrices and residuals by 26. Current `*_unroll4_nofma` kernels use
explicit `mul → sub` ordering. **Implication:** any new kernel must
default to `*_nofma` semantics; an FMA variant may exist behind a
feature flag for benchmarking but cannot be the default while the
inertia gate is hard.

## 3. Scope decisions

### W-2 2×2 (this session)

Extend `lblt_panel_frontal` to handle the **no-swap 2×2 case** inline:
when scalar BK at column `k` would pick a 2×2 with rows `(k, k+1)`
already in pivot position (no symmetric swap required), record the
2×2 in `d_panel` (2 slots) and `subdiag[c]=d21`, scale the L block,
advance `c` by 2, and defer the rank-2 update. `apply_blocked_schur`
gets a mixed-pivot accumulator that walks `d_panel` honoring `subdiag`
to apply rank-1 or rank-2 contributions in pivot order.

**Out of scope for this session, deferred:**
- 2×2 with symmetric swap (still falls through to `ScalarFallback`).
- Rook-rescue 2×2 (`AcceptedRook2x2` — only fires in scalar
  rejection path; orthogonal).

**Why no-swap first:** in the qcqp profile and CHAINWOO root, all 2×2
pivots fire on consecutive columns from the saddle-point structure;
swap is rare on KKT matrices. Catches the common case with O(50 LoC)
and zero risk to the swap path.

### W-2 BLAS-3 DSYRK (next session)

Replace `apply_blocked_schur_panel`'s per-column pulp dispatch with a
register-blocked `MR×NR` micro-kernel that processes a tile of trailing
columns per dispatch. Mirrors faer's `linalg::matmul::triangular::matmul`
flush at `bunch_kaufman/factor.rs:684`. Inputs: rank-`bs` panel `L_p`,
diagonal scales `D_p`, trailing block `A_t`. Computes
`A_t -= L_p · diag(D_p) · L_p^T` in one dispatch per `MR×NR` tile.

Bit-exactness contract: `(MR,NR) = (8,8)` to (4,4) with explicit
`mul → sub` ordering matching `*_unroll4_nofma`. No FMA.

**Pre-implementation checks needed:**
- Confirm tile shape that hits the L1 cache on Apple M-series and
  AVX2/AVX-512 x86. Likely MR=8, NR=4 for AVX2.
- Verify `pulp::WithSimd` can express the doubly-nested SIMD
  accumulators we need (faer wraps the whole tile body in one
  `with_simd`).

### Cache-blocked dense-root factor (separate plan)

Detect supernodes with `nrow == ncol` AND `ncol >= 256` (root or
near-root); route through a `factor_frontal_dense` path that does
N_PANELS = ⌈ncol / bs⌉ blocked iterations, each one panel-pivot +
DSYRK-flush. This is conceptually `factor_frontal_blocked_in_place`
but with `ncol = nrow` instead of `ncol << nrow`, and with the BLAS-3
DSYRK kernel instead of the rank-bs accumulator. **Blocked on W-2
DSYRK** — this is essentially DSYRK + a thin wrapper.

## 4. Bit-exactness reference for W-2 2×2

The mixed-pivot accumulator must produce the same bit pattern as the
scalar reference applied pivot-by-pivot. For a panel with `n_elim`
pivots and a mixed pattern (e.g., `{1, 1, 2, 1}` meaning 1×1, 1×1,
2×2, 1×1), per trailing column `j`:

```text
# scalar reference (the existing eager path, do_1x1_update / do_2x2_update)
for q in 0..n_elim {
    if subdiag[q] != 0.0 {           # 2×2 starts at q
        # rank-2 contribution from cols (q, q+1)
        l_jq  = a[q*n + j]
        l_jq1 = a[(q+1)*n + j]
        dl_jq  = d11[q] * l_jq + d21[q] * l_jq1
        dl_jq1 = d21[q] * l_jq + d22[q] * l_jq1
        for i in j..n:
            a[j*n + i] -= dl_jq  * a[q*n + i]
            a[j*n + i] -= dl_jq1 * a[(q+1)*n + i]
        skip q+1
    } else {
        # rank-1 contribution from col q
        alpha = a[q*n + j] * d_panel[q]
        for i in j..n:
            a[j*n + i] -= alpha * a[q*n + i]
    }
}
```

The deferred kernel must walk pivots in identical order, applying the
identical per-element `mul + sub` sequence. This is straightforwardly
provable by induction on `q` since the order of pivots is preserved.

The 2×2 contribution decomposes into two rank-1 axpy's with alphas
`(dl_jq, dl_jq1)` against srcs `(col_q, col_{q+1})` — exactly the
shape of `axpy2_minus_unroll4_nofma`. So the rank-bs accumulator can
splice each 2×2 as two consecutive rank-1 contributions with the
correct `alphas` and `srcs`. **No new SIMD kernel needed for W-2 2×2;
only the alpha-vector construction changes.**

## 5. Acceptance criteria for W-2 2×2

- All existing `tests/blocked_ldlt.rs` tests pass byte-identical.
- New test `test_2x2_inside_panel` constructs a fixture with two
  consecutive no-swap 2×2 blocks at columns (32, 33) and (40, 41)
  inside a single panel, verifies byte-identical scalar/blocked
  parity.
- New test `test_mixed_pivots_in_panel` covers a panel of pattern
  `{1, 1, 2, 1, 2, 1}` and verifies parity.
- `diag_chainwoo_profile` and `diag_qcqp_profile` total time
  unchanged or improved (CHAINWOO has `n_2x2 == 0` so impact is on
  ACOPR-style matrices).
- ≥ 5 of the canonical IPM tail matrices (ACOPR30, CRESC100, NELSON,
  LAKES, SWOPF) show no regression.

## 6. Open questions deferred

- DSYRK MR/NR selection — needs micro-benchmarking on the target
  arch (Apple M-series here, x86-v3 in CI).
- Whether a pure-`pulp` register-blocked kernel matches BLIS-style
  `core::arch` intrinsics. The `pulp` route preserves the dep
  story; intrinsics would need cfg gates per arch.
- Cache-blocked root threshold: empirical question, likely
  `ncol >= 256` from the qcqp profile (root is 1061; next-largest
  fully-square is 524).
