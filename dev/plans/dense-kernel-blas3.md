# Plan: Dense kernel — W-2 2×2, BLAS-3 DSYRK, cache-blocked dense root

Date: 2026-04-27
Status: ACTIVE — Phase A (W-2 2×2 inline) lands this session
Owner: dense BK kernel
Successor to: `dev/plans/dense-kernel-speedup.md` (W-1, W-2 1×1, W-3, W-4 — landed)
Inputs:
- `dev/research/dense-kernel-w2-2x2-and-blas3.md` (this session's research note)
- `dev/research/feral-kernel-profile-chainwoo.md` (CHAINWOO 24 ms profile)
- `dev/research/faer-dense-speed-reference.md` (faer reference)
- `dev/tried-and-rejected.md` 2026-04-14 (deferred-update no-SIMD regressed; FMA breaks bit-exactness)

## 1. Why this plan exists

Post-W-4 profile of `qcqp1500-1c_0000` (n=12,008) shows two leftover
ceilings:

- **47 supernodes with `nrow > 128` consume 94.7% of factor time.** Of
  these, 2 supernodes with `ncol > 128` (one is the 1061×1061 root)
  consume 20.8% — pure dense block work the multifrontal driver
  treats as a tall-narrow front.
- **31 supernodes with `ncol = 17–32` (43.2% of factor time) and 14
  with `ncol = 33–64` (27.7%)** run the W-2 1×1 fast path well, but
  any 2×2 pivot trigger forces a full panel restart through
  `PanelStatus::ScalarFallback`. ACOPR-style KKT matrices fire 2×2
  pivots heavily; CHAINWOO has zero so its 6.4× win does not
  generalize.

## 2. Three-phase scope

### Phase A — W-2 2×2 inline (no-swap fast path) — THIS SESSION

Extend `lblt_panel_frontal` to handle the no-swap 2×2 case without
bailing to scalar:

- **Trigger:** `akk < alpha_bk * gamma0` AND argmax row `r == col + 1`
  AND swap-1×1 fails (`arr < alpha_bk * gamma_r`) AND
  LAPACK-extension 1×1 fails (`akk * gamma_r < alpha_bk * gamma0^2`)
  AND the Duff-Reid growth bound + SSIDS det floor pass — same
  predicates as `scalar_pivot_step:1717-1828`.
- **Action:** record `d_panel[c] = d11`, `d_panel[c+1] = d22`,
  `subdiag[k+c] = d21`. Apply the 2×2 inverse scaling to L columns
  (rows `(k+c+2)..nrow` of cols `k+c, k+c+1`) — copied from
  `do_2x2_update:2075-2080`. Update inertia. Advance `c` by 2.
- **Bail-outs preserved:** swap-required 2×2 (`r != col + 1`),
  swap-required 1×1, 2×2 rejection (growth or det floor) → still
  return `ScalarFallback` and let `scalar_pivot_step` handle.
- **Peek-ahead update:** `peek_ahead_column` walks pivots `q = 0..c`
  and now must honor `subdiag[k+q]`. When non-zero, `q` is the start
  of a 2×2 pair; apply the rank-2 contribution
  `d -= dl_jq * col_q + dl_jq1 * col_{q+1}` via
  `axpy2_minus_unroll4_nofma` and skip `q+1`. Otherwise the existing
  rank-1 axpy applies.
- **Deferred Schur (`apply_blocked_schur` fallback):** mirror the
  peek-ahead — pivot-outer/column-inner loop becomes pivot-pair-or-
  singleton-outer; pairs use `axpy2_minus_unroll4_nofma` (matching
  scalar's `do_2x2_update:2094`) and singletons use
  `axpy_minus_unroll4_nofma` (matching `do_1x1_update`).
- **Rank-bs fast path (`apply_blocked_schur_panel`):** keep gated on
  "all 1×1 pivots in this panel". When any 2×2 pivot is in the
  stream, fall through to the rank-1+rank-2 reference path. Reason:
  the SIMD body in `schur_panel_minus_nofma_strided` accumulates
  contributions per-q sequentially (`acc -= a_q * s_q` per q), which
  is bit-exact with sequential rank-1 axpys but NOT with
  `axpy2_minus_unroll4_nofma` (which fuses `add` before `sub`). A
  rank-bs kernel that knows pair boundaries is Phase B-2, deferred.

**Bit-exactness contract:** the deferred path must reproduce the
exact bit pattern that scalar `factor_frontal` produces. Per-element:
- 1×1 pivot at q: `d = round(d - round(alpha_q * src_q[i]))`
- 2×2 pivot at (q, q+1): `d = round(d - round(round(dl_jq * src_q[i])
  + round(dl_jq1 * src_{q+1}[i])))` (the fused axpy2 path)

Tests assert byte-identical parity against the scalar path on the
new fixtures (Phase A acceptance §4) and on the existing
`tests/blocked_ldlt.rs` suite.

**Out of scope (deferred):**
- 2×2 with symmetric swap (still `ScalarFallback`).
- Rook-rescue 2×2 (orthogonal — only fires in scalar rejection
  fallback path).
- Rank-bs accumulator that handles 2×2 pairs natively (Phase B-2).

### Phase B-1 — True BLAS-3 DSYRK micro-kernel (next session)

Replace the per-column rank-`n_elim` SIMD axpy in
`apply_blocked_schur_panel` with a register-blocked MR×NR
micro-kernel processing a tile of trailing columns per dispatch.
Mirrors faer `bunch_kaufman/factor.rs:684`'s
`linalg::matmul::triangular::matmul` flush.

- Inputs: rank-`bs` panel `L_p` (n_elim cols), diagonal scales `D_p`
  (length n_elim), trailing block `A_t`. Computes
  `A_t -= L_p · diag(D_p) · L_p^T`.
- Bit-exactness: `(MR, NR)` chosen so each tile's accumulation order
  matches sequential rank-1 (no FMA, explicit `mul → sub` per q in a
  tile-local accumulator).
- Stays gated to all-1×1 panels; Phase B-2 lifts that.

**Pre-impl checks needed:** confirm tile (8,4) on AVX2 / Apple
M-series; verify `pulp::WithSimd` body composition.

### Phase B-2 — Rank-bs accumulator with mixed pivot stream

After Phase B-1 lands, lift the all-1×1 gate by extending the
DSYRK micro-kernel to honor `subdiag[q] != 0`. For 2×2 pairs the
inner accumulator emits the fused `add` before `sub`. Bit-exact
with the Phase A reference on mixed streams.

### Phase C — Cache-blocked dense-root factor (after B)

Detect supernodes with `nrow == ncol && ncol >= 256` (e.g.,
qcqp1500-1c's 1061×1061 root). Route through a `factor_frontal_dense`
path that runs `N_PANELS = ceil(ncol / bs)` blocked iterations,
each one panel-pivot + DSYRK flush. Conceptually
`factor_frontal_blocked_in_place` with `ncol = nrow` and the BLAS-3
DSYRK kernel from B-1 instead of the rank-bs accumulator.

## 3. File-level changes for Phase A

| file | function | change |
|---|---|---|
| `src/dense/factor.rs` | `lblt_panel_frontal` | add `ncol`, `subdiag` params; on 2×2 trigger evaluate r/swap/growth/det; on accept record d11/d22/d21 and scale L block; advance c by 2 |
| `src/dense/factor.rs` | `peek_ahead_column` | take `subdiag` slice; for q with `subdiag[k+q]!=0` apply rank-2 via `axpy2_minus_unroll4_nofma`, skip q+1 |
| `src/dense/factor.rs` | `apply_blocked_schur` | take `subdiag`; gate fast path on "no 2×2 in stream"; fallback walks pivot pair-or-singleton |
| `src/dense/factor.rs` | `factor_frontal_blocked_in_place` | thread subdiag/ncol into panel call; pass subdiag into apply_blocked_schur |
| `tests/blocked_ldlt.rs` | new | `test_2x2_inside_panel`, `test_mixed_pivots_in_panel` |

LoC estimate: ~120 production, ~80 tests.

## 4. Phase A acceptance criteria

- All existing `tests/blocked_ldlt.rs` tests pass byte-identical.
- New `test_2x2_inside_panel`: fixture with two consecutive no-swap
  2×2 blocks at panel-internal positions; byte-identical scalar vs.
  blocked.
- New `test_mixed_pivots_in_panel`: panel pattern `{1,1,2,1,2,1}`;
  byte-identical.
- `cargo run --bin diag_chainwoo_profile --release` total time
  unchanged (CHAINWOO has `n_2x2 == 0`, so impact must be zero).
- `cargo run --bin diag_qcqp_profile --release` total time
  improved (target: 5–15% on qcqp1500-1c, where 2×2 panels are
  common).
- No regression on the canonical IPM tail matrices (ACOPR30,
  CRESC100, NELSON, LAKES, SWOPF) per `cargo run --bin bench
  --release`.

## 5. Risks and mitigations

| risk | mitigation |
|---|---|
| Wrong bit pattern on mixed streams (axpy2 vs sequential rank-1) | gate rank-bs fast path off when any 2×2; fallback uses axpy2 for pairs (bit-exact with scalar `do_2x2_update`) |
| Forgotten code path: 2×2 followed by zero-d 1×1 | fallback walks pivot stream sequentially, honors d_q==0 with skip — same as W-2 1×1 path |
| Subdiag aliasing when peek-ahead crosses a panel boundary | subdiag is owned by `factor_frontal_blocked_in_place` and threaded into the panel; no aliasing because the panel writes only `subdiag[k..k+n_elim]` and reads only `subdiag[k..k+c]` |
| Perf regression on 1×1-only panels | rank-bs fast path is unchanged; the 2×2 detection cost is a single `subdiag[k+q] != 0` check per peek-ahead (negligible) |

## 6. Out of scope across all three phases

- Adopting faer as a runtime dependency (clean-room policy; see
  `dev/decisions.md`).
- Multi-threaded panels (single-threaded per supernode; assembly
  tree parallelism is orthogonal — Phase 2.5.2).
- FMA enable behind feature flag (would break inertia gate; see
  `dev/tried-and-rejected.md` 2026-04-14).
