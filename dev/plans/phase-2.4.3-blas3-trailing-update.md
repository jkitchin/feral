# Plan: Phase 2.4.3 — BLAS-3 trailing-update kernel (quad-column)

Date: 2026-05-12
Status: ACTIVE
Issue: #9 (re-scoped from block-32 register-resident kernel)
Inputs:
- `dev/research/blas3-trailing-update.md` (this session)
- `dev/research/feral-kernel-profile-chainwoo.md` (Snode 1933 = 62 %)
- `dev/plans/dense-kernel-blas3.md` Phase B-1 (this plan narrows B-1)
- `dev/decisions.md:464` (2026-04-14 mul+sub, no FMA)
Constraints: pure Rust, stable toolchain, no BLAS/LAPACK/Fortran.

## Why this plan supersedes block_ldlt32 for the dominant cost

Issue #9 was framed around a 32×32 register-resident kernel. The
CHAINWOO_0000 profile shows the 32×32 in-block factor is ≤ 8 % of
the front cost; the 1984×32 trailing-update is 62 % of the whole
factorization. The block_ldlt32 work (Steps 1 + 2a landed) remains
as infrastructure (the primitives + bit-parity harness) but is not
the dominant win. This plan targets the trailing-update directly.

## Goal

Add a quad-column trailing-update kernel
`schur_panel_minus_nofma_strided_quad` that processes 4 destination
columns per pulp dispatch, sharing each src-column vector load
across 4 destinations (vs 2 in the current dual kernel). Wire into
`apply_blocked_schur_panel`. Bit-exact per column with sequential
single-column dispatches.

## Steps

### Step 1 — Quad kernel + bit-parity tests (this session)

File: `src/dense/schur_kernel.rs`.

- Add `pub fn schur_panel_minus_nofma_strided_quad(dst0, dst1, dst2,
  dst3, src_block, src_first_col, n_elim, col_stride, src_row_offset,
  alphas0, alphas1, alphas2, alphas3)`.
- Length contract: `len0 = len1 + 1 = len2 + 2 = len3 + 3` (4 adjacent
  columns of a lower-triangular L panel; column j+1 is one row shorter
  than column j).
- Caps: dst0[0], dst1[0], dst2[0] computed by scalar q-loop (3 elements
  — bit-exact with single-element rank-1 step).
- Bulk: one `WithSimd` body with unroll=2 (8 acc regs: 2 chunks × 4
  dst columns). Per q, per chunk: load src vector once, then four
  `(splat alpha_j, mul, sub)` triples into the four accumulators.
- Tail: leftover full-lane SIMD vectors after the unroll-2 main body,
  then a masked tail. Both reuse the same per-element ordering.
- Tests in `mod tests`:
  - `quad_matches_single_rank1_sweep`: length sweep ∈ {0..1024 plus
    SIMD boundaries}, n_elim sweep ∈ {0, 1, 2, 3, 7, 8, 32}. For each
    config, run quad once and 4 single-column rank-1 strided dispatches
    on freshly cloned buffers; assert `f64::to_bits` equality per
    element.
  - `quad_zero_alpha_row`: alphas with random sparsity (e.g., every
    third alpha == 0). Quad path skips q if all four alphas are zero.
    Compare against single-column reference under the same alphas.
  - `quad_singleton_first_col_cap`: covers the 3-cap-element scalar
    prologue.

Acceptance: all tests pass; `cargo clippy --lib --tests -- -D warnings`
clean; `cargo fmt --check` clean. Existing 231 tests still pass.

### Step 2 — Wire into apply_blocked_schur_panel

File: `src/dense/factor.rs:1867`.

- Replace the `while j + 1 < nrow { ...dual... j += 2 }` loop with a
  quad-first walk: `while j + 3 < nrow { ...quad... j += 4 }`.
- Then fall through to dual for a 2-or-3-column remainder: `if j + 1
  < nrow { ...dual... j += 2 }`.
- Then fall through to single for the 0-or-1-column remainder: `if j
  < nrow { ...single... }`.
- alphas0/1/2/3 buffers all `[0.0; MAX_N_ELIM]` (MAX_N_ELIM = 64).
- Triple `split_at_mut` to carve out the 4 disjoint dst slices. Same
  pattern as the dual carve (current code lines 1923-1926); extended
  once more for cols 2 and 3.

Acceptance:
- All `tests/blocked_ldlt.rs` tests pass byte-identical.
- `cargo run --bin diag_chainwoo_profile --release` factor time
  reduced (target: 89 ns/nnz → ≤ 70 ns/nnz).
- No regression on dense p90 vs MUMPS (currently 1.86).

### Step 3 — Benchmark + journal

- Run `cargo run --bin bench --release` for the full corpus.
- Update `dev/journal/2026-05-12-06.org` with measured numbers.
- Update `dev/sessions/YYYY-MM-DD-NN.md` checkpoint.
- If the quad win is < 1.3× on snode 1933, escalate to a true
  MR×NR tiled DSYRK micro-kernel (separate plan, separate session).

## Out of scope

- MR×NR tiled DSYRK (gated on Step 3 measurement).
- Mixed-pivot (1×1/2×2) panel rank-bs path — already gated off when
  any 2×2 is in the stream, per `dev/plans/dense-kernel-blas3.md`
  Phase B-2.
- Cache-blocked dense root (Phase C in the original BLAS-3 plan).
- FMA enable behind feature flag (would break bit-exactness; see
  `dev/tried-and-rejected.md` 2026-04-14).

## Risks (see research note §6 for full table)

- Bit-pattern divergence on mixed alpha sparsity: tests sweep the
  case where alphas0..3 have independent zero patterns.
- Register spill on AVX2: unroll=2 chosen specifically to fit AVX2's
  16-ymm budget.
- Quad win below threshold: micro-benchmark before wiring (kernel
  exists in isolation; we measure once before paying integration
  cost).
