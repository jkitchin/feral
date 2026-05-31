# Lever 1.2 — cache blocking + L-panel packing for the dense Schur update

**STATUS: 1.2a IMPLEMENTED, MEASURED, REJECTED (2026-05-31). Naive row-band
blocking is bit-exact but a 0.74-0.95x REGRESSION, not a gain. Reverted, not
shipped. See "Measured result" below.**

## Measured result (2026-05-31) — supersedes the original prediction

This note originally *predicted* a ~10-30% bandwidth gain that would be "below
the noise floor." Prompted to verify rather than predict, 1.2a (row-band
blocking) was implemented on branch `claude/lever-1.2a-row-band-blocking` and
benchmarked A/B (`ROW_BAND_ENABLED` off vs on, sequential factor, dense SPD
fronts). Both prior claims were WRONG:

- **Measurable**: yes, clearly and repeatably (not noise).
- **Slower, not faster**:

  | matrix | off (ms) | on (ms) | speedup |
  |--------|---------:|--------:|--------:|
  | dense_spd_800  |  22-24 |  24-27 | 0.89x |
  | dense_spd_1200 |  68-74 |  78-91 | 0.79-0.95x |
  | dense_spd_1600 | 151-160 | 189-213 | 0.75-0.80x |
  | dense_spd_2000 | 294-302 | 386-411 | 0.74-0.76x (3 runs) |

  Monotonic with size; a real ~10-25% regression.

- **Correctness prediction held**: the gate test
  `row_band_blocking_matches_non_banded` passed byte-identically (factor +
  inertia) at every size. The bit-exactness argument was sound; the performance
  argument was wrong in sign and magnitude.

### Why it regressed

The non-banded path uses the SIMD quad kernel
(`schur_panel_minus_nofma_strided_quad`), which shares each pivot-column `src`
load across 4 destination columns (register blocking). The naive 1.2a loop
replaced that with a per-column scalar-alpha `axpy_minus_unroll4`, trading 4x
register reuse for cache reuse; the register-reuse loss dominated the bandwidth
saving.

### What a winning 1.2 would require

Band WHILE preserving the quad kernel: within each row band `[r0, r1)`, call the
strided quad/dual/single kernels on band sub-slices (`src_row_offset = r0`,
`len = r1 - r0`), keeping register blocking AND adding cache blocking. Materially
more code, fiddly at the diagonal band. Deferred as a larger evidence-backed
effort; the cheap "reuse existing kernels via a scalar axpy" plan is disproven.

### Methodological note

The original "deferred because unmeasurable" rationale was unverified
speculation. Implementing + measuring took ~30 min and gave a definite reject.
Measure the cheap version before asserting a lever's magnitude or even its sign.

---

**STATUS: DEFERRED (2026-05-31).** Analysis + plan complete; implementation not
started. Deferred for two reasons documented below under "Scope, risk, and
measurement honesty": (1) it restructures the hot, bit-exact-tested Schur kernel
(higher risk than Lever 1.1, which only wrapped it), and (2) its payoff is a
~10–30% *bandwidth* gain that is **below the run-to-run noise floor** on this
shared dev machine (Lever 1.1's identical A/B swung 1.2×–2.5× under contention),
so it cannot be measured trustworthily here. Revisit when an idle machine or a
hardware cache-miss counter is available, implementing **1.2a (row-band
blocking, reusing existing kernels) first, 1.2b (packing) only if 1.2a's
measurement justifies it**. Lever 1.1 already banked the large intra-front win;
1.2 is a refinement, not a prerequisite for the other levers.

Source: `dev/research/perf-review-2026-05-31.md` §2 Tier-1 #2. Follows Lever 1.1
(intra-front parallelism), which hit a ~3× bandwidth ceiling this note targets.

## The traffic problem (measured-by-analysis)

The trailing Schur update applies an `n_elim`-wide pivot panel (default
`block_size = 64`) to every trailing column. The quad kernel
(`schur_panel_minus_nofma_strided_quad`, `src/dense/schur_kernel.rs:1650`)
processes trailing columns in groups of 4 and, **per group**, streams all
`n_elim` pivot columns of `src_block` at stride `col_stride = nrow`.

For a front with `nrow` rows and `n_elim = 64`:

- **L-panel** = 64 columns × `nrow` × 8 B. At `nrow = 2000` ≈ **1 MB** — larger
  than a typical 512 KB–1 MB L2, so it lives in L3, not L2.
- The panel is re-read once per 4 trailing columns → `(nrow - 64)/4 ≈ 484`
  times at `nrow = 2000`. That is ≈ **480 MB** of L3-bandwidth panel reads for a
  single block step, versus ≈ 32 MB of trailing-block touch. Panel
  re-streaming dominates — this is the bandwidth wall Lever 1.1 plateaued on.

## The fix (standard GEBP-style blocking)

Block the **row dimension** so a horizontal band of the panel stays L2-resident
across all trailing columns before moving to the next band:

```
for row-band [r0, r0+RB) of the front:          # RB chosen so RB*n_elim*8 <= L2
    # panel band  = src_block[*, r0..r0+RB]      (RB*n_elim*8 bytes, L2-resident)
    for each trailing column j:                  # (or quad group)
        dst[j][r0..r0+RB] -= sum_q alpha_q[j] * panel[q][r0..r0+RB]
```

Each panel band is streamed from L3→L2 once, then reused across all trailing
columns from L2. Optionally **pack** the panel band into a contiguous
`RB × n_elim` buffer so the inner reads are unit-stride instead of `col_stride`
— a copy, removing the strided-gather penalty.

## Bit-exactness (the hard constraint)

Both transforms are bit-exact with the current kernel — and that is the whole
reason this lever is viable:

- **Row-band blocking**: reorders *which rows of which column* are computed when,
  but each output element `dst[j][i]` is still reduced over the same ascending
  `q = 0..n_elim` on one thread. No accumulation-order change → identical
  IEEE-754 result. (Same argument as Lever 1.1's column partition, applied to
  the row axis.)
- **Packing**: a `copy` of panel entries into a contiguous buffer. The kernel
  reads the same f64 values in the same per-element order; only their addresses
  change. No arithmetic, no reassociation.

So `parallel_corpus_parity` must stay at 0 mismatch by construction, and the
existing `schur_kernel` bit-exactness tests remain the reference.

## Scope, risk, and measurement honesty

This is the **highest-risk lever** in the sweep:

1. It restructures a hot, bit-exact-tested kernel (vs Lever 1.1, which wrapped
   the kernel unchanged). Every one of the quad/dual/single + fma/nofma variants
   (`schur_kernel.rs` has 6 strided kernels) would need a blocked form, or a
   shared blocking layer above them.
2. The payoff is **bandwidth**, which on this shared/contended dev machine I
   cannot measure reliably — Lever 1.1's *same* A/B swung 1.2×–2.5× run to run.
   A 10–30% bandwidth win (the realistic target) is below that noise floor here.
   Trustworthy numbers need an idle machine or a controlled `perf`/cache-miss
   counter, not wall-clock on a loaded box.
3. RB and the pack threshold are tuning parameters that need calibration on the
   target hardware — values picked here may not transfer.

### Recommended increment (smallest bit-exact step)

Do **row-band blocking first, packing second**, as two separate commits:

- **1.2a — row-band blocking**, no packing. A blocking layer in
  `apply_schur_panel_range` that loops row-bands and calls the *existing*
  kernels on `dst[r0..r0+RB]` sub-slices with the panel addressed at the band
  offset. Reuses the proven kernels verbatim (kernels already take a
  `src_row_offset` + `len`, so a band is just a different offset/len — minimal
  new code, maximal bit-exactness confidence). Calibrate `RB` for L2.
- **1.2b — panel packing**, only if 1.2a's measurement justifies the extra copy.

This keeps each step independently revertible and independently measurable, and
front-loads the low-risk half.

## A/B plan

Per the session decision, kernel-swap levers use **git before/after** (no
runtime toggle): measure `bench_intrafront` (dense fronts) and the full
`cargo run --bin bench --release` (small-front geomean must not regress) on the
commit before vs after, ideally on an idle machine. Gate: `parallel_corpus_parity`
0 mismatch; full test suite green.
