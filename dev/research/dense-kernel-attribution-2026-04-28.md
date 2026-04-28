# Dense kernel — profile attribution (2026-04-28)

Date: 2026-04-28
Status: research note for the post-B-1 dense kernel work
Inputs:
- `dev/profiles/bench_solver_corpus_2026-04-28.{json.gz,syms.json,summary.txt}`
- `dev/research/dense-kernel-w2-2x2-and-blas3.md` (W-2 + B-1 plan basis)
- `dev/plans/dense-kernel-blas3.md` (Phase A landed, B-1 NR=2 landed)

## 1. Why this note exists

The 2026-04-28 re-profile of `bench_solver_corpus` shows the dense
Schur kernel is the single biggest hot kernel after symbolic
amortization (8.5% combined self). The next step in the B-1 plan
was widening the dual kernel from NR=2 to NR=4. This note
re-attributes the actual hot samples to caller and concludes that
**NR=4 widening is not the highest-ROI next step**.

## 2. Caller attribution from samply stacks

Walking each leaf-`axpy_*_unroll4_nofma` sample up to the nearest
non-pulp/non-closure `feral::dense::*` parent (n=9039 samples,
9.0 s wall, dev/profiles/bench_solver_corpus_2026-04-28.json.gz):

### `axpy_minus_unroll4_nofma` — 469 samples, 5.2% wall

| caller                                         | samples | % of kernel | % of wall |
|------------------------------------------------|--------:|------------:|----------:|
| `do_1x1_update`                                |     324 |       69.1% |      3.6% |
| `peek_ahead_replay`                            |     132 |       28.1% |      1.5% |
| `factor_frontal_blocked_in_place` (direct)     |      13 |        2.8% |      0.1% |

### `axpy2_minus_unroll4_nofma` — 174 samples, 1.9% wall

| caller                                         | samples | % of kernel | % of wall |
|------------------------------------------------|--------:|------------:|----------:|
| `do_2x2_update`                                |      94 |       54.0% |      1.0% |
| `factor_frontal_blocked_in_place` (Phase A 2×2)|      76 |       43.7% |      0.8% |
| `peek_ahead_replay`                            |       4 |        2.3% |      0.0% |

### Aggregated by purpose

| code path                                      | wall %  |
|------------------------------------------------|--------:|
| **Scalar fallback** (`do_1x1_update` + `do_2x2_update`) |   4.6%  |
| **Panel work** (`peek_ahead_replay` + blocked direct)   |   2.4%  |

## 3. Implication for the B-1 widening (NR=2 → NR=4)

The dual kernel `schur_panel_minus_nofma_strided_dual` is **not in
this hot list** at all — its self-time is captured under
`schur_panel_minus_nofma_strided_dual` (1.4% self in the profile).
That kernel's caller is `apply_blocked_schur_panel`, which is the
post-panel deferred flush.

A successful NR=4 widening would cut the dual kernel body cost by
~50% on rectangular bulk regions. Best-case wall savings:
1.4% × 0.5 ≈ **0.7% of total wall** (≈1% of solver wall).

By contrast, the scalar fallback path (`do_1x1_update` +
`do_2x2_update`) accounts for **4.6% of wall** (≈7% of solver wall).
The fallback fires whenever the panel cannot continue inline:

- 2×2 with symmetric swap (`r != col + 1`)
- swap-1×1 alternative wins (`arr >= alpha_bk * gamma_r`)
- LAPACK-extension 1×1 wins (`akk * gamma_r >= alpha * gamma0^2`)
- Duff-Reid 2×2 growth bound fails
- SSIDS det floor fails
- 1×1 rejection escalates to rook-rescue
- Panel cap or ncol boundary exhausted

On each fallback, `do_1x1_update` (or `do_2x2_update`) walks the
ENTIRE trailing block one column at a time, calling
`axpy_minus_unroll4_nofma` per trailing column. This is rank-1
BLAS-2 throughput and cannot share src loads across columns the
way `apply_blocked_schur_panel` already does (NR=2).

## 4. The right next lever

**Hypothesis (untested):** the corpus has a high enough rate of
fallback triggers that the scalar path's per-pivot rank-1 sweep
dominates the dense kernel work. KKT matrices are particularly
prone to swap-required 2×2's because the dual block's
saddle-point structure does not put the maximally-coupled rows
in `r == col + 1` order.

**Diagnostic before implementation:** instrument
`factor_frontal_blocked_in_place` to count, per supernode:

- `n_panel_full` — panels that committed all `bs` pivots inline
- `n_panel_partial` — panels that bailed with `n_elim < bs`
- `n_scalar_fallback_1x1` — scalar steps after `ScalarFallback*`
- `n_scalar_fallback_2x2` — scalar 2×2 steps
- `n_scalar_tail` — tail-loop scalar steps (`remaining < PANEL_MIN_NCOL`)
- bail reason histogram (one bucket per condition above)

Run on `qcqp1500-1c_0000`, `vesuvio_0000`, `acopr30_0000`,
`hs118_0000`, and a few CHAINWOO-class samples. Verify whether
swap-required 2×2 dominates the bail reasons.

**If swap-required 2×2 dominates:** extend Phase A to handle the
swap-2×2 case inline. The growth + det checks already exist; the
new work is performing the symmetric swap in-place inside the
panel without breaking the deferred-Schur invariant. Estimated
impact: cut `do_2x2_update` (1.0% wall) by 50%+ and let
`apply_blocked_schur_panel` cover the rank-2 update via the
existing `subdiag[k+q] != 0` path.

**If 1×1 rejection (rook-rescue) dominates:** the rook-rescue
path is currently scalar-only by design (it permutes the L
column). The lever is harder — would need a panel-aware rook
rescue or a "delay and apply later" approach. May be best left
in scalar.

**If panel boundaries dominate** (ncol close to bs): widen the
panel `bs` for square supernodes near the root.

## 5. Allocator cost as a parallel lever

Self-time category breakdown showed **15.3% wall on
allocator/memset/free**. Top offenders:
- `_xzm_free` 3.8%, `_platform_memset` 2.5%, `__bzero` 1.8%,
  `_platform_memmove` 1.7%, `xzm_malloc` 1.3%, `madvise` 1.0%

Some is benchmark harness churn (per-iterate `from_triplets`
builds new `Vec<usize>` and `Vec<f64>`; `read_mtx` reads each
matrix file). On a real IPM workload these go to zero.

The non-harness allocator share (post per-iterate alloc) is
likely 5–8% of solver wall. Candidates for elimination:
- `factor_frontal_blocked_in_place:1031-1040` allocates `perm`,
  `subdiag`, `d_panel` on each call. These could live on the
  caller's workspace (`SuperNodeWorkspace` already exists per
  `dev/plans/factor-workspace.md`).
- `factor_one_supernode` (19.1% inclusive) allocates the
  contribution block — pre-sized workspace is the lever.

## 6. Recommended sequencing

1. **Build the panel/scalar diagnostic** (Phase B-1.5,
   instrumentation only, atomic counters, ~50 LoC).
2. **Run on the representative matrix mix** and compute the
   panel-full vs scalar-fallback ratio + bail reason histogram.
3. **Decide the lever** based on what dominates:
   a. If swap-2×2 is the dominant bail reason → Phase A2:
      swap-2×2 inline.
   b. If rook-rescue dominates → defer dense kernel work; pivot
      to allocator workspace pooling.
   c. If panel cap dominates on near-root supernodes → bump
      `bs` for `ncol >= 256`.
4. **Implement the chosen lever** with research note + plan.
5. **Re-profile** to verify the gain and pick the next lever.

The B-1 NR=4 widening is **deferred** until the diagnostic shows
that `apply_blocked_schur_panel` actually dominates dense work.
On the current evidence it does not.

## 7. Out of scope for this note

- FMA-enabled axpy variant (still breaks bit-exactness, see
  `dev/tried-and-rejected.md` 2026-04-14).
- Mixed-pivot rank-bs accumulator (Phase B-2). Deferred until
  B-1 NR-widening demonstrates value.
- Cache-blocked dense-root factor (Phase C). Still blocked on a
  real BLAS-3 DSYRK kernel.

## 8. References

- `dev/sessions/2026-04-28-02.md` — re-profile checkpoint
- `dev/profiles/bench_solver_corpus_2026-04-28.summary.txt`
- `src/dense/factor.rs:1222-1474` — `lblt_panel_frontal` and bail
  reasons
- `src/dense/factor.rs:2310-2361` — `do_1x1_update`, `do_2x2_update`
- `src/dense/factor.rs:1677-1775` — `apply_blocked_schur_panel`
  (NR=2 dual)
- `src/dense/schur_kernel.rs:946-1170` —
  `schur_panel_minus_nofma_strided_dual`
