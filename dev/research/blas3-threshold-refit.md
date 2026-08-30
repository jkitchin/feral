# Research — is `BLAS3_NRHS_THRESHOLD = 32` costing pounce anything?

Issue: #189 item 4. Related: #131, pounce#698.
Status: **fix landed, bit-neutral** — the threshold is unchanged; the defect
found was in the panel GEMM's tail handling.

## The question

`BLAS3_NRHS_THRESHOLD` (`src/numeric/solve.rs:17`) routes `solve_sparse_many`
through the register-blocked BLAS-3 panel kernels at `nrhs >= 32` and through
the rank-1 kernels below. The claim to test: 32 is too high, pounce runs
below it, and lowering it is free performance.

## Finding 1 — 32 has never been measured

The constant's own doc comment cites "the `k ~ 16` crossover from
`dev/research/multi-rhs.md` D3". Following that citation:

- **`multi-rhs.md` D3 is a design note, written before the kernels existed.**
  Its statement is "Real BLAS-3 trsm/gemm shapes start paying off above
  `k ~ 16`; the IPM hot path doesn't go there." That is a rule of thumb about
  BLAS shapes in general. There is no measurement in D3, and none of feral.
- **`issue-57-blas3-panel.md:121` then picks 32 relative to that assertion**:
  "D3 put the BLAS-3 crossover at `k ~ 16`; 32 is a conservative crossover
  that guarantees the microkernel's fixed setup [is amortized]".
- **Its Results section (2026-05-30) measures `nrhs` in {31, 32, 37, 64, 256}
  only**, and only on 2-D Laplacians (`n` = 484, 1024, 2025).

So the shipped constant is a safety margin applied to an unmeasured estimate,
and the entire band 2..31 — plus every KKT matrix — is unmeasured. The
citation chain reads like evidence and is not.

This is the same failure mode as the retracted `cb_core_profitable` claim in
`dev/journal/2026-08-20-01.org`: a number that everyone downstream treats as
measured because it is written down precisely.

## Finding 2 — lowering it breaks a documented bit-for-bit contract

`tests/multi_rhs.rs:227` `solve_many_refined_band_16_31_is_bit_identical_to_per_column`
and `:256` `solve_many_refined_indef_band_is_bit_identical_to_per_column`
assert, at `nrhs` = 24 and 20 respectively:

```rust
// Bit-identical: not merely "close". Tolerance is exact zero.
assert_eq!(max_diff, 0.0, "max |batched - per-column| = {max_diff:.3e}");
```

The contract exists because `BLAS3_REFINE_THRESHOLD = 16` routes *refinement*
through the batched path at `nrhs >= 16`, and bit-identity with the per-column
refiner is what makes that safe (issue #58). It holds only because
`solve_sparse_many` stays on the rank-1 kernels below 32.

Lowering `BLAS3_NRHS_THRESHOLD` below 25 fails the first test; below 21 fails
the second. Neither can be repaired by relaxing the assertion without
approval — the assertion *is* the contract, and CLAUDE.md forbids loosening a
tolerance unilaterally.

**This is not an abstract concern.** pounce calls
`solve_many_refined(m, &self.afs, self.n_s)` at
`../pounce/crates/pounce-feral/src/schur.rs:303` — the exact guarded path.
A threshold change would silently alter the W matrix that pounce's Schur
complement is built from. pounce#710 was the same class of issue.

## Finding 3 — pounce's hot path is `nrhs = 1`, not 2 and not 24

Call-site census in `../pounce`:

| path | file | nrhs |
|---|---|---|
| IPM factor + back-solve | `kkt/std_aug_system_solver.rs:497` | **1**, hardcoded |
| IPM re-solve | `kkt/std_aug_system_solver.rs:625` | **1**, hardcoded |
| sensitivity / jacrev cotangents | `pd_full_space_solver.rs:719,957` via `try_resolve_many_flat` | = parameter count |
| Schur W formation | `pounce-feral/src/schur.rs:303` | = `n_s`, caller-supplied |
| Schur block back-substitution | `pounce-feral/src/schur.rs:391` | **loops columns** (pounce#698) |

So the dominant interior-point path never reaches *any* multi-RHS kernel —
it is single-RHS, and `BLAS3_NRHS_THRESHOLD` is irrelevant to it. The
threshold can only matter for sensitivity solves with few parameters and for
the opt-in Schur path with small `n_s`.

The Schur path is opt-in ("the caller opts in by supplying the block",
`kkt/schur_aug_system_solver.rs:14`) and `n_s` is problem data, not a
constant, so no single re-fit is right for all callers.

## What would have to be true for a re-fit to be worth shipping

1. BLAS-3 must actually beat rank-1 somewhere in `[16, 32)` on KKT matrices,
   by a margin larger than the measured run-to-run spread.
2. The winning band must be one a real pounce caller occupies.
3. The bit-identity contract must be either preserved or consciously traded
   away with human approval and a recorded justification.

If (1) fails, the answer is "leave 32 alone, fix the citation" and nothing
ships. That is a legitimate outcome and is the one to expect if the D3
rule of thumb was roughly right.

## Method

`crates/feral-diagnostics/src/bin/probe_blas3_crossover.rs`. For each of the
seven large KKTs and each `nrhs` in {2,4,6,8,10,12,14,16,20,24,28,31,32,40,48},
**both kernels are timed at the same `nrhs`, alternating within each of 9
replicates, in one process**. The `kernel-probe` Cargo feature (off by default,
compiled out entirely) exposes `set_blas3_nrhs_threshold`, so the dispatch can
be flipped between the two solves. The probe also reports
`max |rank1 - blas3|`, since that difference is what a threshold cut would
trade away. All buffers are allocated outside every timed region.

### The method this replaced, and why it failed

The first attempt built two binaries — stock (32) and patched to 2 — and ran
them alternating, treating alternation as sufficient defence against drift. It
is not, and the data was discarded rather than reported. A full sweep takes
~25 minutes, so "adjacent" runs are 25 minutes apart: blocked measurement with
an interleaved label. Two controls caught it. The looped single-RHS path, which
is byte-identical in both builds and must read 1.00, read **1.15, 1.58, 1.97,
2.10**. And at `nrhs >= 32`, where both builds dispatch to the *same* kernel and
the ratio must be exactly 1.00, it read **0.86-0.90**. Full write-up in
`dev/tried-and-rejected.md`.

## Results

### The threshold is the wrong question. The row stride was.

Timing both kernels at the same `nrhs` shows a sawtooth, not a crossover.
`rank1 / blas3` on bcsstk38 — above 1.0 means BLAS-3 wins:

| nrhs | 2 | 4 | 6 | **8** | 10 | 12 | 14 | **16** | 20 | **24** | 28 | 31 | **32** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ratio | 1.00 | 0.83 | 0.79 | **2.12** | 1.40 | 1.06 | 0.89 | **1.60** | 1.00 | **1.36** | 1.14 | 1.03 | **1.31** |

BLAS-3 wins decisively at every multiple of 8 and degrades as `nrhs` moves
away from one. The period is `NR = 8`.

The cleanest statement of it needs no curve fit and no cross-process
comparison — compare `nrhs = 31` against `nrhs = 32` *within one run*.
`nrhs = 31` is 3% *less* work, so a healthy kernel gives ~0.97:

| matrix | bcsstk38 | r05_kkt | bratu3d | qap15_kkt | dirichlet120_kkt | cont-201 | cont5_late_kkt |
|---|---|---|---|---|---|---|---|
| t(31)/t(32) | 1.61 | 1.41 | 1.79 | 1.47 | 1.80 | 1.47 | 1.64 |

All seven. Doing less work took 1.4-1.8x longer.

### First hypothesis — the scalar column tail — was real but secondary

`gemm_scalar_block` handled the `nrhs % NR` column tail with an unblocked
triple loop: for *every* output element it re-walked all of `A`'s row (stride
`col_stride`) and `B`'s column (stride `nrhs`), with no reuse. Fitting
`blas3_us` on the multiples of 8 and attributing the excess to the tail
columns put a tail column at 1.6-5.1x a blocked column.

Replacing it with `gemm_tile` (full `NR`-width register tiles, unused lanes
zero-padded) moved `t(31)/t(32)` from 1.41-1.80 to only **1.39-1.51**. A real
improvement, and bit-neutral, but not the cause.

**The discriminating experiment.** `nrhs = 33` has *one* tail column;
`nrhs = 31` has *seven*. If tail-column count drives the cost they should
differ sharply. On bratu3d:

| nrhs | 31 | **32** | 33 | 34 | 36 | 38 | **40** |
|---|---|---|---|---|---|---|---|
| blas3 us | 72,298 | **46,582** | 74,154 | 74,043 | 76,060 | 80,859 | **58,760** |

`nrhs = 33` costs 1.59x `nrhs = 32` while doing 3% *more* work, and matches
`nrhs = 31` to within 3%. Tail-column count does not predict the cost. The
row stride does: 32 and 40 doubles are 256 and 320 bytes — whole multiples of
a 64-byte cache line — and 31, 33, 34, 36, 38 are not.

### Cause and fix — cache-line alignment of the row-major stride

`y` and `w` are row-major with leading dimension `nrhs` (issue #57). When
`nrhs` is not a multiple of 8, every row of every supernode panel straddles
cache lines, and the misalignment compounds across the gather, the kernels,
and the scatter.

`padded_ldw(nrhs) = nrhs.div_ceil(8) * 8` is now the row stride for the panel
path, so every row starts on a cache-line boundary. The padded columns carry
a zero right-hand side and are never read back, so every real column's
arithmetic and its order are untouched. The rank-1 path keeps the raw `nrhs`
stride — it gains nothing from the padding and would only pay for the extra
columns.

**Result — the sawtooth is gone**, and this is a within-run statistic, immune
to cross-process drift:

| matrix | t(31)/t(32) before | after |
|---|---|---|
| bcsstk38 | 1.61 | **0.98** |
| r05_kkt | 1.41 | **1.00** |
| bratu3d | 1.79 | **1.00** |

### How much of this ships

The probe forces BLAS-3 at every `nrhs`. The shipped library runs it only at
`nrhs >= 32`, so the large ratios above at `nrhs` 6-31 describe a path that
does **not** ship at those widths — they are diagnostic, not a user-visible
gain. Quoting them as the headline would overstate the result by ~35%.

The shipped regime is `nrhs >= 32` and not a multiple of 8. At `nrhs = 33`,
measured two independent ways that agree:

| matrix | t(33)/t(32) unpadded | padded | anchored speedup | raw us, unpadded -> padded |
|---|---|---|---|---|
| bcsstk38 | 1.49 | 1.24 | **1.21x** | 10,093 -> 8,272 (1.22x) |
| r05_kkt | 1.41 | 0.88 | **1.59x** | 15,194 -> 12,426 (1.22x) |
| bratu3d | 1.59 | 1.21 | **1.32x** | 74,154 -> 56,230 (1.32x) |

**Geomean 1.36x** on the multi-RHS BLAS-3 solve, for the 7-in-8 `nrhs` values
that are not a multiple of 8.

The residual `t(33)/t(32) ~ 1.21` after the fix is not leftover misalignment —
it is exactly `40/33 = 1.21`, the cost of solving the 7 padding columns. The
padding buys alignment by doing up to `8/nrhs` extra flops, so the waste is
21% at `nrhs = 33`, 7% at 100, under 1% at 1000.

**Follow-up available, not taken here:** keep the aligned stride but iterate
only the `nrhs` live columns, masking the final tile — `gemm_tile` already
supports a partial tile via its `live` parameter. That would recover the
remaining ~18% at `nrhs = 33`. It needs a second parameter (stride vs. live
width) threaded through all five kernels, so it is a separate change with its
own before/after.

**Bit-neutral**, three independent checks:

1. `gemm_tail_tests::panel_gemm_is_bit_identical_to_the_reference_on_every_tail_shape`
   asserts `to_bits()` equality against a scalar left-fold reference over 800
   shapes (16 `nrhs` x 10 `m_dim` x 5 `k_dim`), covering every `nrhs % 8` and
   `m_dim % 4` residue.
2. The probe reports `max |rank1 - blas3|`; rank-1 is untouched, so any change
   in the BLAS-3 output moves it. Unchanged at every point compared.
3. `tests/multi_rhs.rs` — all 13 pass, including both
   `assert_eq!(max_diff, 0.0)` band contracts and the BLAS-3 parity tests.

So no threshold change, no tolerance change, and no sign-off on the Finding 2
contract is needed. **`BLAS3_NRHS_THRESHOLD` stays at 32.**

### Accuracy of the two kernels, measured

Independently of the fix, the probe answers what Finding 2 asserted
qualitatively. `max |rank1 - blas3| / max |x|` is round-off, not zero:
1e-16 to 3e-16 on bcsstk38, 5e-15 on r05_kkt, 2e-14 on bratu3d. Small, but
categorically not the `0.0` the band tests assert. Finding 2 stands: a
threshold cut cannot be made bit-neutral, and remains gated on human approval.

### What this does and does not buy pounce

It is live on the Schur path today — `schur.rs:303` calls
`solve_many_refined(m, &self.afs, self.n_s)` with `n_s` problem data, so it
lands on a non-multiple of 8 seven times in eight. It does **not** touch the
IPM hot path, which is `nrhs = 1` and hardcoded (Finding 3).
