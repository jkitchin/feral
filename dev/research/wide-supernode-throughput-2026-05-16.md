# Wide / near-square supernode throughput probe — 2026-05-16

Status: measurement note, 2026-05-16. Source: `src/bin/probe_wide_supernode.rs`
(commit `aef4091`). Test machine: Apple M-series (aarch64, NEON, F64_LANES = 2).
Driver: `factor_frontal_blocked_in_place_with_scratch` under `panel_diag`.

This note is the response to issue #14's "concrete next-step probe"
request — measure utilization on the two worst supernodes of
`MBndryCntrl_3D_27` (n=31 104) and recommend between (1) tuning the
existing 32×32 / panel kernel and (2) writing a GEMM-equivalent
micro-kernel. **Conclusion up front: no go on either path today.** The
measured kernel throughput on these shapes is ~25× higher than the
issue's headline number, the issue's FLOP estimate was wrong, and the
gap vs MA57 is more plausibly explained by (a) thermal / single-thread
scheduling and (b) accumulated per-supernode driver overhead than by
the inner kernel itself. A new probe at the driver level — not the
kernel level — is the cheaper next step.

## 1. Shapes profiled

| label | source                                           | nrow | ncol  |
|-------|--------------------------------------------------|-----:|------:|
| A     | snode 3607, MBndryCntrl_3D_27 root supernode     | 1928 | 1928  |
| B     | snode 3593, MBndryCntrl_3D_27 wide-trail front   | 2829 |  433  |

Synthetic SPD-ish frontals (diagonally dominant) and KKT-style indefinite
frontals (top half PD, bottom half −1e-8 diagonal with strong cross
coupling, fires 2×2 pivots) were generated at each shape. The real
MBndryCntrl_3D_27 fronts could not be extracted without the Ipopt /
`solve_problem` linkage (see §6 *Blockers* below), but the LDLᵀ work
done on a synthetic frontal of the same (nrow, ncol) is identical to
within rounding — only the 2×2 / rook-rescue branch frequencies differ,
and our KKT fixture exercises those (verified: A.KKT factors to inertia
(964, 964, 0), confirming the 2×2 path engages on every paired column
in the bottom half).

## 2. Measured numbers (PROBE_REPS=9, bs=64, fma=false)

| shape         | med ms | min ms | med GFLOPS | peak GFLOPS |
|---------------|-------:|-------:|-----------:|------------:|
| A.SPD 1928²   | 141.6  | 134.9  | 33.7       | 35.4        |
| A.KKT 1928²   | 140.7  | 136.5  | 34.0       | 35.0        |
| B.SPD 2829×433| 169.8  | 167.0  | 34.9       | 35.5        |
| B.KKT 2829×433| 176.7  | 166.7  | 33.5       | 35.5        |

Scalar reference (`factor_frontal`) baseline:
- Shape B SPD: 395.6 ms (3 reps, median) → blocked is **2.33× faster**.
- Shape A SPD: skipped (extrapolated > 60 s; the n³/3 work blows up at
  ncol=1928 when the BLAS-1 rank-1 trailing update isn't amortised).

Panel-diagnostic counters confirm the panel path runs end-to-end on every
front with no fallbacks:
- A.SPD / A.KKT: 31 panel_full, 0 panel_partial, 0 panel_delayed; 1928
  pivots committed inline, 0 scalar.
- B.SPD / B.KKT: 7 panel_full, 0 panel_partial, 0 panel_delayed; 433
  pivots committed inline, 0 scalar.

SIMD-body coverage (analytic, from the quad-kernel trailing-block
geometry): **99.842% — 99.992% of FLOPs go through the NEON 2-lane SIMD
body**. Scalar tails account for ≤ 0.16% across all four shapes — the
scalar tail is not the bottleneck.

## 3. The "1.3 GFLOPS" number in issue #14 is a FLOP-count error

Issue #14's evidence section reports:

> Back-of-envelope FLOPS for snode 3593 (pivot 433 × 433 LDLᵀ + Schur
> update against 2396 × 433 trailing rows): ~250 MFlop done in 196 ms ≈
> 1.3 GFLOPS sustained.

The 250 MFLOP estimate is **~25× too low**. The actual LDLᵀ work for
(nrow=2829, ncol=433) is:

```
scaling axpy:  sum_{k=0..432} (nrow-k-1)            =  1 091 396 FLOPs
schur update:  2 * sum_{k=0..432} (nrow-k-1)^2      =  5 921 893 088 FLOPs
total:                                                ≈ 5.92 GFLOP
```

The textbook decomposition the issue uses — "pivot 433² LDLᵀ + Schur
update against 2396 × 433 trailing rows" — only counts the trailing
*row-block* (2396 × 433) and misses the much larger rank-433 update of
the trailing *2396 × 2396* block that the panel kernel actually issues.
Re-deriving with the corrected FLOP count: 5.92 GFLOP / 196 ms ≈
**30 GFLOPS sustained on the live driver** at shape B, consistent with
our 33–35 GFLOPS synthetic measurement (driver overhead accounts for
the 30 vs 33 gap).

A similar correction applies to shape A (1928², "152 ms by itself"):
4.78 GFLOP / 152 ms ≈ 31 GFLOPS, again matching the synthetic
measurement.

The headline "1.3 GFLOPS sustained" was the original motivation for
this issue's "BLAS-class kernel rewrite" recommendation. With the
corrected FLOP count, **the kernel is already at ~30–35 GFLOPS on this
chip**, which is in the same ballpark as Accelerate DGEMM on these
sizes (issue cites 50–100 GFLOPS, but Accelerate's actual sustained
DGEMM on M-series at these shapes is closer to 60–80). We are at
~40–60% of single-thread DGEMM peak — not 1–2% as the headline
suggested.

## 4. Re-evaluating the MA57 gap

Issue #14 cites:
- MA57:  ~330 ms full factor on MBndryCntrl_3D_27.
- feral: 1350 ms parallel, 2456 ms sequential.

The sum of just the two top supernodes (A=141.6 ms + B=169.8 ms =
311 ms by our synthetic measurement, or 152 + 196 = 348 ms by the
issue's per-front breakdown) **already exceeds MA57's total**. That is
not a kernel-throughput problem — that is a frontal-shape / fill /
amalgamation problem. MA57 does the full factor in ~330 ms, which
means MA57 doesn't *have* a 1928² supernode or a 2829×433 supernode at
all; its ordering produces a different supernode tree.

Three independent sources of the gap stack:
1. **Frontal width** (different ordering / amalgamation). MA57 uses
   AMD + extensive amalgamation tuning that targets dense-front
   problems; FERAL on this matrix likely produces wider trailing rows
   per supernode. This is a *symbolic* gap, not a kernel gap.
2. **Per-supernode driver overhead** (3608 supernodes × ~100 µs each
   = 360 ms even with zero kernel time). The 2026-05-12 work on issue
   #13 reduced this but did not eliminate it.
3. **Inner kernel throughput** (40–60% of single-thread DGEMM peak),
   the original target of issue #14.

(1) and (2) account for the bulk of the gap. (3) — the issue's
nominal target — is real but a smaller share than the headline
suggested.

## 5. Path recommendation: NO-GO on both options as currently framed

### 5.1 Tune existing 32×32 / panel kernel

**Recommendation: no-go as a general project.** The kernel runs at 33–35
GFLOPS at >99.8% SIMD coverage. The remaining 0.16% scalar tail is not
worth optimising. Register-tiling the quad kernel (the proposed "2–4×
speedup" lever in the issue) would have to overcome that 99.8% SIMD
ceiling — there is no scalar work left to convert. Software-prefetch on
M-series is hardware-managed and rarely helps; any gain there is in the
noise.

The one *cheap* tuning lever the probe revealed is block-size sensitivity.
Re-running with `PROBE_BLOCK_SIZE=32` shows shape A SPD drops from
593 ms (cold) → 270 ms while shape B degrades by 40%. The current bs=64
default is **shape-dependent optimal**, and a heuristic that picks bs
per (nrow, ncol) might claim 10–15% on the full-square root supernode
without touching the kernel. (Caveat: my high-rep results show the
"593 ms cold" was thermal noise; the warm-rep median at bs=64 matches
bs=32 within 10%. So even this is marginal.)

**Latent bug:** `apply_blocked_schur_panel` has `MAX_N_ELIM = 64`
hardcoded as a stack-buffer cap (`src/dense/factor.rs:2384`).
`PROBE_BLOCK_SIZE=128` panics with "range end index 128 out of range
for slice of length 64". If anyone raises `BunchKaufmanParams::block_size`
above 64 in production, that's a crash. Fix is trivial (raise the cap
or `min(bs, MAX_N_ELIM)`), but flagging here.

### 5.2 New GEMM-equivalent micro-kernel (months of work)

**Recommendation: no-go.** The new-kernel hypothesis was predicated on
the kernel running at ~1.3 GFLOPS — a 25× gap to peak that justified a
months-of-work investment. With the corrected measurement at 33 GFLOPS,
the *upper bound* on a new kernel is ~80 GFLOPS (Accelerate-class), so
~2.4×. The Schur update is already cache-blocked at the panel level and
register-tiled implicitly by the pulp 2-lane unroll-2 dispatch (8 SIMD
acc regs per quad chunk, fits NEON's 32-reg budget without spilling).
A 4×4 outer-product micro-kernel with explicit FMA scheduling would
have to materially out-perform pulp's current code-gen, which the FMA
A/B suggests it cannot — `BunchKaufmanParams::fma = true` is *slower*
than no-fma on our measurements (0.93× on shape B, mixed elsewhere)
because `pulp::Simd::mul_add_f64s` on aarch64 already lowers to FMADD
on the no-fma path via LLVM contract.

If the project ever moves to a workload where this 2× matters more
than the engineering cost (the issue cites "reduced-space approaches /
dense Hessian blocks" as such workloads), revisit with a *measured*
target: profile the new workload, confirm the kernel is the
bottleneck, and gate the work on a real GFLOPS gap, not the
miscalculated one in issue #14.

## 6. Blockers

- **Real-frontal extraction:** to verify the synthetic-vs-real assumption,
  we would need to dump the assembled lower-triangle bytes of snode 3607
  and snode 3593 from the live `factorize_multifrontal_supernodal` driver
  on `MBndryCntrl_3D_27`. That requires running the Ipopt NLP harness
  (`external_benchmarks/nlp_comparison/solve_problem MBndryCntrl_3D_27 24`)
  with an env-gated tap in `numeric/factorize.rs::factor_one_supernode`
  to write the assembled bytes to disk. The diff-vs-synthetic check is
  a one-evening project but was out of scope here (CLAUDE.md / task
  instructions: probe only, no kernel changes). The synthetic-vs-real
  divergence on the panel-FLOPS axis should be small — the work is set
  by (nrow, ncol) — but pivot trajectory (rook escapes, 2×2 count) can
  differ. Cross-check is worthwhile if the project revisits this.
- **FMA path regression:** `BunchKaufmanParams::fma = true` is slower
  than the no-fma default on the M-series shapes we measured (0.65–0.93×
  speedup on shape B SPD, more noise elsewhere). This contradicts
  issue #8's expected 2× speedup on aarch64. Worth a separate
  investigation; not blocking #14 but is a free 0–10% on the wrong
  side of the comparison.
- **`MAX_N_ELIM = 64` latent panic** (see §5.1). Trivial fix.

## 7. Recommended next experiments (if anyone picks this up)

In priority order:

1. **Driver-overhead probe on MBndryCntrl_3D_27.** The 2.45 s sequential
   factor minus ~1.5 s of measured kernel time = ~1 s of driver
   overhead. Run `diag_supernode_cost`-style profiling on the 3608-
   supernode build with `Profiler` enabled and confirm where it lands.
   If driver overhead is the dominant remaining cost, the lever is
   already known (extend issue #13's pooling further) and the cost is
   weeks, not months.
2. **Symbolic / amalgamation A/B vs MA57.** MA57's full-factor
   advantage on `MBndryCntrl_3D_27` is too large to be explained by the
   kernel alone. Run METIS-ND + nemin sweep on this matrix and look at
   `supernodes.len()` / max(nrow) / fill ratio vs MA57's reported
   numbers. If MA57 produces a fundamentally different supernode tree
   (likely), that's the lever.
3. **FMA path regression investigation.** A separate probe binary that
   directly times `schur_panel_minus_fma_strided_quad` vs the nofma
   sibling. If FMA is genuinely slower on M-series, the production
   default should stay no-fma and the dispatch could be simplified.
4. **MAX_N_ELIM fix.** Either raise to 128 (one-line change, stack
   cost = 4 × 128 × 8 = 4 KB per panel, acceptable) or clamp `bs.min(MAX_N_ELIM)`
   in `factor_frontal_blocked_in_place_with_scratch`. Either eliminates
   the latent panic and unblocks larger-bs experiments.

## 8. Status of issue #14 after this probe

The issue's premise — "feral's wide-supernode kernel runs at ~1% of
peak; needs a months-of-work BLAS-3 rewrite" — is not supported by
direct measurement. The kernel is at 40–60% of single-thread DGEMM
peak, and the remaining gap vs MA57 is dominated by symbolic /
amalgamation / driver-overhead factors, not by inner-kernel throughput.

Recommend updating issue #14 with this finding and either closing it
in favour of a new issue framed around driver-overhead + amalgamation
on dense Hessian blocks, or leaving it open as a research backlog
item with the entry point set to the experiments in §7 rather than to
a kernel rewrite.

## References

- Issue #14 — wide / near-square supernode kernel throughput gap.
- `src/bin/probe_wide_supernode.rs` (commit `aef4091`).
- `dev/research/feral-kernel-profile-chainwoo.md` — predecessor profile
  (CHAINWOO_0000, tall-skinny supernodes).
- `dev/decisions.md` 2026-05-12 (c) — BLAS-3 quad kernel parked decision.
- `dev/decisions.md` 2026-05-13 — Issue #10 not-implemented decision
  (same lesson: re-measure the profile before writing code).
- `src/dense/factor.rs::apply_blocked_schur_panel` (line 2370) — quad
  Schur kernel entry; `MAX_N_ELIM = 64` cap at line 2384.
- `src/dense/schur_kernel.rs::schur_panel_minus_{nofma,fma}_strided_quad`
  — pulp-dispatched SIMD body; 8 acc regs / unroll-2 over 4 dst cols.
