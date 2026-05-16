#+TITLE: Issue #33 — SmallLeafBatch A/B on 1D-banded Mittelmann NLPs
#+DATE: 2026-05-16

* Problem

Issue #33 reports pounce-feral on Mittelmann `clnlbeam` spends 97%
of main-thread time in `do_1x1_update → axpy_minus_unroll4_nofma`
(scalar 1×1 pivot path), with 8× rayon workers parked. The issue's
suggested action 1 is: re-run the SmallLeafBatch (SLB) default-flip
experiment specifically on this clnlbeam-class workload, because
the Phase 2.11 attempt was killed at within-noise on synthetic
chains and may now tip the signal-vs-noise on a real workload.

Issue #11 documents the SLB flip decision criterion: median
speedup ≥ 1.10x on the target panel AND no per-matrix slowdown
> +5%.

* Method

Built `src/bin/diag_clnlbeam_slb.rs`. For each matrix in the four
1D-banded Mittelmann families (clnlbeam, henon120, lane_emden120,
dirichlet120 — the matrices #33 explicitly calls out as either
scalar-1×1-bound or blocked-path-bound on this corpus), runs:

  - `NumericParams::default()` with `small_leaf: Off` (current)
  - `NumericParams::default()` with `small_leaf: On`

reports min-of-7 per config, the per-matrix speedup ratio, and
per-family and panel summaries. Symbolic factor is shared across
both configs; warm-up uncounted.

* Result

#+CAPTION: Per-family A/B summary
| family        | n  | geomean | median | min   | max    | wins (≥1.05x) | losses (≤0.95x) |
|---------------+----+---------+--------+-------+--------+---------------+-----------------|
| clnlbeam      |  2 |   1.01x |  1.01x | 1.00x |  1.02x | 0             | 0               |
| henon120      |  6 |   1.00x |  1.00x | 0.98x |  1.05x | 1             | 0               |
| lane_emden120 |  6 |   1.00x |  1.00x | 0.95x |  1.07x | 1             | 1               |
| dirichlet120  |  6 |   1.00x |  1.00x | 0.98x |  1.01x | 0             | 0               |

Panel total (n=20): geomean 1.002x, median 0.999x, min 0.947x,
max 1.071x, 2 wins, 1 loss.

**Decision: FAIL the #11 criterion.** Signal is within ±7%
measurement noise. The SLB flip is empirically not the lever for
clnlbeam-class workloads.

* Per-matrix observations

The matrices that show ANY benefit (1.05x-1.07x) all have very
wide small-leaf groups (`avg=200+`):
- `henon120_0000`: 153 groups × 200.8 avg members → 1.05x
- `lane_emden120_0000`: 240 groups × 233.5 avg members → 1.07x
- `dirichlet120_0000`: 240 groups × 217.5 avg members → 1.01x

The matrices with many small groups (`avg=4.7–6.9`) show ZERO
benefit:
- `henon120_0001-0005`: 437 groups × 6.6 avg → 0.98x..1.00x
- `lane_emden120_0001-0005`: 1043 groups × 6.9 avg → 0.95x..1.02x
- `dirichlet120_0001-0005`: 829 groups × 4.7 avg → 0.98x..1.01x

This shape is consistent with the hypothesis that per-supernode
driver overhead is NOT the dominant cost. Even with thousands of
tiny supernodes batched, the speedup is 1.00x — the per-front
kernel work (TPP scan + scalar Schur update) dwarfs the
per-front malloc/setup overhead that SLB was designed to
amortize.

* Interpretation

This is the empirical answer #11 was waiting for, run on the
specific panel #33 hypothesized would tip the signal:

**The per-front kernel really is the bottleneck.** SLB cannot
help. APP (#10) is required to move the needle.

The two wins on the iter-0000 matrices (1.05x-1.07x with wide
groups) reflect a small kernel-launch amortization that's real
but tiny — not enough to motivate flipping the default given the
per-matrix worst-case slowdown of -5.3% (`lane_emden120_0002`).

* Recommendation

1. **Do not flip `SmallLeafBatch::default()`.** Append a new entry
   to `dev/tried-and-rejected.md` covering this A/B, with the
   per-family data table above as evidence. The Phase 2.11
   measurement was on synthetic chains and was correctly
   inconclusive; this measurement is on the actual workload
   #33 calls out and is conclusive.
2. **Keep #11 open**, with a comment noting the measurement
   shifted the criterion: flipping SLB now requires evidence
   that *with* APP (#10) the per-front overhead becomes visible.
   The flip decision is deferred until #10 lands and a fresh
   panel re-run shows the speedup tipping out of noise.
3. **Issue #33 closure path:** route through #10 APP. The 97%
   scalar-1×1 dominance reported by pounce-feral is the same
   regime #12 documented on CHAINWOO; APP is the architectural
   change that addresses it. Add a comment to #33 with this A/B
   result so the consumer knows the SLB path was tried and
   rejected on their workload.

* References

- `src/bin/diag_clnlbeam_slb.rs` — this binary.
- `dev/tried-and-rejected.md` Phase 2.11 entry — original SLB
  flip attempt on synthetics.
- Issue #11 — re-evaluate SmallLeafBatch default flip.
- Issue #12 (closed meta) — 6× per-nnz gap vs MUMPS on chain KKTs.
- Issue #10 (open) — APP pivoting path. The blocking lever.
- Issue #33 — clnlbeam 97% scalar-1×1 profile.
