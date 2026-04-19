# Dense-Kernel Work for the VESUVIO-Class Factor Tail

**Status:** Pre-planning research note
**Date:** 2026-04-18
**Related plans (stale, see §1):** `dev/plans/phase-2.4.1-blocked-ldlt.md`,
`dev/plans/phase-2.4.2-simd-schur-kernel.md`
**Related research:** `dev/research/dense-ldlt.md`,
`dev/research/phase-2.4.2-simd-schur-kernel.md`
**Related abandoned attempts:** `dev/tried-and-rejected.md`
2026-04-14 entries on Phase 2.4.1a and Phase 2.4.2 unroll4-FMA
**Diagnostics:** `src/bin/vesuvio_diag.rs`,
`dev/journal/2026-04-18-08.org` entries 2 (VESUVIO) and 5 (MUONSINE)
**Goal of this note:** decide what to do *next* about the
factor/MUMPS tail that the session-08 diagnostics attributed to
the dense kernel — not to re-derive what the 2.4.1/2.4.2 plans
already say.

## 1. What is actually in the tree today

The two existing plans were drafted on 2026-04-14, before the
NEON unroll4 work landed. They are accurate about the algorithm
but stale about the starting point.

Current state of `src/dense/factor.rs` and `src/dense/schur_kernel.rs`:

- `do_1x1_update` and `do_2x2_update` call
  `axpy_minus_unroll4_nofma` / `axpy2_minus_unroll4_nofma` on
  aarch64 only (factor.rs:1096–1101, 1132–1137). All other
  targets fall back to a scalar inner loop that rustc auto-
  vectorizes only modestly.
- The `nofma` variants do separate `mul` + `sub` so the result is
  bit-identical to scalar — this is the mitigation for the four
  ACOPP/FBRAIN inertia regressions documented in
  `tried-and-rejected.md` 2026-04-14.
- The pulp-dispatched FMA kernels (`axpy_minus_unroll4`,
  `axpy2_minus_unroll4`) and the criterion microbench remain in-
  tree at `src/dense/schur_kernel.rs` and `benches/schur_kernel.rs`,
  unwired.
- No blocked / panel outer driver was ever wired. Phase 2.4.1a
  (contribution-block deferral, scalar) was implemented and
  reverted; 2.4.1b (faer-style fully blocked) was never attempted.
  Both were declared mooted in the 2.4.1a tried-and-rejected entry
  on the grounds that "loop reordering without a SIMD trailing-
  update kernel is pure overhead". That conclusion predates the
  NEON kernel; it should be re-examined now that we have one.

So the live questions for the VESUVIO tail are not "should we
build a blocked kernel" or "should we adopt SIMD" — they are
"given a 4-way-unrolled NEON inner kernel, what is the actual
ceiling, and which of (wider SIMD, x86 port, blocked outer driver,
something else) closes the remaining gap most cheaply?"

## 2. The workload, sharpened

Session-08 diagnostics identified four matrices in the IPM bench
tail with the same shape (`vesuvio_diag.rs`):

| matrix       |  n  |  max_col_nnz  | root frontal | factor (us) | MUMPS (us) | ratio |
|--------------|----:|--------------:|--------------|------------:|-----------:|------:|
| VESUVIOU_0000| 3083|          1026 | 2059×959     | ~236 000    |  ~2 800    |  84   |
| VESUVIO_0000 | 3083|          1026 | 2059×959     | ~230 000    |  ~2 800    |  82   |
| VESUVIA_0000 | 3083|          1026 | 2059×959     | ~230 000    |  ~2 800    |  82   |
| MUONSINE_0000| 1537|           512 | 1024×315     |    21 555   |    369     |  60   |

All four are arrow-KKT systems where a single dense linking
column over `≈ n/2` trivial slacks forces any reasonable
ordering to push the linking column to the root. The root
frontal is then dense, well-conditioned, and large enough that
its factor cost dominates the multifrontal sweep.

**Cost model with the current scalar-equivalent NEON kernel.**
M-series cores reach ~3.4 GHz × 2-wide FP64 NEON × ~2 instructions/
cycle on a non-FMA AXPY — call it 12–14 GF/s effective. The 2059×959
root frontal is `nelim · cr · (cr+1)/2 ≈ 959 · 2059·2060/2 ≈ 2.0 GFLOP`
on the rank-1 cascade. At 12 GF/s that is 167 ms. Observed VESUVIO
factor is ~236 ms — about 70% of the kernel time accounted for by
the rank-1 cascade, the rest is pivot search, 2×2 logic, and per-
pivot overhead. The model holds.

MUMPS on Accelerate runs DGETRF for the same shape at ~400 GF/s
(BLAS-3 register-blocked, AVX2-on-Rosetta or NEON via vendor
kernel) → 2.0 GFLOP / 400 GF/s = 5 ms. Observed 2.5–2.8 ms (better
than this estimate by 2× because MUMPS exploits symmetry). The
**~50× kernel-throughput gap** between our NEON-AXPY and
Accelerate-DGEMM is exactly what the factor ratio measures.

So the question reduces to: by how much can we close 50× without
a vendor BLAS, and at what implementation cost?

## 3. Three candidate levers

### Lever A — wider / FMA-fixed SIMD (no blocking)

Replace `axpy_minus_unroll4_nofma` with a kernel that exposes more
ILP without the inertia-flip risk.

- **(a1) NEON unroll8.** Go from 4 to 8 independent FMA
  accumulators. M-series has 4 FMA pipes; 8 accumulators saturates
  better than 4. Estimated win: 1.3–1.6× over current. Bit-exact
  with scalar if we keep the `nofma` mul+sub form.
- **(a2) NEON FMA with inertia mitigation.** Use FMA for the
  inner accumulation, then on rejection-boundary fronts re-run
  the trailing pivots scalar. Risk: complex to gate; the
  rejection-boundary heuristic is fragile.
- **(a3) x86_64 AVX2/FMA path.** The current kernel is aarch64-only.
  Adding x86_64 lifts Linux/Intel users out of the auto-vectorized
  scalar fallback. Same FMA-vs-non-FMA tradeoff as a2.

**Ceiling.** Lever A is bounded by AXPY memory bandwidth on the
trailing block (each `dst[i] -= α·src[i]` reads two doubles and
writes one, no reuse across iterations). On M-series L1 that is
~30–40 GF/s peak. Even doubling our kernel to 24–28 GF/s would
take VESUVIO from 236 ms to ~120 ms → ratio 84 → 42. Better but
still not MUMPS-class.

### Lever B — blocked outer driver (faer-style 2.4.1b)

Defer the trailing rank-1/rank-2 updates over a panel of `bs ≈ 64`
columns and apply them as a single triangular-rank-`bs` update at
the panel boundary. The deferred update is a strict-lower-
triangular `A_22 -= W · L^T` where `W` and `L` are panel-tall and
have `bs` columns. This converts the kernel from BLAS-2 (AXPY,
memory-bound) to BLAS-3 (rank-`bs` update, compute-bound on a
register-tiled GEMM).

- **Why it might work now (vs the 2026-04-14 verdict).** The
  2.4.1a tried-and-rejected concluded that loop reordering without
  a SIMD inner kernel is overhead. That's still true. But we now
  *have* a SIMD inner kernel, and the BLAS-3 reformulation
  amortizes the memory-bandwidth cost across `bs` register-resident
  accumulators per `dst[i]` location instead of one. Faer's
  measured speedup at this exact reformulation (with pulp register
  tiles) is ~5–10× over BLAS-2 SIMD on dense indefinite matrices
  of comparable size.
- **Why it might still not work.** Our rank-1 kernel is 4-way
  unrolled but not register-blocked across `dst` and `src`. The
  faer speedup comes from the `Ukr<MR, NR, T>` 2D register tile,
  not from deferral itself. Implementing a register-tiled
  triangular GEMM in pure Rust is a ~500-line, 1–2 session effort
  with non-trivial verification (the partial 2×2 boundary state at
  panel edges and the pivot-rejection fallback both need careful
  handling — see §2.4.1b plan, "Risks").

**Ceiling.** Faer hits ~150 GF/s for this kernel on M-series. At
that level VESUVIO drops to 2.0 GFLOP / 150 GF/s = 13 ms (factor
ratio 5×) — close to but not at MUMPS parity. Combined with
lever A this is plausibly the path to ratio 2–3×.

### Lever C — exploit the arrow-KKT structure directly

The VESUVIO root frontal is dense *because* the linking column
got merged into a supernode of `≈ n/2` slack columns. But the
slack structure is special: the contribution from each slack to
the linking column is a single-element rank-1 update. If the
multifrontal driver detected the arrow pattern and elided the
slack supernode entirely (eliminate the slacks first as
diagonal 1×1 pivots, then form a *small* dense Schur complement
on just the linking variables), the root frontal would shrink
from ~2000×1000 to ~50×50.

- **Risk.** This is workload-specific — it might pessimize other
  KKT patterns. Needs a feature-detection step in symbolic.
- **Upside.** Largest potential win (factor ~100× if it works);
  also fixes solve cost on the same matrices.
- **Where it lives.** Not in `src/dense/`; this is a symbolic /
  multifrontal-driver change in `src/symbolic/supernode.rs`.

This lever is structurally different from A and B and out of
scope for the dense-kernel work. Listed here only because any
research note that ignores it would mis-rank the dense work
against alternatives. It deserves a separate research note if
the dense-kernel route stalls.

## 4. Recommended next steps

The 2.4.2 plan as drafted is the right structure but the wrong
gate. We should *not* re-litigate the SIMD-vs-pulp choice or the
contribution-block deferral. The first action is a 1–2 hour
**measurement pass** that the existing plans assumed but did not
actually do:

1. Run `cargo bench --bench schur_kernel` on the dev machine and
   record the current `axpy_minus_unroll4_nofma` GF/s for L = 64,
   128, 256, 512, 1024. This pins lever A's ceiling.
2. Add a one-shot timer in `vesuvio_diag.rs` that breaks the
   factor cost on VESUVIOU into (a) `do_1x1_update` time, (b)
   `do_2x2_update` time, (c) pivot search, (d) other. This pins
   how much of the 236 ms is even reachable by kernel work.
3. Implement a tiny prototype of lever B at `bs = 32`, no
   pivot-rejection support, no 2×2 boundary handling — just
   measure the BLAS-3 trailing update speed on a 1024×1024
   trailing block in isolation. Compare against the rank-1
   cascade time at the same size. If lever B wins ≥ 3× in the
   prototype, commit to the full 2.4.1b plan; if not, drop it
   and pursue lever A only.

Steps 1 and 2 are pure measurement (no code in the hot path);
step 3 is a throwaway prototype in a `bin/` or `examples/`
binary. Total budget: one session.

The full 2.4.1b implementation (with rejection support, 2×2
panel boundary, may_delay interaction, six correctness tests,
and validation report) is 3–5 sessions if step 3 says "go".
The 2.4.2 follow-on — if step 1 shows we are far from peak —
is another 1–2 sessions.

## 5. What this note explicitly does not cover

- **K1 as a generic preprocessor** — separate research note,
  separate trigger condition (session-08 checkpoint, "Next
  Session Should" #2). Independent of dense-kernel work.
- **HEMSR coarsening for feral-metis** — gated on closing the
  dense-kernel gap; deferred until lever A or B ships. Tracked in
  the session-06 Mongoose recommendation list.
- **Lever C (arrow-KKT structural fix)** — flagged in §3 as
  alternative; needs its own note and plan if pursued.
- **Vendor-BLAS opt-in** — explicitly out of scope per the
  CLAUDE.md "zero non-Rust dependencies in the core solver" rule.
  The MUMPS oracle is allowed; we are not.

## 6. Decision pending

This note recommends executing the §4 measurement pass before
authoring a fresh plan document. If the measurements support
lever B, the existing `dev/plans/phase-2.4.1-blocked-ldlt.md`
is the right starting point — it needs updating to reference the
NEON kernel and to drop the "scalar-only" framing of step 3, but
the structure (panel routine + apply_blocked_schur + step-by-step
implementation order + 6 correctness tests + risks) stands.
If the measurements support lever A only, the existing
`dev/plans/phase-2.4.2-simd-schur-kernel.md` covers steps 4–7
verbatim; we'd add an unroll8 / x86_64 step to its top.

Either way, the next deliverable after this note is a measurement
report, not code. That is the action item this note authorizes.
