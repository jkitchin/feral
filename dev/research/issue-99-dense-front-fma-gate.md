# Issue #99 — dense-front throughput: per-front FMA size gate (Lever 3)

## Session context & discovered blockers (2026-07-01)

Issue #99 is a **follow-up to PR #92**, but PR #92 (`issue-91-preprocess-misfire`,
the `OrderingPreprocess::Auto` fill-verification fix that made qap15 tractable)
is **open/unmerged**. This branch (`claude/issue-99-b9zphx`) is cut from current
`main` (`a17fb7a`) and therefore does **not** contain:

- the qap15 fixture `tests/data/large/qap15_kkt.mtx` (gitignored) or its
  generator `dev/scripts/{gen,regen}_qap15_kkt.{py,sh}` (live only in PR #92);
- the harnesses `examples/{profile_qap15,bench_qap15}.rs`;
- the two research notes the issue cites for "the full diagnosis"
  (`issue-91-parallel-dense-front-2026-06-30.md`,
  `…-dense-kernel-profile-2026-06-30.md`) — these exist on **no** remote branch
  (unpushed/lost work);
- PR #92's `INTRAFRONT_MIN_AREA` recalibration (still `256*256` here).

Consequence: the end-to-end qap15 number that *defines* the issue's target
cannot be reproduced on this branch, and this container has **4 cores** (the
issue's numbers are 10-core). So parallel-scaling levers (1 assembly, 2 schur
scaling) cannot be validated against the issue's targets here.

What **is** reproducible and measurable on this hardware is the **per-core**
kernel-throughput lever (issue Lever 3): FMA vs nofma on a large indefinite
front. That is the subject of this note. A self-contained stand-in harness,
`examples/bench_dense_front.rs`, builds a synthetic indefinite front of a chosen
size (default 2955 = the qap15 root) and factors it through the real
blocked-panel path, timing nofma/FMA × serial/intrafront with an
inertia-equality gate.

## The lever

The trailing-update kernel has two numerically-distinct SIMD paths, both already
implemented in `src/dense/schur_kernel.rs`:

- `schur_panel_minus_nofma_strided*` — explicit `mul` then `sub`, two roundings
  per multiply-accumulate; **bit-exact** cross-arch with the scalar reference.
  This is the production default (`BunchKaufmanParams::fma = false`).
- `schur_panel_minus_fma_strided*` — one `mul_add` per accumulate (single
  rounding). ~2× arithmetic throughput on x86 V3 (AVX2+FMA) and aarch64 NEON.
  One ULP per accumulate off the nofma reference; **not** bit-exact cross-arch.

FMA is opt-in and **global** today: `Solver::with_fma(true)` →
`NumericParams::fma` → (solver.rs:966) `bk.fma` → every front uses FMA. The owner
kept it opt-in on purpose (`dev/tried-and-rejected.md` 2026-04-14): on 4 KKT
matrices (ACOPP14_0001, ACOPP30_0004, FBRAIN3LS_0848/0851) the FMA rounding
perturbs the Bunch-Kaufman pivot classification.

## Design — additive, default-off, front-size gated

Add `BunchKaufmanParams::fma_min_front_area: Option<usize>` (default `None`).
When `Some(t)` **and** the front's `nrow * ncol >= t`, the dense front factor
uses the FMA kernels **even if `bk.fma == false`**; small fronts stay nofma.
This is exactly the issue's Lever 3 ask: "opt-in FMA-on-large-fronts gated so the
4 small-front pivot-drift KKTs keep nofma."

- **Single insertion point:** `factor_frontal_blocked_in_place_with_scratch`
  (`src/dense/factor.rs`) — both the sequential and parallel multifrontal drivers
  funnel every front through it. At entry, derive
  `effective_fma = params.fma || area_ge_threshold`, and if it differs from
  `params.fma`, shadow `params` with a local clone carrying `fma = effective_fma`.
  Everything downstream reads `params.fma` unchanged.
- **Plumbing mirrors `fma` exactly:** `NumericParams::fma_min_front_area`
  (default `None`) → funnel at solver.rs alongside `bk.fma = fma` →
  `Solver::with_fma_large_fronts(area)` setter.
- **Default `None` ⇒ zero behavior change** anywhere. No CI/corpus regression
  risk; the production bit-exact contract is untouched. This delivers the lever
  as a knob with measured evidence, leaving the *default-on* policy decision
  (which changes cross-arch bit patterns) to the owner.

## Correctness

- `bench_dense_front` asserts inertia is identical across nofma/FMA/serial/
  intrafront on the synthetic root (the throughput-lever correctness gate).
- A unit test factors one large front with `fma_min_front_area = Some(small)`
  vs the default and asserts (a) the gate flips FMA on for a large front
  (result matches an explicitly `fma=true` factor, bit-for-bit) and (b) a
  below-threshold front is byte-identical to the nofma default.

## RESULTS (measured on this 4-core x86_64 container)

`cargo run --release --example bench_dense_front 2955 5` (n=2955 = qap15 root
size, best of 5 reps), rayon_threads=4:

| variant           | time (ms) | GFLOP/s | vs nofma-serial |
|-------------------|----------:|--------:|----------------:|
| nofma serial      |  25586.15 |    0.34 |           1.00× |
| nofma intrafront  |   8631.85 |    1.00 |           2.96× |
| **fma serial**    |  15422.96 |    0.56 |       **1.66×** |
| **fma intrafront**|   5142.82 |    1.67 |       **4.98×** |

`inertia = (+1478, −1477, 0)` — **identical across all four variants** ✓.

Findings:

1. **FMA is a real per-core win here: 1.66× serial** (25586 → 15423 ms), holding
   at **1.67× inside the intrafront path** (8632 → 5143 ms). Larger than the
   issue's cited +23–32% because this measures the *pure* front with no
   assembly/tree overhead diluting it. Inertia is unchanged — the gate is safe on
   a well-conditioned indefinite root. This confirms Lever 3 is worth wiring.
2. **FMA and intrafront compose multiplicatively** → 4.98× over the nofma-serial
   baseline on 4 cores.
3. **The absolute ceiling is still low: 1.67 GFLOP/s.** A tuned BLAS-3 core is
   ~50–100 GFLOP/s. feral's rank-`n_elim` panel update streams the whole trailing
   submatrix per panel (O(1) arithmetic per byte loaded, memory-bandwidth-bound),
   whereas faer's 2-D register-tiled GEMM gets O(n) arithmetic per byte. That
   ~30–50× structural gap is the `dev/plans/dense-kernel-blas3.md` rewrite — a
   multi-session effort, **not** something any single lever in this issue closes.
   FMA-on-large is the largest *single, low-risk, contract-preserving* step
   available on this hardware.

## Scope delivered this session vs. deferred

**Delivered (additive, default-off, parity-tested):** the per-front FMA size gate
+ `Solver::with_fma_large_fronts` + `bench_dense_front` harness + this note.
`tests/issue99_fma_front_gate.rs` pins bit-identity to `fma=true` above the
threshold, bit-identity to nofma below it, and inertia preservation.

**Deferred (blocked here):**
- **Lever 1 (adaptive `INTRAFRONT_MIN_AREA`)** — the issue requires "verify no
  regression across the bench corpus," which needs the bench corpus + a
  representative core count. Not validatable on this 4-core box without the
  fixtures; a speculative retune could silently regress the tuned constant.
- **Lever 2 (assembly parallelism)** — deep, and its win is parallel-scaling,
  unvalidatable against the issue's 10-core target here.
- **Default-on FMA-large / Lever 4 (static-SQD)** — cross-arch bit-pattern /
  inertia-perturbation policy calls the owner deliberately deferred (see the
  interactive question that the harness failed to deliver). The gate is left
  **opt-in**; flipping a default is not this session's call to make.
- **BLAS-3 2-D tiled GEMM** — the only path to faer-class GFLOP/s; tracked by
  `dev/plans/dense-kernel-blas3.md`.
