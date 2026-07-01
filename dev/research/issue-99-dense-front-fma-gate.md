# Issue #99 — dense-front throughput: packed BLAS-3 kernel + shape-aware intra-front + FMA gate

## UPDATE 3 (2026-07-01) — packed BLAS-3 trailing update: byte-exact ~8–10× on dense fronts

The maintainer asked to take on the BLAS-3 GEMM as a dedicated follow-up. The
result is a **byte-exact** win, bigger than expected.

**Diagnosis.** A full dense-front profile attributes ~94 % of factor time to the
trailing Schur update, yet it measured only **0.33–0.39 GFLOP/s** — ~10× below
even scalar peak, on a CPU with AVX2/AVX-512/FMA. An isolated kernel microbench
(`examples/bench_schur_micro`) pinned the cause: the production strided kernels
re-read the eliminated panel at **column-stride `nrow` every `q`**, so the inner
rank-`n_elim` loop touches `n_elim` cache lines spread across memory — cache
latency, not compute or DST bandwidth. (The 2026-06-30 "DST-bandwidth-bound"
conclusion, reached from a *source-panel-pack-into-same-strided-kernel* variant,
does not hold for a proper packed micro-kernel on this hardware.)

**Fix.** A packed, register-tiled (`MR=8 × NR=4`) trailing update
(`apply_schur_panel_range_packed`): pack the panel into `q`-contiguous MR/NR
micro-panels so the inner `q` loop is L1-resident, then a plain register-tiled
kernel the compiler autovectorizes. **Byte-exact** — each `A[i,j]` (i≥j) is still
reduced over ascending `q` with the identical `mul → sub` (nofma) / `mul_add`
(fma); packing changes only memory layout, not arithmetic order. Default on;
`FERAL_PACKED_SCHUR=0` restores the strided path.

**Measured (4-core x86_64):**

| case | strided | packed | speedup | correctness |
|---|---:|---:|---:|---|
| isolated kernel `bench_schur_micro` (any size) | 0.38 GFLOP/s | 9–10.5 GFLOP/s | **22–26×** | byte-exact ✓ |
| dense 1500 front, schur phase | 3165 ms | 309 ms | **10.2×** | byte-exact |
| dense 2955 front, nofma serial | 25586 ms | 3202 ms | **8.0×** (0.34→2.69 GFLOP/s) | inertia identical |
| synth qap15 stand-in end-to-end (+ intra-front) | 3379 ms | 2029 ms | **1.67×** on top | inertia identical |

**Scope (B-1).** As shipped in B-1 the packed path was reached only for
**all-1×1-pivot panels**; panels with a 2×2 pivot fell to the un-packed `axpy2`
fallback. **Phase B-2 (below) lifted this** — the packed kernel now handles a
mixed 1×1/2×2/zero-d stream byte-exactly, so strongly-indefinite fronts get the
win too.

## UPDATE 4 — Phase B-2: packed mixed 1×1/2×2/zero-d streams (byte-exact)

`apply_schur_panel_range_packed` now walks the pivot stream per element: 1×1 →
`acc -= (L[j,q]·d_q)·L[i,q]` (`mul→sub` / `mul_add`, skipping `d_q==0`); 2×2 →
the fused `acc -= dl0·L[i,q] + dl1·L[i,q+1]` (add-then-sub nofma / two chained
FMAs) with `dl0=d11·L[j,q]+d21·L[j,q+1]`, `dl1=d21·L[j,q]+d22·L[j,q+1]`. Byte-exact
with `do_1x1_update`/`do_2x2_update` and the strided `axpy`/`axpy2` fallback.
`subdiag` is threaded through the panel dispatch; `apply_blocked_schur` routes
**every** panel through packed when enabled (strided fast-path + fallback stay
under `FERAL_PACKED_SCHUR=0`).

- Correctness: full suite **736 / 0**, incl. the indefinite/2×2 KKT parity gates
  byte-exact with packed default; `packed_matches_scalar_reference_bit_for_bit`
  sweeps 1×1/2×2/zero-d in both fma modes.
- Perf: indefinite 2955 front nofma serial **25586 → 2780 ms (9.2×**, 0.34 →
  3.09 GFLOP/s).
- FMA note: on the degenerate ±n-diagonal `bench_dense_front` synthetic, fma is
  ~5× slower than nofma at equal inertia — a matrix-specific BK pivoting
  interaction (fma rounding shifts 1×1-vs-2×2 choices), not a kernel effect (both
  packed). Not a regression. Reinforces FMA opt-in.

Remaining headroom (untuned): 8×4 tile, L2 cache-blocking, explicit-SIMD (pulp)
packed kernel, FMA-in-packed — re-tune on target hardware; and re-validate on the
real qap15 KKT (needs POUNCE) at 10 cores.

**Full byte-exact stack on the synth qap15 stand-in (32000², 4 cores):**
`9247 → 3457 (intra-front) → 2029 ms (+packed)` = **4.56× byte-exact**, inertia
`(+30000,−2000,0)` unchanged; stacks further with opt-in FMA on all-1×1 fronts.

Correctness gates: the byte-exact factor-parity suite (`blocked_ldlt` 21,
`dense_ldlt` 17, `parallel_parity` 8, `parity` 8, `factor_workspace_parity` 21)
all green with packed as default, plus a dedicated
`packed_matches_scalar_reference_bit_for_bit` unit test (size sweep incl.
non-multiple-of-tile and chunk-offset cases).

---

# Issue #99 — dense-front throughput: shape-aware intra-front gate (Lever 1) + FMA row gate (Lever 3)

## UPDATE 2 (2026-07-01, session -03 cont.) — PR #92 merged, and the real win found

The maintainer authorized (a) merging PR #92's ordering fix + qap15 harness onto
this branch and (b) breaking the bit-exactness/inertia rules to see what is
possible. Both done. The headline result is **not** FMA — it is a **byte-exact**
intra-front parallelism fix (issue #99 Lever 1), found by profiling.

**Fixture:** the real qap15 KKT still cannot be regenerated (its generator needs
POUNCE + highspy + qap15.mps, absent here). So `dev/scripts/gen_synth_kkt.py`
synthesizes an *arrowhead* stand-in — a dense `K`-node border (the root front)
plus `L` degree-2 leaf columns (the LdltCompress signature) — runnable end-to-end
through `examples/bench_qap15`. Default `K=2000, L=30000` ⇒ 32000×32000, 2.06M nnz.

**The profile (`profile_qap15`) exposed the real bottleneck.** The dense border
does **not** factor as one 2000-wide front; the leaves break supernode
amalgamation so it becomes **~117 tall-thin fronts of `2000 × 16`** carrying
99.9 % of the schur loop. Both size gates key on *area*:

- the **FMA gate** was `nrow * ncol >= 65536`; `2000 * 16 = 32000` — never fired.
- the **intra-front parallel gate** is `(nrow-j_start) * n_elim >= 32768`;
  `1984 * 16 = 31744` — **just under**, so intra-front parallelism **never fired**
  and the dominant fronts ran serially even on the parallel driver.

Area is the wrong metric for tall-thin fronts: their parallelizable work is
`~n_elim * trailing_cols²`, large despite small area. Two fixes:

1. **Lever 1 (byte-exact) — shape-aware intra-front gate.** Fire intra-front
   parallelism when a front is wide enough to split (`trailing_cols >=
   INTRAFRONT_TALL_MIN_COLS = 512`) AND deep enough to amortize the fork
   (`n_elim >= INTRAFRONT_TALL_MIN_ELIM = 8`), **OR-ed** with the existing area
   gate so no already-parallel front regresses. Pure scheduling ⇒ byte-exact
   (each trailing column still reduced on one thread; `parallel_parity` holds).
2. **Lever 3 (opt-in) — FMA gate keyed on `nrow`, not area.** Renamed
   `fma_min_front_area` → `fma_min_front_rows`; fire FMA when `nrow >= t`. Catches
   the tall fronts an area gate misses.

**Measured end-to-end on the synthetic KKT (32000², 2000×16 fronts, 4-core x86_64,
`bench_qap15` steady):**

| config | ms | vs orig default | correctness |
|---|---:|---:|---|
| original default (area gates, pre-fix) | ~9247 | 1.00× | byte-exact |
| **+ shape-aware intra-front (Lever 1)** | **3457** | **2.68×** | **byte-exact** |
| **+ FMA (Lever 3, opt-in)** | **2347** | **3.94×** | opt-in (inertia identical) |

Inertia `(+30000, −2000, 0)` identical across every config. Confirmed the
diagnosis with a pure-scheduling env override (`FERAL_INTRAFRONT_MIN_AREA=16384`:
9247 → 4354 ms byte-exact) before writing the shape-aware gate (which does better,
3457 ms, without globally lowering the floor). `with_fma_large_fronts(256)` matches
global FMA end-to-end (2350 vs 2347 ms), proving the row gate fires.

**Takeaway:** the largest lever on this workload was **byte-exact** (fix the
parallel gate's shape blindness, 2.68×), not the rule-breaking FMA (a further
1.47× on top). The area→shape metric bug affected *both* gates identically.

---

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
