#+TITLE: Issue #10 — APP vs MAXFROMM: choosing the right TPP acceleration
#+DATE: 2026-05-16

* Problem

Feral's dense LDL^T kernel runs *threshold partial pivoting* (TPP)
on every pivot column. For each candidate pivot, the kernel scans
the column for `AMAX` (in-column off-diagonal max) and again past
the block boundary for `RMAX` (out-of-block max), then checks
`|pivot| >= u * max(AMAX, RMAX)`. On benign matrices (e.g.
1D-banded NLP KKTs like Mittelmann clnlbeam — issue #33) the
scans are essentially wasted work.

Measured per-nnz_L cost on CHAINWOO_0000 (per
`dev/research/mumps-small-frontal-speed.md` and
`dev/research/ssids-small-frontal-speed.md`):

| solver | ns/nnz_L | technique           |
|--------+----------+---------------------|
| MUMPS  |       14 | MAXFROMM-fused TPP  |
| SSIDS  |       29 | APP (block-level)   |
| feral  |       89 | TPP per-pivot scan  |

Issue #10 originally proposed APP. After the empirical SLB A/B
in issue #33 confirmed per-front kernel cost is the dominant
bottleneck (`dev/research/issue-33-slb-ab.md`), the architectural
question is now: **APP or MAXFROMM?**

This note evaluates both against the SSIDS and MUMPS reference
implementations and recommends MAXFROMM-first.

* What APP does (SSIDS reference)

(via `spral-expert` reading
`ref/spral/src/ssids/cpu/kernels/block_ldlt.hxx:289-414` and
`ref/spral/src/ssids/cpu/kernels/ldlt_app.cxx:303-1064, 2297-2486`.)

1. **Inner block factor** (`block_ldlt<T,32>`): factor a 32×32
   diagonal block with NO threshold checks. Always advances by
   `pivsiz` (1 or 2). Hard-aborts on `|bestv| < small` only.
   Output: full block factor, `d[]`, `ldwork[]`, `lperm[]`.
2. **A-posteriori check** (`apply_pivot_app + check_threshold`):
   after the block is done, scan each non-eliminated entry for
   `|aval| > 1/u`. Return the count of leading PASSED columns.
3. **Per-column reduction** (`Column::npass_`): the diagonal-block
   pass count is the upper bound; each off-diagonal apply on
   column `blk` calls `update_passed(blkpass)` taking the min.
   `adjust()` handles the split-2×2 fix-up (a 2×2 split across
   the block boundary is forced to "both-defer").
4. **Failed-pivot fallback** (`failed_pivot_method=TPP`, default):
   the failed columns are physically compacted to the trailing
   `(n - num_elim)` columns of the front; then `ldlt_tpp_factor`
   runs strictly-serial Bunch-Kaufman on the residual. Still-failing
   columns become `ndelay_out` for the parent.

Cost profile: **flat on benign matrices** (one BLAS-3 GEMM per
block, one post-test, zero retries). **Cliff on bad matrices**:
worst case ~2× the work per failed column (one wasted block
factor + one restore + one TPP retry of those columns).

Implementation surface for feral (6 phases, per the expert):
1. Port `block_ldlt<T,32>` (we have a 32×32 SIMD kernel —
   `block_ldlt32` — so this is mostly wiring).
2. Port `apply_pivot<OP_N/T>` + `check_threshold`.
3. Port `Column` (per-column reduction with 2×2 split fix-up).
4. Port `run_elim_pivoted` (outer block driver).
5. Port backup/restore + `move_back`/`move_up_diag`/`move_up_rect`
   post-processing.
6. Wire TPP fallback via `failed_pivot_method` at end of front.

* What MAXFROMM does (MUMPS reference)

(via `mumps-expert` reading
`ref/mumps/src/dfac_front_aux.F:1147-1879, 1273-1307, 1813-1879`.)

The MUMPS rank-1 update streams through column `NPIV+1` (the
next pivot column) to apply the elimination. Free byproduct:
capture `max|.|` of that column. Hand it forward via
`MAXFROMM` + `IS_MAXFROMM_AVAIL` to the next pivot search.

The acceptance test in `DMUMPS_FAC_I_LDLT:1344-1368`:

```
IF IS_MAXFROMM_AVAIL THEN
  IF MAXFROMM > PIVNUL THEN
    IF |PIVOT| >= UULOC * MAXFROMM AND
       |PIVOT| > max(SEUIL, tiny) THEN
      GOTO 415  ! accept, skip AMAX+RMAX scan entirely
    ENDIF
  ENDIF
  IS_MAXFROMM_AVAIL = .FALSE.  ! single-shot
ENDIF
```

Key invariants:
- **Single-shot** producer/consumer per pivot. The capture is
  invalidated after one use.
- **Per next-pivot column** granularity — one scalar per pivot.
- **Free in cache** — the values are already in registers from
  the rank-1 update that just wrote them.
- **No 2×2 short-circuit.** 2×2 selection always requires the
  full AMAX scan (since it needs the off-diagonal argmax).
- **`Inextpiv` is orthogonal** — a separate "skip past failed
  pivots" heuristic layered on top.

Cost profile: **smooth degradation**. Every pivot pays the same
single compare on success; on failure, fall back to the full
scan with no wasted work (the rank-1 update is committed
regardless).

Implementation surface for feral (much smaller — surgical):
1. Modify `do_1x1_update` (`src/dense/factor.rs:2832`) to
   capture `max|.|` of the FIRST updated column (col `k+1`)
   during the trailing update sweep. Stash in a new caller-owned
   `Option<f64>`.
2. Modify `scalar_pivot_step` (`src/dense/factor.rs:2310`) to
   accept the cached `maxfromm` and short-circuit the AMAX/RMAX
   scan when `|pivot| >= u * maxfromm`.
3. Hoist `inv_d = 1.0 / d` outside the column loop (it already
   is) and ensure the trailing update inner loop matches MUMPS
   inner-loop tightness.

Both arms (`fma=true` / `fma=false`) need the change. The 32×32
block-routed path (`block_ldlt32::update_1x1_block32`) is a
separate question and might be addressed by either MAXFROMM
inside the block kernel or APP wrapping it.

* Comparison

| dimension                  | APP                              | MAXFROMM                        |
|----------------------------+----------------------------------+---------------------------------|
| code surface               | 6 phases, new code path          | 2 functions modified            |
| MUMPS-reference ns/nnz     | 29 (SSIDS, ~2× MUMPS)            | 14 (MUMPS itself)               |
| benign matrices            | flat low cost                    | flat zero overhead              |
| bad matrices (≥1 fail/blk) | cliff: 2× work per failed col    | smooth: scan cost as today      |
| 2×2 handling               | split fix-up needed; some delay  | full scan; same as today        |
| interaction with SIMD      | requires block-aligned fronts;   | works on every front shape,     |
|                            | small fronts fall back to TPP    | including narrow ones (#33)     |
| handles delayed pivots     | propagates via ndelay_out (new)  | nothing new                     |
| risk of regression         | inertia-correctness corpus needs | minimal — same predicate as TPP |
|                            | careful gating; new test panel   | (`|d|>=u*max(AMAX,RMAX)`) just  |
|                            | for near-singular APP→TPP        | precomputed for one column      |
| relevance to #33 clnlbeam  | requires `ncol >= 32`, but       | applies to every front,         |
|                            | clnlbeam supernodes are narrow   | including narrow 1D-banded      |
|                            | → may fall back to TPP anyway    | supernodes                      |

* Recommendation: MAXFROMM-first

**Implement MAXFROMM in feral as the next architectural move.**
Defer APP as a possible follow-up.

Rationale:

1. **#33 is the immediate target.** clnlbeam has narrow supernodes
   (typical for 1D-banded NLP KKTs). APP requires `ncol >= 32`
   for the block factor; clnlbeam supernodes fall short and would
   route to TPP anyway under SSIDS. MAXFROMM works on every front
   shape. This is the lever that addresses #33 directly.
2. **Better per-nnz target.** MUMPS at 14 ns/nnz beats SSIDS at
   29 ns/nnz on the exact regime #33 documents. If we implement
   only one, picking the faster reference is the right call.
3. **Far smaller code surface.** ~2 functions modified vs ~6
   phases of new code. Much lower regression risk, much faster
   to land + benchmark.
4. **Easier to gate.** MAXFROMM is bit-exact equivalent to TPP
   on the test predicate (`|d| >= u*max(AMAX,RMAX)` for a
   specific column, just precomputed). The corpus inertia gates
   should pass unchanged. APP's cliff behavior requires a new
   near-singular test panel to confirm the failed-pivot fallback
   correctly delays pivots that TPP would have caught in-front.
5. **APP remains an option later** if the BLAS-3 throughput on
   big fronts becomes a binding constraint that MAXFROMM cannot
   address. The two are not mutually exclusive — MUMPS itself
   pairs MAXFROMM-TPP with BLR (block low-rank) panels; APP
   would be feral's analogue of that.

The original #10 proposal (PivotMethod enum with Tpp/App/Auto)
remains directionally correct, but the *first* alternative path
should be a MaxfrommTpp variant (Tpp + MAXFROMM acceleration),
not App. APP can be added later as a third arm if needed.

* Empirical predictions (to be measured after implementation)

Based on MUMPS's reference ns/nnz_L and the expert's per-pivot
cost analysis, MAXFROMM in feral should:

- **CHAINWOO_0000:** drop from 89 → ≤20 ns/nnz_L (5× speedup,
  closing most of the gap to MUMPS).
- **clnlbeam panel** (`diag_clnlbeam_slb.rs` panel, 20 matrices):
  factor times should drop 3–5× given how dominant the 1×1
  scalar pivot path is per #33 (97% of main-thread). Concrete
  panel target: median speedup ≥ 2.0×.
- **Indefinite KKTs (ACOPP30, etc.):** modest improvement (the
  short-circuit fires less often), within ±5% — should not
  regress.
- **Inertia corpus:** strictly unchanged. MAXFROMM is a
  precomputed cache of the same test value; the acceptance
  predicate is bit-identical.
- **Bench partitions:** small-frontal sparse p90 target moves
  from 1.74 → ≤1.20 (significant relative improvement; absolute
  target was 2.0).

If the panel speedup is < 1.5×, abandon MAXFROMM-first and
re-evaluate APP. If it's ≥ 2× and inertia gates hold, this is
the close-out path for #33 and a major step on #12 (closed meta).

* Proposed implementation plan

Phase 1 — single-thread MAXFROMM in `do_1x1_update` + `scalar_pivot_step`:

  1. Add an opt-in `BunchKaufmanParams::tpp_method:
     TppMethod::{Plain, Maxfromm}` enum, default Plain.
  2. Modify `do_1x1_update(a, n, k, fma)` to take an
     `Option<&mut f64>` for the captured max. Compute
     `max|a[(k+1)*n + i]|` for `i in (k+2)..n` during or
     immediately after the inner trailing update.
  3. Thread the captured max into the next `scalar_pivot_step`
     call via the `factor_*` driver functions (`factor`,
     `factor_frontal`, `factor_frontal_blocked*` family).
  4. In `scalar_pivot_step`, when `maxfromm.is_some()` and the
     short-circuit predicate holds, jump straight to the 1×1
     accept path (bypass AMAX/RMAX scans and the 2×2
     consideration).
  5. After consuming, set `maxfromm = None` (single-shot).

Phase 2 — corpus validation:

  1. Run the inertia gate (`external_benchmarks/consensus`) under
     `TppMethod::Maxfromm`. Must be unchanged.
  2. Run `diag_clnlbeam_slb` adapted as `diag_clnlbeam_maxfromm`
     for a Plain-vs-Maxfromm A/B on the same 20-matrix panel.
  3. Run full bench (`cargo run --release --bin bench`) and
     compare ratio partitions to baseline.

Phase 3 — default flip decision:

  - If panel median speedup ≥ 2× AND inertia bit-identical AND
    no per-matrix slowdown > +5% AND bench partitions strictly
    pass (small-frontal sparse p90 ≤ 1.74): flip
    `TppMethod::default()` to `Maxfromm`.
  - Otherwise: keep opt-in, document the finding, return to
    APP-first.

Phase 4 — 32×32 block kernel: separately decide whether to wire
MAXFROMM into `block_ldlt32::update_1x1_block32` so the block
path also benefits. This is a self-contained follow-up.

* References

- `dev/research/issue-33-slb-ab.md` — the SLB A/B that ruled
  out per-supernode driver overhead as the bottleneck and made
  #10 the next blocking lever.
- `dev/research/mumps-small-frontal-speed.md` — original 14 vs
  89 ns/nnz_L measurement.
- `dev/research/ssids-small-frontal-speed.md` — APP overhead
  measurement (29 ns/nnz_L).
- Issue #10 (open) — original APP proposal.
- Issue #12 (closed meta) — small-front gap meta-tracker.
- Issue #33 (open) — clnlbeam 97% scalar-1×1 main-thread profile.
- SSIDS reference: `ref/spral/src/ssids/cpu/kernels/block_ldlt.hxx`,
  `ldlt_app.cxx`, `factor.hxx`.
- MUMPS reference: `ref/mumps/src/dfac_front_aux.F` (FAC_I_LDLT
  at 1147, FAC_MQ_LDLT at 1677, MAXFROMM short-circuit at
  1344-1368).
