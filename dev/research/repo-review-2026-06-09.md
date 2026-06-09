# Whole-repo code review — 2026-06-09

Scope: full review of `src/` (dense, sparse/symbolic/ordering, numeric,
lu, scaling/io/capi), `src/bin/bench.rs`, and the six ordering crates
under `crates/`, hunting for silent bugs, logical flaws vs the published
algorithms, performance issues, and cross-path inconsistencies. Conducted
as six parallel module audits, then consolidated here.

Each finding carries: location, severity (high/medium/low), confidence
(certain/likely/possible), what's wrong, and why it matters. Issue IDs
are stable within this document (D = dense, N = numeric, S = sparse/
symbolic/ordering, L = lu, X = scaling/io/capi/bench, O = ordering
crates) so individual items can be filed as GitHub issues and checked
off.

**Verified-correct cores (no findings, recorded so future reviews can
skip re-deriving them):**

- Extend-add child→parent index mapping incl. delayed-column
  canonicalization (`numeric/factorize.rs:3723-3750`)
- Delayed-pivot layout `[own | delayed | trailing]` and re-entry
  bookkeeping; inertia accumulation (each pivot counted exactly once at
  its eliminating node; Schur columns excluded)
- perm vs perm_inv conventions across factorize/solve/scaling
- Scaling bracket algebra (factors hold D·A·D; refinement residuals
  against the unscaled matrix) — factorize↔solve consistent
- Parallel task-graph synchronization (deposit-under-mutex before
  AcqRel fetch_sub; leaf collection before scope; error stall)
- Workspace reset invariants (row_map, build_seen mirror-clear,
  from_pooled_buf lower-triangle zeroing)
- Liu etree, Gilbert/Ng/Peyton column counts (faithful to CSparse
  cs_counts.c), fundamental-supernode detection, trivial_chain merge
- Hungarian/MC64 numerical core: dual feasibility, augmenting paths,
  reduced-cost formula, exp clamp at ±709, log-of-zero impossible
- LU permutation algebra on both paths (P B Q = L U, FT eta ordering,
  BG update algebra, Gilbert–Peierls reach via sorted topological order)
- Sparse LU bump rollback coverage and diagonal-first invariant
- AMD port in feral-ordering-core (pme2_excl underflow guard, wflg
  overflow margins, hash-bucket hijack/restore, absorbed-supervariable
  expansion)
- metis König separator extraction; kahip push-relabel + gap heuristic;
  scotch compress_graph determinism and expand_perm validation
- schur_kernel.rs SIMD kernels (strided prologues, split_at_mut carving,
  body/tail splits match the rank-1 reference rounding chain)

---

## 1. High severity

### D1. Legacy `factor()` corrupts the next column after a ForceAccept'd zero pivot

`src/dense/factor.rs:4471-4482` (`do_1x1_pivot` strict-zero ForceAccept
branch) and the 2×2 twin at `:4649-4655`. Severity **high**, confidence
**certain** (code path) / **likely** (practical reachability).

The branch returns `(gamma0_next, r_next) = (0.0, k+2)` as the fused
argmax for the next column — but no rank-1 update ran, so column `k+1`
keeps its original off-diagonals. Every caller stores the fused value
(`factor.rs:936-951, 963-978, 985-1000, 1037-1051`); the next iteration
sees `gamma0 == 0.0`, takes the "zero off-diagonal column" branch
(`factor.rs:913-929`), and **discards the real off-diagonals of column
k+1** (set_l_column_identity + skipped trailing update + 1×1 count from
the untouched diagonal). Silent factor corruption and potentially wrong
inertia — the project's hard contract.

Reachability: legacy `factor()` only, with `ZeroPivotAction::ForceAccept`,
strict-zero pivot passing the BK alpha test at `k ≤ n-3`. `factor()` is
production-reachable via `SchurBlock::solve_with`
(`numeric/factorize.rs:824`) and the crate-root `factor` re-export.

Fix: do not return a fused next-column value when no update was applied
(return a sentinel forcing a fresh argmax), in both the 1×1 and 2×2
degenerate branches.

### L1. Dense LU update commits singular bases; dense U-solves emit Inf/NaN silently

`src/lu/dense_update.rs:85` and `src/lu/dense_solve.rs:151,163`.
Severity **high**, confidence **certain**.

The bump elimination checks pivots only for `k in q..m-1`; the final
diagonal `u[m-1,m-1]` is never validated, and when the leaving slot is
the last column (`q == m-1`) the loop body never runs at all. The dense
`usolve`/`ut_solve` divide with no zero/finite guard. A numerically
singular replacement basis (routine in simplex: degenerate ratio test)
returns `Ok(())`, then every subsequent `ftran`/`btran` silently emits
Inf/NaN. The sparse path guards both ends: `sparse_update.rs:280-293`
checks `k in r..=h` inclusive, and `sparse_solve.rs:164-180` errors with
`SingularBasis` on a zero/non-finite stored diagonal, with a regression
test (`zero_u_diagonal_errors_instead_of_inf`).

Fix: after the dense elimination loop, check the last diagonal against
`ztol` before commit; optionally port the solve-time guard + regression
test from the sparse path.

### N1. `with_fma` / `NumericParams::fma` is a silent no-op

`src/numeric/solver.rs:433-436` writes `NumericParams.fma`, but nothing
ever copies it into `BunchKaufmanParams.fma`, despite the doc at
`src/dense/factor.rs:555-558` claiming "the sparse multifrontal driver
copies `NumericParams::fma` here". All BK call sites pass `&params.bk`
untouched (`factorize.rs:2395-2405, 2575-2585, 1418-1428`); the parallel
driver clones params but only sets `bk.intrafront_parallel`
(`factorize.rs:2822-2826`). Severity **high** (dead documented feature),
confidence **certain**.

`tests/fma_opt_in_roundtrip.rs` cannot catch this: it asserts "same
inertia + small residual", which holds trivially when FMA never engages.
The documented ~2× kernel-throughput opt-in (issue #8) never fires
through the public API.

Fix: one line per driver (`local.bk.fma = params.fma` next to the
existing `intrafront_parallel` copy). Strengthen the test to assert the
kernel actually dispatched FMA (e.g. via a counter or profiler tag).

### S1. Default `postorder()` is O(n²·log n) on star-shaped etrees

`src/ordering/postorder.rs:33-37`. Severity **high** (performance, in
the default symbolic pipeline), confidence **certain** (behavior) /
**likely** (real-world impact).

The loop body clones and re-sorts `children[node]` on every stack
visit — a node with `c` children pays `c+1` clone+sorts, O(c² log c).
On a star etree (dense-border KKT row, AMD sends the border last —
exactly the arrow pattern in this codebase's own tests) the default
pipeline (`symbolic_factorize_with_method` step 3, `symbolic/mod.rs:806`)
pays O(n² log n). Correctness currently survives only because
`sort_unstable` is deterministic on identical input, so `child_idx`
indexes consistently across re-sorts — a fragile, undocumented
dependency.

Fix: copy the stack layout of the three correct siblings in the same
file (`biased_postorder` line 105, `schur_constrained_postorder`,
`EliminationTree::postorder` with its `next_child` cursor) — compute the
sorted child list once per node.

---

## 2. Medium severity — correctness and contract violations

### D2. Panel inline 2×2 path skips `static_pivot_floor`

`src/dense/factor.rs:2616-2706` (panel) vs `:3622-3631` (scalar).
Severity medium-high when the knob is on, confidence certain (code
differs) / likely (observable divergence).

`scalar_pivot_step` perturbs a sub-floor 2×2 block via
`perturb_2x2_to_floor` *before* the growth/det gates; the panel's inline
2×2 accept path has no such call (grep: only sites are 3623 and 4609).
With `static_pivot_floor > 0` (wired from
`NumericParams::static_pivot_threshold`), a block that passes the gates
unperturbed is accepted unperturbed → L, D, `n_tiny`,
`needs_refinement`, possibly inertia differ from the scalar path —
violating the module's documented bit-parity contract and the MA57
static-pivot semantics.

### N2. Static-pivot floor computed in user space, enforced in scaled space

`src/numeric/solver.rs:904-923`. Severity medium, confidence possible.

`static_pivot_floor = t · ‖A‖∞` is computed from the **unscaled** user
matrix, then enforced by the BK kernels on pivots of the **scaled**
matrix D·A·D. Under MC64/InfNorm scaling the two norms can differ by
orders of magnitude, so `t = 1e-8` may behave like `1e-2` or `1e-14` in
pivot space, drifting from the MA57 `cntl(1)` analogy the docs promise.
The F-01 null-pivot floor does this correctly (scaled infinity norm,
`factorize.rs:3302-3316`). Likely masked because ripopt's recommended
Identity scaling makes the norms coincide.

### D3. Rook rescue bypasses strict-zero / band inertia accounting

`src/dense/factor.rs:4038-4052` (splice site); `src/dense/rook.rs:281-300`.
Severity medium, confidence certain (logic) / possible (frequency).

Rook runs only after the threshold test `|d| ≤ max(u·col_max,
null_pivot_tol)` rejected the pivot. Rook's 1×1 accept gate
(`|a_rr| ≥ u·gamma_r`) has no `zero_tol`/`null_pivot_tol` floor, so it
"rescues" precisely the pivots rejected by the null floor — including
strict zeros when the whole column is noise (`gamma_r ≤ |d|/u`). These
are sign-counted with no `needs_refinement` and no zero bucket,
contradicting the issue-#54 SSIDS strict-zero rule and the F-01 band
rule that `try_reject_1x1_frontal` implements. Inertia can differ ±1
from the documented convention whenever `pivot_threshold > 0`.

### D4. Solve-time 2×2 gate inconsistent with factor-side acceptance

`src/dense/solve.rs:193-205`. Severity medium, confidence likely.

Two mismatches vs the factor side: (a) the gate uses the naive
`a*c - b*b` where the factor classifies inertia with the
cancellation-free `det_sym2x2` (`factor.rs:4197-4202`) — a genuinely
nonsingular block whose naive det rounds to 0.0 is silently *skipped*
during solve (wrong solution, no error, no flag); (b) the gate's floor
is absolute (`zero_tol_2x2` ≈ EPS² ≈ 4.9e-32) where frontal acceptance
uses the SSIDS scale-invariant floor — a validly accepted
well-conditioned block at small scale (d11 = d22 = 1e-16, d21 = 0) is
skipped at solve time.

### D5. Legacy `factor()` + ForceAccept + exactly singular 2×2 → 1/0 → NaN

`src/dense/factor.rs:4658-4662`. Severity medium, confidence certain
(path) / likely (triggering).

`t = 1.0 / (d00*d11 - 1.0)` with det exactly 0 yields ±inf; w0/w1 become
inf/NaN and are subtracted into the entire trailing block. No guard
fires: `count_2x2_inertia` under ForceAccept merely counts; the
Duff-Reid bound with default `pivot_threshold = 0.0` is vacuous; the
"degenerate" check threshold (≈2.2e-26) is unreachable since d10 is the
column max. The frontal path is protected (SSIDS det floor at
`factor.rs:3684-3696`; `do_2x2_update` early-returns on det == 0);
`factor()` has neither. NaN then propagates silently — all subsequent
pivot comparisons return false, and `d > 0.0 ? pos : neg` counts NaN as
negative.

### D6. `unsafe set_len` over uninitialized memory (library UB)

`src/dense/factor.rs:1668-1670` and `:2148-2150`. Severity medium
(soundness hygiene), confidence certain (contract violation) / possible
(real-world misbehavior).

`contrib.set_len(cdim2)` after only `reserve`, then `&mut [f64]`
materialized over uninitialized memory. Every cell is written before
read so it works today, but it violates `Vec::set_len`'s safety
contract (Miri flags it); the safety comment's write-before-read
argument does not satisfy the precondition. Fix: `spare_capacity_mut()`
+ `MaybeUninit` writes, or `resize` and zero only the upper-triangle
cells.

### L2. `pivot_threshold` silently ignored on the sparse LU path

`src/lu/sparse_factor.rs:266-268` (`let _ = utol; pivot_row = ipiv`).
Severity medium, confidence certain.

Contradicts `sparse_factor.rs:3` ("threshold partial pivoting"),
`lu/mod.rs:67-70` (parameter doc), and drifts from the dense path which
honors the threshold (`dense_factor.rs:230`). Bump elimination
(`sparse_update.rs:282-291`) also uses strict max pivoting. A user
setting `pivot_threshold = 0.1` changes dense behavior and silently
changes nothing on the sparse path, defeating the fill-reducing column
ordering for ill-scaled columns. Implement, or reject non-1.0 values on
the sparse path and fix the three doc sites.

### X1. `feral_num_neg` returns stale or wrong inertia

`src/capi.rs:401-411` vs `:140-144, 213-214, 315`. Severity medium,
confidence certain.

`neg_evals` initializes to 0, resets to 0 on `feral_set_structure`, and
is not updated when `feral_factor` returns `FERAL_SINGULAR`/`FERAL_FATAL`.
So (a) with no factor it returns 0, never the documented −1 (that
sentinel only fires for a null handle); (b) after a failed re-factor it
silently reports the previous matrix's count — a plausible-but-wrong
inertia signal to an IPM host.

### X2. MTX parser: declared nnz never validated; duplicate handling diverges

`src/io/mtx.rs:114-176`; `src/dense/matrix.rs:73-79` vs
`src/sparse/csc.rs:103-148`. Severity medium, confidence certain
(divergence) / possible (corpus impact).

A truncated file (fewer data lines than the header nnz) or a file with
extra lines parses successfully into a different matrix — a hard
corpus-integrity hazard for a project whose correctness gates run on
downloaded matrices. Separately, duplicate coordinates are **summed** by
`to_csc` (`from_triplets`) but **overwritten** by `to_dense`
(`from_lower_triangle`/`set`) — the bench compares dense-vs-sparse
failure sets, so a duplicate-bearing file masquerades as a solver
discrepancy.

### X3. Bench dense-KKT loop uses the sparse pivot params

`src/bin/bench.rs:1569` vs the rationale block at `:1356-1375`.
Severity medium (harness integrity), confidence certain (mismatch) /
likely (bug rather than undocumented change).

The comment mandates `pivot_threshold = 0.0` for the dense KKT path
(no delayed-pivot landing zone; 0.01 force-accepts/zeros structural
pivots on HYDCAR20/METHANL8/DEGENLPA/HS118) and `params_kkt_dense` is
built accordingly — but the real dense KKT validation loop calls
`factor_single_front(&matrix, &params_kkt_sparse)` with threshold 0.01.
`params_kkt_dense` is only used by synthetic micro-benchmarks. Either
dense pass rates are quietly depressed or the comment is stale; both
corrupt dense-vs-sparse triage.

### X4. MC64 partial-singular fallback covers unmatched columns but not unmatched rows

`src/scaling/mc64.rs:215-222` vs doc at `:27-28`. Severity medium,
confidence likely.

On a partial matching, index i can have column i matched while row i is
unmatched (the unmatched row/column sets differ even on symmetric
patterns). Then `u[i]` was zeroed in `build_matching`
(`hungarian.rs:703-707`) and `s[i] = exp((0 + v[i] - cmax[i])/2)` mixes
a meaningless half-dual into the symmetric average — exactly the
"duals are meaningless on the unmatched part" condition the adjacent
comment warns about. Affects PartialSingular matrices (routine for
rank-deficient KKTs per issue #43): D·A·D quality can be badly
asymmetric.

### X5. `FERAL_SCALING` / `FERAL_ORDERING` env-var vocabulary drift

`src/capi.rs:78-85` vs `src/bin/bench.rs:1285-1303, 1248-1261`.
Severity medium (silent experiment invalidation), confidence certain.

capi accepts `identity/infnorm/mc64/auto` and silently ignores anything
else; bench accepts `identity/infnorm/mc64/adaptive` and warns. So
`FERAL_SCALING=auto` works in the shim but not bench, and
`FERAL_SCALING=adaptive` works in bench but is a silent no-op in the
shim — a cross-tool experiment with one spelling measures different
configurations. `ordering_method_from_env` maps any unrecognized
`FERAL_ORDERING` (typos included) to forced AMD with no warning.

### X6. `CscMatrix::validate` lacks a col_ptr monotonicity check

`src/sparse/csc.rs:151-198`, reached from `src/capi.rs:197-211`.
Severity medium, confidence certain (gap) / possible (trigger).

A non-monotone `ia` whose endpoints line up produces empty/skipped
column ranges: entries silently dropped, the wrong matrix factored,
`FERAL_SUCCESS` returned. (Negative i32 entries sign-extend to huge
usize and are caught by validate or a caught index panic → FERAL_FATAL,
so no UB — the monotonicity hole is the silent one.)

### O1. AMF fill-score i32 overflow for n ≳ 46k

`crates/feral-ordering-core/src/quotient_graph/algo.rs:948` (also
`:969, :1020`). Severity medium-high, confidence certain (arithmetic).

`ws.wf[e] = dext * (2*degree[e] - dext - 1)` with both factors O(n)
exceeds i32::MAX for n ≳ 46k and wraps silently in release; the RMF
`rmf = ... - ws.wf[i] as f64` then drives pivot selection with garbage.
The bucket clamp prevents OOB, so the symptom is pure ordering-quality
degradation (and debug-build panics) on exactly the large KKTs AMF
exists for. MUMPS computes the RMF in DBLE. Fix: compute wf in
f64 or i64.

### O2. KaHIP twin reduction iterates a HashMap → nondeterministic permutations

`crates/feral-kahip/src/data_reduction.rs:425-436, 466-477`. Severity
medium, confidence certain (mechanism); currently masked by the
conservative default preset.

`for (_sig, group) in closed_groups` iterates a RandomState HashMap, so
`ReductionOp::Twin` op-stack order — and therefore the final
permutation — varies run-to-run, violating the crate's documented
determinism contract. Latent today (driver uses
`ReduceOptions::conservative()`, Rule 1 only); a landmine for the
planned Rules 2–4 rollout. Fix: BTreeMap or sort groups.

---

## 3. Medium severity — performance

### N3. Parallel driver ignores `pattern_reused_hint`, `profiler`, and `small_leaf`

`src/numeric/factorize.rs:2795-3059`. Severity medium, confidence
certain.

The parallel driver — the `Solver` default — uses plain
`permute_csc_values` (`:2854`), so the issue-56 Lever A.2 warm-refactor
permute cache never engages on the large matrices it was built for; the
`pattern_reused_hint` doc (`:258-268`) claims "the numeric drivers
consult `permute_cache`" (plural). It also ignores `params.profiler`
(`Solver::with_profiling(true)` returns an empty report on the default
dispatch; the `profile_report` doc at `solver.rs:1651-1659` doesn't
list this case) and `params.small_leaf` (benign today, drift trap).

### N4. MC64 retry (issue #65) has no "tried and not adopted" latch

`src/numeric/solver.rs:1059-1110`. Severity medium, confidence likely.

On adoption the sticky pick pins MC64 (`:1151-1159`); on non-adoption
(genuinely singular matrix — the case the comment anticipates) nothing
is recorded, so every subsequent `factor()` on the same pattern re-pays
a full Hungarian + complete second factorization. The comment's "cost:
one wasted factorization" is actually one per call, indefinitely, and
issue-#43 docs say IPM hosts routinely factor structurally
rank-deficient KKTs. Fix: per-pattern-fingerprint "retry attempted"
flag.

### D7. 32×32 dispatch routes through the `factor_block32` stub

`src/dense/factor.rs:1869-1873` → `src/dense/block_ldlt32.rs:53-69`.
Severity medium (perf), confidence certain.

`factor_block32` is still the Step-1 stub delegating to public
`factor_frontal`, which re-runs `matrix.validate()` (full NaN scan),
allocates and copies an 8 KB scratch, and builds a fresh
`FactorScratch` — bypassing the caller's pooled buffers, inside
`factor_frontal_blocked_in_place_with_scratch` whose whole point
(W-3a / issue #13) is avoiding exactly that. Pure overhead on the
self-described dominant front size for KKT chains.

### N5. Per-call allocation churn (factor + solve paths)

Severity medium (hot path), confidence certain.

- `factorize.rs:2930-2942`: parallel driver builds `num_threads + 1`
  fresh `FactorWorkspace`s per factor call (row_map n×usize, build_seen
  n bools, local_contribs n_snodes options) plus two mutex-wrapped
  stores; the sequential path pools all of this. Telemetry measures the
  cost (`phase_thread_ws_ns`) but nothing amortizes it.
- `solve.rs:62-71`: `Solver::solve`/`solve_sparse` allocates a fresh
  `SolveWorkspace` + result vector per call; no pooled solve workspace
  on `Solver`. `estimate_inverse_norm_1` (`condition.rs:75-143`)
  compounds this ~11×.
- `factorize.rs:3482-3487`: warm permute path clones col_ptr + row_idx
  (O(nnz) memcpy) every warm factor; the structure is immutable per
  pattern.

### X7. `feral_factor` clones the whole matrix every call

`src/capi.rs:266-269`. Severity medium (perf), confidence certain.
O(nnz) alloc + memcpy per IPM iteration (borrow-checker workaround vs
`&mut s.solver`). Restructure to avoid the clone.

### L3. Scaled ftran/btran and refine allocate per call

`src/lu/dense_solve.rs:24,43,110,195`; `src/lu/sparse_solve.rs:23,41,107,234`.
Severity medium (perf), confidence certain. Contradicts the struct's
own "no per-call allocation in solves" claim
(`dense_factor.rs:44-45`). ftran/btran run once or twice per simplex
iteration; with scaling enabled every solve allocates and zeroes a
fresh vector. A second scratch buffer on the struct fixes it.

### L4. `ata_pattern` is O(m²) on a single dense row

`src/lu/sparse_matrix.rs:201-240`. Severity medium
(scalability), confidence certain (blow-up) / likely (impact).

Builds explicit AᵀA adjacency; one dense row (LP budget/convexity
constraint) makes every column pair adjacent — O(m²) time and memory in
`SparseLuSymbolic::analyze` before AMD runs. COLAMD drops dense rows for
exactly this reason; add a dense-row guard.

### L5. Growth monitor tracks only the max multiplier

`src/lu/dense_update.rs:95`; `src/lu/sparse_update.rs:305`. Severity
medium, confidence likely (design intent may differ from doc).

`max_growth` (doc: "growth monitor") records the largest single
elimination multiplier ever seen. A chain of updates with multipliers
~100 each compounds element growth ~100^k while the monitor sits at
100. Combined with L7 (stale per-slot column scaling) accuracy decay
between refactors is effectively unmonitored except by `max_updates`.
Standard FT/BG implementations monitor ‖U‖-type growth.

### L6. Absolute `zero_pivot_tol` (1e-13) with default scaling None

`src/lu/mod.rs:94,99`; used at `dense_factor.rs:230,243`,
`sparse_factor.rs:250`, both update paths. Severity medium, confidence
likely. A well-conditioned basis with entries ~1e-10 is declared
SingularBasis at column 0; one with entries ~1e+10 gets effectively no
singularity detection. Make the tolerance relative to the (scaled)
matrix magnitude or document the interaction.

---

## 4. Lower severity

### Dense (`src/dense/`)

- **D8** `symmetric_row_offdiag_max` doc says position (r,k) is
  excluded; the loop `for j in k..r` includes it. The code matches
  LAPACK dsytf2 (ROWMAX includes A(IMAX,K)) — the **comment** is wrong
  and load-bearing: a future "fix" toward it would corrupt the BK test.
  `factor.rs:4288-4309`. low/certain.
- **D9** Drift among the four duplicated 1×1/2×2 zero-pivot
  implementations: `do_1x1_pivot` PerturbToEps doesn't increment
  `n_tiny` (`factor.rs:4485-4496`) while `try_reject_1x1_frontal`
  (`:3922-3926`) and `count_1x1_inertia` (`:4761-4771`) do; `factor()`
  evaluates the Duff-Reid bound on the unperturbed 2×2 while
  `scalar_pivot_step` perturbs before the gates; `factor()` lacks the
  SSIDS det floor and the issue-#46 partner fallback; stale F-01
  comment at `factor.rs:3891-3897` describes the pre-2026-05-17
  zero-count rule (code sign-counts; the later comment at 3938+ is
  correct); `Factors::needs_refinement` doc (`:761-762`) says "when
  ForceAccept fired" but it is also set by growth flagging,
  PerturbToEps, band pivots, and static pivoting. low/certain.
- **D10** `ncol == 0` early returns (`factor.rs:1412, 1509, 1834,
  2310`) hand back `contrib: matrix.data.clone()` with a
  non-normalized upper triangle — stale garbage for pooled buffers
  (`from_pooled_buf`), unlike the normal extraction path which
  zero-fills (`:1674-1683`). Bit-compares of full contrib buffers (the
  block32 harness does this) see nondeterministic data. low/certain
  (asymmetry) / possible (impact).
- **D11** `equilibrate_scaling` (`equilibrate.rs:15-42`) is O(10·n²)
  with branchy `matrix.get()` per element and stride-n access for the
  j>i half — runs on every `factor()`/`factor_single_front` call.
  low/certain (perf).
- **D12** `flag_growth_for_refinement` adds a full O(nrow·nelim) pass
  over L at every dense-factor exit (`factor.rs:1691, 2171, …`); could
  be fused into the L-extract loop. low/certain (perf).
- **D13** `update_2x2_block32` det==0 early-return leaves L columns
  holding raw A values — unreachable through gated frontal paths, only
  via the D5 legacy path. low/certain.

### Numeric (`src/numeric/`)

- **N6** `solve_sparse_many_into` validates `ws.nrhs`/`ws.n` but not
  `ws.scaled_rhs` vs `scaling_info` (`solve.rs:424-429` vs
  `:364-368`) — a workspace built for unscaled factors used with scaled
  factors of the same shape panics OOB (`:449-456`) in a crate that
  otherwise returns Result. low/certain (code) / possible (practice).
- **N7** Cold permute path rebuilds the value-map cache unconditionally
  (`factorize.rs:3493-3515`): O(nnz·log) binary searches + three O(nnz)
  vectors paid by every one-shot caller and thrown away. Gate on
  `pattern_reused_hint`. low/certain.
- **N8** Empty-supernode early return (`factorize.rs:2138-2175`, twin
  at `:2506-2543`) does not drain `contrib_blocks[child]`; if the
  symbolic layer ever emits an empty supernode with contributing
  children, their Schur data and delayed-pivot inertia are silently
  dropped. Unreachable today; add a drain or
  `debug_assert!(children have no contribs)`. low/possible.
- **N9** Sequential profiler per-supernode phase deltas read
  process-global atomics (`factorize.rs:2020-2047`); two solvers
  factoring concurrently in one process cross-contaminate the deltas.
  Diagnostics only. low/certain.
- **N10** Doc/code drift: `condition.rs:7,27` "3-5 solves" (actually up
  to ~11); `solve.rs:389-391` claims an nrhs==1 thin-wrapper dispatch
  that doesn't exist (bit-identical anyway); `BucketStats.pct_of_total`
  (`factorize.rs:441-447`) is percent of loop_us, not total_us; the
  Schur inner driver (`factorize.rs:1645`) uses `compute_scaling`, not
  `compute_scaling_with_cache` — MC64 cache reuse silently doesn't
  apply on the Schur path. low/certain.
- **N11** Solve-time null-pivot semantics: a force-accepted zero pivot
  is skipped, leaving the forward-substituted RHS value in `w[k]`
  (`solve.rs:266-271`, `:770-775`) rather than zeroing the null-space
  component (MUMPS ICNTL(25)=0 convention). Documented as deliberate
  (`dev/plans/threshold-mismatch-fix.md`); flagged for awareness — the
  unrefined `Solver::solve()` exposes it directly. low/certain
  (behavior) / possible (that it's wrong).

### Sparse / symbolic / ordering (`src/`)

- **S2** Placeholder `Supernode.nrow = col_counts[first_col].max(ncol)`
  (`symbolic/supernode.rs:399-405`) undercounts amalgamated frontals
  (size-based merges need not nest). Downstream bias:
  `contrib_sizes`/`peak_contrib_bytes` (`mod.rs:963-965`) underestimate
  pool needs; `factor_nnz_estimate` (`mod.rs:935`) excludes
  amalgamation-induced zeros and AutoRace (`mod.rs:640`) ranks on the
  biased metric; `find_small_leaf_groups` (`small_leaf.rs:138-140`)
  gates on placeholder nrow ≤ 16 while sizing the arena on actual
  rows.len(). Numeric correctness unaffected (`build_row_indices`
  recomputes). medium/likely.
- **S3** `compute_leaf_rows` (`small_leaf.rs:208-217`) lacks the
  defensive `r < own_last` filter its numeric twin
  (`factorize.rs:3632-3645`) applies; if the leaf invariant ever
  cracks, the batched path diverges silently from the per-supernode
  path. One-line guard. medium-low/possible.
- **S4** `schur.rs::run_amd` (`ordering/schur.rs:186-214`) skips the
  perm-length check `run_external_ordering` (`symbolic/mod.rs:591-597`)
  performs; only a debug_assert stands before `permute_pattern`
  corruption in release. Neither path checks bijectivity.
  medium-low/certain (drift) / possible (impact).
- **S5** Reference `column_counts` (`symbolic/column_counts.rs:56-65`)
  is O(n³)-class (`contains` + re-sort per eliminated column) but
  documented "O(n²) elimination simulation" (`mod.rs:855-857`) and
  publicly re-exported. Production uses GNP everywhere; burdens tests.
  low/certain.
- **S6** Unchecked documented invariants: no `CscPattern::validate()`
  (etree wants upper-triangle entries, GNP wants lower — both silently
  require full-symmetric; a lower-only pattern yields an edgeless
  forest with counts of 1 and no error); `subtree_sizes` assumes
  `parent[j] > j` (comment only, fields pub);
  `schur_constrained_postorder` correctness leans on parent>child
  within the Schur set, guarded only by a debug_assert tail check
  (`symbolic/mod.rs:1092-1098`). low/certain.
- **S7** AutoRace passes the same profiler Arc to all four candidates
  (`symbolic/mod.rs:629-656`): 4× stages against the last candidate's
  total, guaranteeing the "stage sum exceeds total" warning and
  misleading attribution. low/likely.
- **S8** Allocation-in-loop cluster: per-column Vec in
  `sort_and_sum_duplicates` (`csc.rs:121`); provably redundant final
  per-column sort in `symmetric_pattern` (`csc.rs:242-246`);
  `children()` builds Vec<Vec> per postorder variant and in GNP which
  only needs a leaf flag (`elimination_tree.rs:66-74`);
  `compress_pattern` builds ~n two-element Vecs
  (`ldlt_compress.rs:161-166`); HashSet for a contiguous range test
  (`mod.rs:1331`). low/certain.
- **S9** `symv` validates nothing (`csc.rs:258`): panics on short x/y,
  silently zero-fills only the first n of an oversized y — inconsistent
  with the scrupulous from_triplets/validate on the same type.
  low/certain.
- **S10** `predict_merges` vs `find_supernodes` drift
  (`supernode.rs:628-671`): prediction ignores the Phase-B4 root cap
  and uses original parent ncols vs the real pass's cumulative ones.
  Harmless heuristic bias, but the root-cap omission is undocumented
  and the MUONSINE over-merge analysis suggests it's load-bearing.
  low/certain.

### LU (`src/lu/`)

- **L7** Stale column scaling on entering columns
  (`dense_update.rs:55-57`, `sparse_update.rs:174`): entering column
  scaled by `d_col[leaving_slot]` computed for the old column.
  Algebraically consistent but equilibration quality decays arbitrarily
  over an update chain, inflating bump multipliers (interacts with L5).
  low/certain (mechanism) / possible (severity).
- **L8** `DenseLu::update` doc claims `SingularBasis` on a vanishing
  bump pivot (`dense_update.rs:30`) but the code returns
  `NeedsRefactor` (`:87-89`); the sparse update does return
  `SingularBasis` — drivers get different signals from the two paths
  for the same event. low/certain.
- **L9** `SingularBasis { column }` reports internal pivot/factorization
  positions (`sparse_factor.rs:253`, `sparse_update.rs:100,293`), not
  basis slots; the error doc (`error.rs:49-51`) implies the caller can
  repair the basis, but callers know slots, not AMD-dependent pivot
  positions. low/certain (value) / likely (usability).
- **L10** Diagonal-first `u_rows` invariant enforced only by
  debug_assert (`sparse_solve.rs:168,190`); a future violation makes
  release-mode usolve treat an off-diagonal as the pivot silently.
  Hardening. low/certain (guard level) / possible (firing).
- **L11** `DenseLu::perm_inv` is dead state — built and maintained
  (`dense_factor.rs:31,59-61,94-96`), never read. low/certain.
- **L12** Allocation cluster: `u_entries` Vec per column in the factor
  loop (`sparse_factor.rs:215`); two Vecs per column in
  `remap_and_sort_l` (`:490-491`); `pivot_data` clone + per-op row
  allocation in bump elimination (`sparse_update.rs:299,309`); full
  L/U/qcol clones for rollback on every dense update
  (`dense_update.rs:66-68`) — an undo log of touched entries would be
  far cheaper on the success path. low/certain.
- **L13** Sparse perturbation branch picks the first unpivoted row by
  O(m) linear scan (`sparse_factor.rs:257-259`) — O(m²) on heavily
  rank-deficient bases, and index-first rather than largest-|w|
  (dense path perturbs the threshold-selected row — another drift).
  low/certain.

### Scaling / IO / capi / bench / misc (`src/`)

- **X8** Status-code doc drift: `capi.rs:13-14` claims codes "mirror
  Ipopt's ESymSolverStatus", but FERAL_FATAL = 3 collides with
  SYMSOLVER_CALL_AGAIN; a numerically pass-through shim would turn
  fatal errors into call-again loops. low/likely.
- **X9** `feral_set_structure` doesn't invalidate `solver.last_factors`
  (`capi.rs:213-215`): a protocol-violating set_structure → solve gets
  the old factor refined against the new matrix, FERAL_SUCCESS if
  dimensions match. low/certain (gap) / possible (impact).
- **X10** Unbounded `Vec::with_capacity(nnz)` from an untrusted MTX
  header (`mtx.rs:114`): a corrupt nnz triggers a multi-exabyte
  allocation request → abort, not Err. Clamp to file size.
  low-medium/certain.
- **X11** MTX header strictness: exact single-space lowercase banner
  match rejects legal multi-space/tab banners (`mtx.rs:48-54`);
  `%` comments after the size line error as data (`:115-128`);
  `parse::<f64>` accepts NaN/inf silently (bench filters; library
  callers don't). low/certain.
- **X12** `build_cost_graph` per-column sort is dead work with O(n)
  per-run allocations (`mc64.rs:342-351`): the two-pass expansion
  already emits rows ascending. MC64 is documented as the dominant
  symbolic cost. low/likely (ordering argument) / certain
  (allocations).
- **X13** `value_bound.rs:180-189,222-226`: the defensive-fingerprint
  comment ("makes the subsequent check reject") is false — with
  mean_diag_0 = 0, condition 3 is vacuous; impact nil today only
  because the length gate rejects first. low/certain.
- **X14** `factor_nnz` for the dense bench path is n² with a comment
  claiming strictly-lower-triangle count (`bench.rs:1642-1645`); the
  code matches the multifrontal nrow·nelim convention, so the comment
  is what's wrong — but fill-parity readers need to know which.
  low/certain.
- **X15** `src/bin/bench.rs` escapes the no-unwrap lint (lib.rs lint
  covers the lib crate only): `.unwrap()/.expect()` at bench.rs:606,
  1192, 1599, 1623, 1627, 1910-1916. The resample `.expect`s are a real
  panic path that would lose a multi-hour corpus run. low/certain.
- **X16** `Inertia::new` (`inertia.rs:12-18`) doesn't enforce the
  documented `pos+neg+zero == n` invariant; doc reads like a guarantee.
  low/certain.

### Ordering crates (`crates/`)

- **O3** `CscPattern::new` in feral-ordering-core documents but never
  enforces sorted rows (`lib.rs:50-52` vs `:78-112`); downstream
  silently depends on it (metis adjacent-dup dedup, scotch
  partition_point insert, metis/scotch diagonal splice). Check (O(nnz))
  or debug-assert. medium-low/certain (unenforced) / possible (impact).
- **O4** Non-aggressive Pass-2 casts a possibly-stale mark difference
  straight to usize with no guard (`algo.rs:407`, AMF branch
  `:962-977`); the invariant holds today, but a regression wraps to
  ~2⁶⁴. Add `debug_assert!(we >= ws.wflg)`. low/possible.
- **O5** Dense-threshold formula deviates from its own doc for small n
  / negative alpha (`workspace.rs:204-210`): `.max(16)` overrides the
  documented n−2 for n < 18; `min(max(16,x),n)` ≠ documented
  `max(16,min(n,x))`. No practical effect. low/certain.
- **O6** `n_clear_flag` stat hard-coded 0 in both feral-amd
  (`lib.rs:115`) and feral-amf (`lib.rs:121`) while the stats docs
  claim "every field is populated" in debug builds. low/certain.
- **O7** metis "SHEM" is plain HEM (`coarsen.rs:59-98`): no
  ascending-degree visitation (METIS Match_SHEM bucket-sorts after the
  shuffle); high-degree vertices left unmatched → worse coarse graphs
  on the irregular/KKT inputs the header worries about.
  medium/certain (deviation) / likely (impact).
- **O8** metis stall handling contradicts its comment
  (`coarsen.rs:130-137`): a stalled level is pushed whenever prior
  levels exist even with zero shrinkage; a first-level 4%-shrink stall
  is discarded. The test `coarsen_hierarchy_shrinks_monotonically`
  passes by luck of its inputs. low/certain.
- **O9** metis `two_hop_pass` is O(n²) on hub graphs
  (`coarsen.rs:148-204`) — its own motivating case; rescans
  neighbor-of-neighbor lists from the start per spoke; `mark` allocated
  and never used. medium-low/certain (perf).
- **O10** metis GGP gain is not the GGP gain
  (`initial_partition.rs:63-97`): only ever adds edges-to-A
  (`adding_to_a` always true; the documented subtraction doesn't
  exist) — selection maximizes connectivity-to-A, biasing toward
  high-degree vertices. FM papers over some of it.
  medium-low/certain.
- **O11** metis FM heap seeded with all n vertices each pass
  (`fm_refine.rs:56-61`): Ω(n log n) per pass × 10 passes × every
  level, boundary-only seeding is the standard. Acknowledged trade;
  flagged for cost. low/certain.
- **O12** metis doc drift: `lib.rs:219` still cites HSL_MC68/ICNTL(6)/
  SSIDS for the dense-quotient path while the `MetisOptions` doc
  (`lib.rs:105-121`) explains that belief was audited and found wrong.
  low/certain.
- **O13** scotch vertex-separator FM: imbalance-rejected heads are
  popped, not locked (comment claims a lock, `vertex_separator.rs:319-327`),
  and each outer iteration tries only the post-drain head of each PQ —
  if both heads are imbalance-rejected the pass breaks with feasible
  lower-gain moves still queued (`:464-466`). SCOTCH skips and
  continues. Separators stop improving early under tight
  max_imbalance. medium/certain (behavior) / likely (impact).
- **O14** scotch band FM is dead code while `lib.rs:12-13` advertises
  it; `band_fm.rs` is `#[allow(dead_code)]`, unreachable from
  `node_nd.rs`; projection-loop variable names swapped
  (`band_fm.rs:76-81` — functionally correct, editor trap).
  low/certain.
- **O15** scotch AMD leaves on the compressed graph ignore
  supervariable weights (`node_nd.rs:54-62`, `amd_leaf:281-302`):
  weight-7 supervariables treated as unit vertices, skewing
  degree-based pivots on heavily compressed inputs. Expansion still a
  valid permutation. medium-low/certain (code) / possible (impact).
- **O16** kahip flow refinement balances by vertex count on weighted
  coarse graphs (`flow_refine.rs:226-233`; `cycle.rs:174-201` drops
  vwgt). In Eco/Strong, do_flow runs at every uncoarsening level where
  coarse weights ≫ 1; a count-balanced min-cut can be badly
  weight-imbalanced and the finer level's FM starts from a violating
  state. medium/certain.
- **O17** kahip `apply_degree2` rescans from vertex 0 per chain
  (`data_reduction.rs:307-309`): O(n²) worst case on long paths. Off by
  default (Rule 1 only); will bite when Rule 2 is enabled.
  low-medium/certain.
- **O18** kahip stats drift: `stats.cycles` increments once per
  `multilevel_bisection` call, not per V-cycle as documented
  (`cycle.rs:57`); `n_components` computed and discarded
  (`node_nd.rs:91-103`) with no stats field, unlike metis/scotch.
  low/certain.
- **O19** kahip `flow.rs:271-279` stranded-vertex branch would corrupt
  the gap histogram (sets height without decrementing height_count) —
  unreachable today (excess implies a residual reverse edge); add a
  debug_assert or comment. low/certain (code) / dead (impact).
- **O20** Thrice-copied ND driver scaffolding
  (recurse/connected_components/extract_by_*/build_induced/
  graph_to_csc_pattern/invert_iperm) across the metis/scotch/kahip
  `node_nd.rs`, already drifted: kahip sorts neighbors before the
  diagonal splice, metis/scotch rely on induced sortedness; metis
  validates the inverse perm inline, scotch/kahip have a duplicate-
  position check metis lacks. Consolidate into feral-ordering-core.
  medium-low/certain.
- **O21** AMD/AMF inner-loop duplication (documented decision, ~600
  LoC) is already asymmetric: AMF writes `wf = 0` for dead elements,
  indistinguishable from the first-touch sentinel 0, so a live element
  with true contribution 0 is recomputed every iteration
  (`algo.rs:968`). Harmless; illustrates the drift risk. low/certain.

---

## 5. Cross-cutting themes and structural recommendations

The dominant failure class in this codebase is **sibling-path drift**:

| Sibling pair | Drift found |
|---|---|
| legacy `factor()` vs frontal kernels | D1, D5, D9 (gates and accounting backported to frontal only) |
| dense LU vs sparse LU | L1 (guards), L2 (threshold), L8 (error types), L13 (perturbation row) |
| sequential vs parallel multifrontal driver | N3 (three options silently ignored) |
| scalar vs panel BK pivot step | D2 (static floor), D9 (perturb-before-gates ordering) |
| factor-side vs solve-side 2×2 handling | D4 (det formula + floor) |
| `to_dense` vs `to_csc` MTX paths | X2 (duplicate semantics) |
| three ND drivers in the partitioner crates | O20 |
| four copies of zero-pivot accounting | D3, D9 |

**Recommendation 1 — retire or backport the legacy dense `factor()`
pipeline.** D1, D5, and most of D9 live there while the frontal kernels
have the guards, and the path remains production-reachable via
`SchurBlock::solve_with` and the crate-root `factor` re-export. Either
route it through `factor_single_front`-equivalent gates or consolidate
the four pivot-accounting implementations into one.

**Recommendation 2 — add option-parity gates, not just result-parity
gates.** Existing parity tests (`tests/parallel_parity.rs` etc.) compare
outputs, which is exactly how the FMA no-op (N1), the ignored permute
cache/profiler (N3), and the ignored sparse-LU `pivot_threshold` (L2)
survived: ignoring an option produces "matching" results. A test that
asserts each documented option observably changes behavior (or a
shared options-plumbing struct both drivers consume) closes the class.

**Recommendation 3 — promote documented invariants to checked
invariants** at the trust boundaries: `CscMatrix::validate` col_ptr
monotonicity (X6), a `CscPattern::validate` (S6/O3), perm length +
bijectivity after external orderings (S4), MTX nnz-vs-entries (X2),
solve-workspace scaling compatibility (N6). All are O(n) or O(nnz)
one-time checks guarding silent-wrong-answer paths in an
inertia-certified solver.

## 6. Suggested triage order

1. D1 (silent inertia corruption), L1 (silent Inf/NaN), N1 (dead FMA),
   S1 (quadratic default postorder) — all small, surgical fixes.
2. The bit-parity / accounting items while context is fresh: D2, D3,
   D4, D5, N2.
3. Harness-integrity items so measurements can be trusted: X3 (bench
   dense params), X5 (env-var drift), X2 (MTX nnz check).
4. Hot-path performance: N3, N4, D7, N5, X7, L3.
5. Hardening and the low-severity backlog as opportunistic fixes,
   each citing this document's finding ID.

Findings here were produced by static review; none have been confirmed
with a failing test yet. Before fixing any individual item, write the
reproducing test first (per the spec's tests-first lifecycle), and
treat a finding that cannot be reproduced as a candidate for
`dev/tried-and-rejected.md` with a note referencing this document.
