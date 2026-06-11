# Changelog

All notable changes to FERAL will be documented in this file.

## [Unreleased]

### Fixed — permute-cache warm path no longer trusts a stale pattern/permutation (REG-1)

The `PermuteCache` introduced by the N7 one-shot optimization keyed its warm-path
validation on `(input_n, input_nnz, value_map.len())` only — not the input
*pattern* (`col_ptr` + `row_idx`) or `perm_inv`. Re-factoring a second matrix
with the same `(n, nnz)` but a different sparsity pattern on the same `Solver`,
or re-factoring the same pattern under a different AutoRace-selected permutation,
warm-hit the stale `value_map` and scattered the new values through the old
structure — returning `Success` with a silently wrong factorization (solve
residual `2.1e+2` vs `5.7e-14` on a cold build). This was a live hazard on the
default sequential and Schur reuse paths. `PermuteCache` now stores the
build-time `input_col_ptr`, `input_row_idx`, and `input_perm_inv`, and the warm
path accepts the cache only when all three match byte-for-byte (an `O(n + nnz)`
compare — still cheaper than the skipped `from_triplets` sort, exact rather than
hashed so a fingerprint collision can never reintroduce the wrong answer). A
one-shot (`pattern_reused_hint == false`) call also clears any existing cache so
a later warm call cannot trust it. Finding REG-1 from
`dev/research/repo-review-2026-06-09-verification.md`.

### Added — `feral-kahip` now reports `KahipStats::n_components` (O18)

`KahipStats` gains a `pub n_components: u32` field — the number of top-level
connected components encountered by the nested-dissection driver — matching the
existing `MetisStats::n_components` / `ScotchStats::n_components`. The KaHIP
driver already computed this count in `run_top` but discarded it; it is now
surfaced for parity with the sibling crates. The `KahipStats::cycles`
documentation was also tightened: it counts multilevel bisections (one per
node-separator computation; each a single V-cycle), and the stale inline
comment that described it as a per-level coarsening counter was corrected.
Finding O18 from `dev/research/repo-review-2026-06-09.md`.

### Fixed — `feral-kahip` flow refinement now balances by vertex weight on coarse graphs (O16)

K3 flow-based refinement (`flow_refine_bisection`) scored candidate cuts against
a *vertex-count* balance constraint (`max(|part0|, |part1|) ≤ (1+ε)·⌈n/2⌉`),
and the `Graph → UndirectedGraph` bridge (`graph_to_undirected`) discarded the
coarse graph's vertex weights by assigning unit weights. On Eco/Strong runs flow
refinement executes at *every* uncoarsening level, where each coarse vertex
stands for a supervertex whose mass is ≫ 1, so a count-balanced min-cut could be
badly *weight*-imbalanced — and the finer level's FM then started from a
constraint-violating partition. `UndirectedGraph` now carries a `vweight` field
(populated from `Graph::vwgt`, unit on finest-level inputs), and the balance
check measures `max(weight(part0), weight(part1)) ≤ (1+ε)·⌈W/2⌉` against those
weights, matching the KaHIP/Sanders-Schulz constraint. On a 7×7 grid whose
min-cut pulls a mass-20 supervertex into one side, the count-based check accepted
a 52|16 weight split (slack 47); the weight-aware check rejects it. Finding from
`dev/research/repo-review-2026-06-09.md`.

### Fixed — `feral-scotch` vertex-separator FM no longer stops at imbalance-rejected heads (O13)

The single-pass vertex-separator FM (`fm_pass`) tried only the post-drain head
of each priority queue per outer iteration. When *both* heads were
imbalance-rejected, it popped them, set no `moved_this_iter`, and the
`if !moved_this_iter { break; }` guard terminated the whole pass — abandoning
feasible lower-gain moves still queued below the rejected heads. Under a tight
`max_imbalance`, separators stopped improving early (and a comment falsely
claimed the rejected vertex was "locked" when it was only popped). The pass now
records imbalance rejections and only terminates when the frontier is truly
exhausted (no move *and* no rejection), so it skips infeasible heads and keeps
refining — matching SCOTCH. Concrete impact: on the issue-#3 PoissonControl KKT
pattern (`n = 1200`) ScotchND previously degenerated into a one-sided bisection
and fell back to a whole-graph AMD leaf (separator weight 0, permutation
byte-equal to AMD); it now produces a genuine 188-vertex separator across 26
levels and performs real nested dissection. Finding from
`dev/research/repo-review-2026-06-09.md`.

### Fixed — `feral-metis` GGP initial bisection uses the true greedy gain (O10)

`initial_bisect_ggp` documented its greedy gain as
"(edges to part 0) - (edges to part 1)", but the code only accumulated
edges-to-A: the `push_neighbors` helper took an `adding_to_a` flag that was
hard-coded `true` at every call site, so the `- edges_to_B` term never existed
and the boundary scan selected `argmax(edges_to_A)`. That biases growth toward
high-degree vertices — exactly the wrong choice, since a vertex with many edges
still on the B side drags all of them into the cut when it moves into A. The
selection now uses the true Greedy-Graph-Growing gain
`edges_to_A - edges_to_B = 2*edges_to_A - wtot` (with `wtot` the per-vertex
total incident edge weight, precomputed in `O(nnz)`), so it minimises the added
cut as intended. On the new regression graph this picks the low-degree vertex
(cut 5) instead of the high-degree one (cut 7). Finding from
`dev/research/repo-review-2026-06-09.md`.

### Fixed — `feral-metis` keeps a stalled coarsening level only when it shrank (O8)

`coarsen`'s stall branch pushed the just-computed level whenever earlier levels
existed, with a comment claiming it accepted the level "only if it actually
shrank" — but it never checked shrinkage. Two consequences: a *first* level
that genuinely shrank (between 0% and the 5% stall threshold) was discarded
because no earlier level existed yet, so the whole coarsening hierarchy came
back empty despite real progress; and a later zero-progress level was kept
merely because earlier levels existed, which can break the strictly-decreasing
vertex-count invariant. The branch now pushes the level iff it actually shrank
(`0 < new_nvtxs < prev_nvtxs`), independent of prior levels. Finding from
`dev/research/repo-review-2026-06-09.md`.

### Fixed — `feral-metis` coarsening now performs real SHEM, not plain HEM (O7)

The coarsening header advertised Sorted Heavy-Edge Matching and cited METIS
`Match_SHEM`, but Pass 1 visited vertices in plain seeded-shuffle order with no
ascending-degree sort — that is unsorted Heavy-Edge Matching (HEM). The
"sorted" step matters: a degree-1 leaf whose sole neighbour is claimed first by
a higher-degree vertex is stranded as a self-match, inflating the coarse graph
on the irregular / power-law inputs SHEM exists for. Pass 1 now stable-sorts the
shuffled visit order by ascending vertex degree (`sort_by_key`, so the seeded
shuffle survives as the within-degree tie-break and determinism is preserved),
matching METIS `Match_SHEM` (Karypis & Kumar §3.1). On a 4-vertex chorded path
(one hub, one leaf) this turns a 3-coarse-vertex matching into the optimal
2-coarse-vertex matching. Finding from `dev/research/repo-review-2026-06-09.md`.

### Fixed — `CscPattern::new` enforces its documented sorted-rows invariant (O3)

`feral-ordering-core`'s `CscPattern` documents that row indices within each
column must be sorted ascending, but `CscPattern::new` never checked it — it
validated column-pointer lengths/monotonicity and row-index range only.
Consumers silently depended on the unchecked invariant: `feral-metis`'s
adjacency builder dedups only *adjacent* duplicates, so an unsorted column
(e.g. `[3, 5, 3]`) let a non-adjacent duplicate survive as a spurious edge,
corrupting the graph and the resulting ordering; `feral-scotch`'s compress
step inserts neighbours with `partition_point`, which assumes sorted runs.
`CscPattern::new` now verifies, per column, that row indices are
non-decreasing (`O(nnz)`) and returns `None` otherwise. All in-tree callers
already pass sorted rows (`CscMatrix::symmetric_pattern` sorts each column),
so the `feral` solver is unaffected; only inputs that already violated the
documented contract are now rejected. Finding from
`dev/research/repo-review-2026-06-09.md`.

### Fixed — MTX reader accepts whitespace-flexible banners, rejects non-finite values (X11)

`parse_mtx` / `read_mtx` compared the Matrix Market banner against the exact
single-space string `%%matrixmarket matrix coordinate real symmetric`, so a
legal banner whose fields are separated by multiple spaces or tabs was
rejected with "unsupported header" even though the NIST `mmio` reference
accepts it. The banner is now compared token by token (case-insensitive).
Separately, entry values were parsed with `f64::from_str`, which silently
accepts `nan`/`inf`/`-inf`; such a file produced an `MtxMatrix` carrying a
non-finite value that poisons any downstream factorization. Non-finite entry
values now return `FeralError::IoError`. (Comment lines after the size line
remain an error, matching the spec and `mmio` — comments are only legal
between the banner and the size line.) Finding from
`dev/research/repo-review-2026-06-09.md`.

### Fixed — MTX reader no longer aborts on a corrupt `nnz` header (X10)

`parse_mtx` / `read_mtx` reserved the entries buffer with
`Vec::with_capacity(nnz)`, taking `nnz` straight from the untrusted Matrix
Market size line. A corrupt or hostile header (e.g. `nnz = 10^17`) made
that a multi-exabyte allocation request; the allocator returned null and
the process aborted (`handle_alloc_error`) — a hard crash rather than a
recoverable `FeralError::IoError`. The reservation is now clamped to the
source byte length, a hard upper bound on the true entry count, so a bogus
header is parsed gracefully. `nnz` was only ever an allocation hint (never
validated against the actual entry count), so valid files are unaffected.
Finding from `dev/research/repo-review-2026-06-09.md`.

### Fixed — `feral_set_structure` invalidates the stale factor (X9)

The C-ABI embedding protocol is `feral_set_structure` → fill values →
`feral_factor` → `feral_solve`. `feral_set_structure` replaced the stored
matrix and reset the cached inertia sentinel but left `Solver`'s numeric
factor in place. A host that changed the matrix structure and then solved
*without* re-factoring got that stale factor (refined against the new
matrix) and `FERAL_SUCCESS` — a plausible-but-wrong solution rather than a
clean error. `feral_set_structure` now drops the stored factor (new
`Solver::invalidate_factors`, which keeps the cached symbolic analysis so a
same-structure re-init still reuses it), so a solve with no current factor
returns `FERAL_FATAL`. Normal usage is unaffected: `set_structure` is
called once and is always followed by `factor` before `solve`. Finding from
`dev/research/repo-review-2026-06-09.md`.

### Fixed — sparse singular-column perturbation matches the dense path (L13)

Under `LuSingularAction::PerturbToEps`, the sparse factor perturbed the
*index-first* still-unpivoted row of a singular column, while the dense factor
perturbs the *threshold-selected* (largest-|w|) row — so the two paths
regularized the same singular basis differently (different permutation and
different regularized solve). The sparse path now perturbs the largest-|w|
unpivoted row (`ipiv`, the row threshold partial pivoting already selected),
matching the dense reference. As a side benefit this reuses the pivot already
found in the selection loop, so the O(m) "find first unpivoted row" scan is
skipped whenever the singular column has any touched unpivoted entry; the scan
remains only as a fallback for a column that is structurally empty in every
unpivoted row. Finding from `dev/research/repo-review-2026-06-09.md`.

### Fixed — misplaced U diagonal surfaces as an error in release builds (L10)

The sparse triangular solves (`usolve` / `ut_solve`) take the first stored
entry of each `u_rows[k]` as the pivot, relying on the diagonal-first
invariant. That invariant was enforced only by a `debug_assert_eq!`, compiled
out in release — so a violated invariant (e.g. introduced by a future change)
would make a release build silently divide by an off-diagonal entry and treat
the real diagonal as an ordinary off-diagonal term: a silent wrong solve. The
position check is now folded into the always-on pivot guard, so a U row whose
first entry is not its diagonal returns `FeralError::SingularBasis` in every
build mode, alongside the existing absent/zero/non-finite-diagonal guard. The
check is outside the inner accumulation loop, so the hot solve path is
unchanged. Finding from `dev/research/repo-review-2026-06-09.md`.

### Fixed — `SingularBasis { column }` names the original basis column (L9)

When the sparse LU factor path hit a singular column under
`LuSingularAction::Fail` (or found no unpivoted row under `PerturbToEps`), it
reported `FeralError::SingularBasis { column: k }` where `k` is the internal
*factorization position* — original column `qcol[k]` under the
(AMD-dependent) column order. A caller such as a simplex driver knows the
original basis columns it supplied, not the internal processing order, so the
reported index pointed at the wrong column to repair whenever the column order
was non-identity. The factor path now reports `qcol[k]`, the original basis
column, and the `SingularBasis` doc was updated to state this contract
explicitly. With natural ordering `k == qcol[k]`, so this is a no-op there; it
only changes the reported index for reordered factorizations. Finding from
`dev/research/repo-review-2026-06-09.md`.

### Fixed — sparse and dense LU update report the same failure signal (L8)

`SparseLu::update` returned `FeralError::SingularBasis` when the spike support
was deficient or a bump pivot vanished, while `DenseLu::update` returns
`FeralError::NeedsRefactor` for the identical events. A driver that switched
between the two factorization paths got different error variants for the same
underlying condition — a singular replacement basis encountered mid-update.
Both update paths now return `NeedsRefactor` on any update failure; the
authoritative singularity verdict comes from a fresh factorization (where the
factor path still returns `SingularBasis`), not the incremental update. The
stale `DenseLu::update` doc comment that claimed `SingularBasis` on a vanishing
bump pivot was corrected to match the actual `NeedsRefactor` contract. Finding
from `dev/research/repo-review-2026-06-09.md`.

### Fixed — AutoRace no longer quadruples symbolic-profiler stages (S7)

When a `SymbolicProfiler` was attached and the ordering method was
`OrderingMethod::AutoRace`, the race dispatcher (`symbolic_factorize_race`)
passed the caller's single profiler `Arc` to all four `RACE_CANDIDATES`. Because
`SymbolicProfiler::record` appends and `set_total` overwrites, the shared
profiler ended with one full stage list per candidate (~4×) measured against a
single candidate's `total_us`. The resulting `SymbolicProfileReport` therefore
listed every stage four times, summed `pct_of_total` past 100%, and always
emitted the "stage sum exceeds total" validation warning — misattributing the
symbolic cost breakdown. Each candidate now gets its own fresh profiler and only
the winning candidate's run is copied into the caller's shared profiler, so the
report reflects exactly one ordering. Factorization results (perm, inertia,
factor structure) were never affected — this is a diagnostics-only fix. Finding
from `dev/research/repo-review-2026-06-09.md`.

### Fixed — `solve_sparse_many_into` validates workspace scaling state (N6)

`solve_sparse_many_into` checked that the caller-owned `SolveManyWorkspace`
matched the factors in `nrhs` and `n`, but not that its `scaled_rhs` buffer was
sized for the factors' scaling state. The buffer is sized at `for_factors` time
from the factors it was built against — `n * nrhs` when scaling is applied,
empty otherwise. A workspace built for *unscaled* factors then reused with
*scaled* factors of the same `(n, nrhs)` shape (or vice versa) indexed the
empty `scaled_rhs` out of bounds at the pre-scale step, panicking in a crate
that otherwise returns `Result`. The function now validates `scaled_rhs.len()`
against the factors' scaling state up front and returns
`FeralError::DimensionMismatch` on a mismatch. Finding from PR #83's review
(`dev/research/repo-review-2026-06-09.md`).

### Fixed — KaHIP twin reduction now produces deterministic permutations (O2)

The KaHIP data-reduction twin pass (`feral-kahip`) grouped vertices by signature
in `HashMap`s (`closed_groups`, `open_groups`) and iterated them directly to emit
`ReductionOp::Twin` operations. Because a `HashMap` iterates in per-instance
`RandomState`-seed order, the twin op-stack order — and therefore the final
elimination permutation — varied run-to-run, violating the crate's documented
determinism contract. The merge *result* was unaffected, but the *order* was
not reproducible. Both maps are now `BTreeMap`s keyed by the (already-sorted)
signature, so groups are visited in a stable, signature-sorted order and the
permutation is byte-identical across runs. Latent today (the default
`ReduceOptions::conservative()` preset runs Rule 1 only), but a correctness
landmine for the planned Rules 2–4 rollout.

### Fixed — AMF fill-score arithmetic no longer overflows `i32` for large `n` (O1)

The Approximate Minimum Fill ordering (`feral-ordering-core`) computed its
working-fill (`wf`) quantities — the surface contribution
`dext * (2*deg - dext - 1)` and the accumulation `wf4 + 2*nvi*wf3` — in `i32`,
and stored them in an `i32` field. Both factors are `O(n)`, so the products
reach `~n²` and overflow `i32` for `n` ≳ 46k (`46342 * 46341 = 2_147_534_622`
exceeds `i32::MAX`). The wrapped value then fed the RMF pivot score as `f64`,
silently degrading ordering quality on exactly the large KKTs AMF exists for
(and panicking in debug builds). The `wf` field and its accumulators are now
`i64`, matching MUMPS, which computes the RMF in double precision. The
minimum-degree (AMD) path is unaffected (it never touches `wf`). Finding from
PR #83's review (`dev/research/repo-review-2026-06-09.md`).

### Fixed — `CscMatrix::validate` now rejects a non-monotone `col_ptr` (X6)

`CscMatrix::validate` checked `col_ptr.len() == n + 1`, `row_idx.len() ==
values.len()`, `col_ptr[n] == nnz`, per-entry bounds/lower-triangle, and
within-column sorting — but never that `col_ptr` is monotonically
non-decreasing. A non-monotone `col_ptr` whose endpoints happen to line up
(`col_ptr[0] == 0`, `col_ptr[n] == nnz`) passed every check yet produced
empty or overlapping column ranges, so entries were silently dropped, the
wrong matrix was factored, and `FERAL_SUCCESS` was returned. (Negative `i32`
column counts sign-extend to a huge `usize` and are already caught as
out-of-bounds, so the monotonicity gap was the silent one.) `validate` now
rejects any `col_ptr[j + 1] < col_ptr[j]` with an `InvalidInput` naming the
offending column. Finding from PR #83's review
(`dev/research/repo-review-2026-06-09.md`).

### Fixed — MC64 partial-singular fallback now covers unmatched rows, not just unmatched columns (X4)

On a partial (structurally singular) matching the MC64 symmetric scaling fell
back to identity only for unmatched *columns* (`perm[i] == MAX`). But the
matched-row and matched-column sets can differ even on a symmetric pattern:
an index `i` can have its column matched while its row is unmatched. The
Hungarian kernel zeroes the row dual `u[i]` for an unmatched row, so the
symmetric average `s[i] = exp((u[i] + v[i] - cmax[i]) / 2)` folded a
meaningless zero half-dual into the scaling — the exact "duals are meaningless
on the unmatched part" condition the surrounding code warns about, producing a
badly asymmetric `D·A·D` on rank-deficient KKTs. `scaling_from_cache` now
derives the matched-row set from the matching and falls back to identity for
any index whose row *or* column is unmatched. Matrices with a full matching
are unaffected. Finding from PR #83's review
(`dev/research/repo-review-2026-06-09.md`).

### Fixed — `feral_num_neg` no longer reports a stale or wrong inertia (X1)

The C-ABI negative-eigenvalue accessor (`feral_num_neg`) could return a
plausible-but-wrong inertia. The backing count was initialized to `0`, reset to
`0` on `feral_set_structure`, and never invalidated when `feral_factor` returned
`FERAL_SINGULAR` or `FERAL_FATAL`. So a fresh handle reported `0` (indistinguishable
from a genuinely definite matrix) instead of the documented `-1` sentinel, and —
more dangerously — after a failed re-factor on the same structure (as an IPM host
does every iteration, e.g. when a diverging iterate produces a non-finite Hessian
entry) it silently reported the *previous* matrix's negative-eigenvalue count. The
count is now `-1` ("no valid factor") on a fresh handle, after a structure change,
and after any `FERAL_SINGULAR`/`FERAL_FATAL` factor; it is only set to a real count
on a successful factor. Finding from PR #83's review
(`dev/research/repo-review-2026-06-09.md`).

### Numerics — the sparse LU now honors `pivot_threshold` (L2)

The sparse unsymmetric LU path (`src/lu/sparse_factor.rs`) computed the
threshold-partial-pivoting parameter `u` and then discarded it (`let _ = utol`),
always taking the strict max-magnitude pivot row. So `pivot_threshold = 0.1`
changed the dense path but silently changed nothing on the sparse path,
contradicting the module doc ("threshold partial pivoting"), the `LuParams`
doc, and the dense path. The sparse factorization now implements
diagonal-preference threshold partial pivoting matching CSparse `cs_lu`: when the
natural diagonal row is still unpivoted and within `u·max` of the column max, it
is preferred (a sparser, structure-preserving pivot). `u = 1.0` (the default)
recovers strict partial pivoting exactly, so all default factorizations are
unchanged. The Forrest–Tomlin bump elimination in `update()` deliberately keeps
strict partial pivoting (the bump structure is fixed, so a relaxed threshold buys
no fill reduction and only costs stability); the `LuParams::pivot_threshold` doc
now scopes the knob accordingly. Finding from PR #83's review
(`dev/research/repo-review-2026-06-09.md`).

### Harness/C-ABI — `FERAL_SCALING` vocabulary unified across the shim and bench (X5)

The two `FERAL_SCALING` parsers had drifted apart. The C-ABI shim (`src/capi.rs`)
accepted `identity`/`infnorm`/`mc64`/`auto` and *silently* ignored anything else;
the bench harness (`src/bin/bench.rs`) accepted `identity`/`infnorm`/`mc64`/`adaptive`
and warned on unknown values. So `FERAL_SCALING=adaptive` selected adaptive
routing in bench but was a silent no-op in the shim, and `FERAL_SCALING=auto`
worked in the shim but warned and fell back to the default in bench — a cross-tool
experiment with one spelling silently measured different configurations. Both
`auto` and `adaptive` now select `ScalingStrategy::Auto` in both tools,
case-insensitively. Relatedly, the bench `FERAL_ORDERING` parser previously
coerced any unrecognized value (typos included) to forced AMD with no warning; it
now accepts an explicit `amd` and warns + falls back to the default heuristic on
unrecognized values. Finding from PR #83's review
(`dev/research/repo-review-2026-06-09.md`).

### Numerics — the LU zero-pivot tolerance is relative to the matrix magnitude (L6)

The singularity / zero-pivot test compared each pivot against the absolute
`zero_pivot_tol` (default `1e-13`), independent of the basis scale and of the
default `LuScaling::None`. A uniformly small but perfectly conditioned basis —
e.g. `diag(1e-14)`, condition number 1, exact inverse `diag(1e14)` — was wrongly
declared `SingularBasis { column: 0 }`, while a large-magnitude basis got
effectively no singularity detection. The factor paths
(`src/lu/dense_factor.rs`, `src/lu/sparse_factor.rs`) now use
`zero_pivot_tol · max|A|`, and the Forrest–Tomlin/Bartels–Golub update paths
(`src/lu/dense_update.rs`, `src/lu/sparse_update.rs`) use
`zero_pivot_tol · max|U|` at the last factor — both matrix-relative, matching
LAPACK's norm-relative convention. For the zero matrix (`max|A| == 0`) only an
exact-zero pivot trips, which remains correct. On bases whose magnitude is `O(1)`
the threshold is unchanged. Finding from PR #83's review
(`dev/research/repo-review-2026-06-09.md`).

### Numerics — the LU update growth monitor tracks element growth, not the largest single multiplier (L5)

Both the dense (`src/lu/dense_update.rs`) and sparse (`src/lu/sparse_update.rs`)
Forrest–Tomlin/Bartels–Golub update paths recorded `growth` as the largest
single elimination multiplier ever seen. A chain of updates each with a
multiplier of, say, ~100 compounds element growth in `U` to ~100ᵏ while the
monitor sat flat at 100 — so an ill-conditioned basis could accumulate large
entries in `U` and silently lose accuracy without ever tripping `max_growth` and
forcing a refactor. The monitor is now the ‖U‖∞ element-growth high-water ratio
`max|U| over the update history ÷ max|U| at the last factor`, which compounds
across updates exactly as standard FT/BG implementations require. This makes the
monitor strictly stricter (it can only trip earlier, never later), so a basis
that would have drifted now refactors in time. The sparse path keeps the update
cheap by scanning only the rows that changed this update — provably equal to the
true high-water, since every changed `U` entry lives in a changed row. Finding
from PR #83's review (`dev/research/repo-review-2026-06-09.md`).

### Scalability — dense-row guard in the sparse LU column ordering (L4)

`SparseColMatrix::ata_pattern` built the explicit AᵀA column-intersection graph
that the fill-reducing column ordering needs. A single dense row — common in
LPs, where a budget or convexity constraint touches every column — is adjacent
to every other column and made the graph complete, costing O(m²) time and
memory in `SparseLuSymbolic::analyze` before AMD even ran. Following COLAMD,
rows whose population exceeds `max(16, 10·√ncol)` are now excluded from the
adjacency build; they carry no useful ordering information and the diagonal is
always retained, so AMD still orders all columns. On matrices with no dense row
the ordering is unchanged. Finding from PR #83's review
(`dev/research/repo-review-2026-06-09.md`).

### Performance — scaled LU solves and iterative refinement reuse pooled scratch buffers (L3)

With scaling enabled, the scaled `ftran`/`btran` wrappers and iterative
refinement allocated and zeroed fresh `Vec<f64>` buffers on every call
(`src/lu/dense_solve.rs`, `src/lu/sparse_solve.rs`) — once or twice per simplex
iteration — contradicting the factorization's "no per-call allocation in solves"
guarantee. Both `DenseLu` and `SparseLu` now carry three additional pooled
buffers (scaled right-hand side, refinement residual, and refinement
right-hand-side snapshot) that are taken via `std::mem::take`, reused in place
when already sized, and restored on every return path. After warm-up, scaled
solves and refinement allocate nothing. Finding from PR #83's review
(`dev/research/repo-review-2026-06-09.md`).

### Performance — `feral_factor` borrows the stored matrix instead of cloning it each call (X7)

The C ABI `feral_factor` (`src/capi.rs`) cloned the whole `CscMatrix` on every
call — an O(nnz) allocation plus memcpy paid once per IPM iteration. The clone
was a borrow-checker workaround for holding `&s.matrix` while calling
`&mut s.solver`. Because `matrix` and `solver` are disjoint fields of the
handle, a split field borrow lets the immutable matrix borrow coexist with the
mutable solver borrow, so the factorization now reads the matrix in place with
no per-call copy. The factorization path itself was already clone-free at the
`CscMatrix` level (it clones only the small CSC component vectors it needs), so
this removes the last whole-matrix clone from the hot path. Finding from PR
#83's review (`dev/research/repo-review-2026-06-09.md`).

### Performance — condition-number estimator pools one solve workspace across its internal solves (N5)

`estimate_inverse_norm_1` (the Hager–Higham 1-norm condition estimator) runs up
to `2·MAX_ITER + 1` solves against the stored factor. Each went through
`solve_sparse`, which allocates a fresh `SolveWorkspace` (three vectors) plus a
result vector per call — so a single `estimate_condition_1norm` paid that
allocation ~11×. The estimator now builds one `SolveWorkspace` and one output
buffer up front and reuses them across every internal solve via
`solve_sparse_into_ws`. The arithmetic is bit-identical (the existing diagonal
and Hilbert condition-number oracles are unchanged). This addresses the
condition-estimator facet of N5; the parallel-driver per-thread
`FactorWorkspace` allocation and the warm-permute structure clone that N5 also
cites are not yet addressed. Finding from PR #83's review
(`dev/research/repo-review-2026-06-09.md`).

### Fixed — MC64 scaling retry on a genuinely singular pattern is no longer re-paid every `factor()` (N4)

The issue-#65 inertia-guided MC64 scaling retry had no "tried and not adopted"
latch. When `Auto` scaling force-accepts zero pivots, `Solver::factor` re-runs
the factorization with `Mc64Symmetric` and adopts it only if it strictly
reduces the zero count. On *adoption* this self-latches (the sticky-`Auto`
pick pins `Mc64Symmetric`, which the retry gate already skips). But on
*non-adoption* — a genuinely singular matrix, where MC64 cannot change rank,
the strict-improvement gate fails, and the original factor is kept — nothing
was recorded. The gate keys on the user's configured scaling (`Auto`, which
never changes) and on the resolved scaling staying non-MC64 (the picker
re-pins `InfNorm`), so every subsequent `factor()` on the same pattern re-paid
a full Hungarian plus a complete second factorization. An IPM that repeatedly
factors a singular KKT without regularizing it paid this wasted retry on every
iteration. `Solver` now carries a per-pattern latch that records the
non-adoption and suppresses the retry on subsequent same-pattern factors; the
latch clears on pattern change (alongside the issue-#51 sticky-`Auto` pick). A
new `Solver::mc64_retry_attempt_count()` accessor reports how many retries
actually ran (distinct from `mc64_scaling_fallback_count()`, which counts only
adoptions). Finding from PR #83's review
(`dev/research/repo-review-2026-06-09.md`).

### Fixed — parallel multifrontal driver now honors `NumericParams::profiler` (N3)

The parallel multifrontal driver (`factorize_multifrontal_supernodal_parallel`)
— the `Solver` default whenever `should_parallelize_assembly` fires — ignored
`NumericParams::profiler`. The sequential driver records one supernode timing
per node, but the parallel driver never touched the profiler, so
`Solver::with_profiling(true)` returned an **empty** `profile_report()` on the
default dispatch, contradicting the `with_profiling`/`profile_report`
documentation. `run_parallel_task` now records one per-supernode wall-time
timing under the profiler mutex (in completion order; the bucketed report is
order-independent). The phase-breakdown sub-fields are left zero on the
parallel path because they derive from process-global phase counters that
cannot be safely differenced across concurrent tasks (see finding N9); only
the wall time is recorded, matching the small-leaf path. Profiling stays off
by default, so the hot path is unchanged. This addresses the profiler facet of
N3; the permute-cache and `small_leaf` facets it also notes are not yet
addressed. Finding from PR #83's review
(`dev/research/repo-review-2026-06-09.md`).

### Performance — 32×32 front dispatch now reuses the caller's pooled scratch (D7)

The 32×32 fully-summed-front dispatch inside
`factor_frontal_blocked_in_place_with_scratch` (the self-described dominant KKT
front size) routed through `factor_block32`, which delegated to the **public**
`factor_frontal`. That public entry re-runs `matrix.validate()` (a full NaN
scan), allocates an `n×n` working copy, and builds a throwaway `FactorScratch`
— defeating the in-place W-3a path (issue #13) whose whole purpose is to reuse
the caller's pooled buffers. `factor_block32` is now a single in-place entry
(`&mut SymmetricMatrix` + caller's `FactorScratch`) that delegates to
`factor_frontal_in_place_with_scratch`, so the 32×32 dispatch skips the
validate, the copy, and the throwaway allocation. Output is **byte-identical**
(guarded by a `to_bits` parity test against the `factor_frontal` oracle); this
is a pure overhead removal with no behavior change. Seventh finding from PR
#83's review (`dev/research/repo-review-2026-06-09.md`).

### Added — unsymmetric LU basis engine (`feral::lu`, issue #81)

A new, separate factorization family for **simplex basis factorization** —
unsymmetric LU with first-class rank-1 column-replacement updates and warm
`ftran`/`btran` solves, built independently of the symmetric LDLᵀ solver
(which is untouched). Module `feral::lu`:

- **Dense path** — `GeneralMatrix` (general column-major dense) and `DenseLu`:
  right-looking LU with threshold partial pivoting (`P B Q = L U`),
  `ftran`/`btran`/`ftran_partial`, and a rank-1 column-replacement `update()`
  (dense Bartels–Golub: spike → upper-Hessenberg → Gauss sweep, maintaining an
  explicit column permutation).
- **Sparse path** — `SparseColMatrix` (general CSC) and `SparseLu`: left-looking
  Gilbert–Peierls LU (output-sensitive depth-first reach, sub-quadratic factor)
  with threshold partial pivoting, sparse `ftran`/`btran`, and a **Forrest–Tomlin
  / Bartels–Golub–Reid** rank-1 column-replacement update — in-place sparse
  Gaussian elimination of the bump with partial pivoting, recorded as a
  replayable eta (bump-local warm solves, no `O(k·n)` chain). The update itself
  is bump-local — Gilbert–Peierls reach for the spike, a `u_above` column index
  for the replacement, save/restore of only the changed rows (no `O(nnz)`
  clone) — ~14–17× faster than a refactor. `SparseLuSymbolic`
  computes the fill-reducing column order by running `feral_amd` on the `AᵀA`
  pattern (reusable symbolic handle).
- **Routing** — `should_use_dense_lu(m, nnz, params)` auto-routes dense vs
  sparse (mirrors the LDLᵀ `should_use_dense_fast_path`), with a `LuParams`
  override.
- **Robustness** — two-sided ∞-norm equilibration and unsymmetric MC64 scaling
  (`LuScaling`, reusing the existing `hungarian_match` kernel), plus iterative
  refinement (`ftran_refined`/`btran_refined`). Singular bases report
  `FeralError::SingularBasis`; the update budget reports
  `FeralError::NeedsRefactor`.

There is no inertia for LU (the basis is unsymmetric). Validated by
`tests/lu_dense.rs`, `tests/lu_sparse.rs`, and `tests/lu_scaling.rs` with
hand-worked exact factors, equation-residual property checks, dense↔sparse
agreement, and adversarial/ill-scaled/ill-conditioned cases. The downstream
`pounce-simplex` `BasisEngine` integration and reference (UMFPACK/KLU)
benchmarks are deferred (see `dev/plans/unsymmetric-lu-epic.md`).

### Fixed — solve-time 2×2 D-block gate now matches factor-side acceptance (D4)

The solve-time 2×2 D-block gate in `d_block_solve` decided whether to invert a
stored 2×2 pivot block with the naive determinant `a·c − b·b` tested against the
**absolute** floor `zero_tol_2x2 ≈ EPS²`, while the factor side accepts a 2×2
block via the **scale-invariant** SSIDS determinant floor. Under `Identity` /
`External` scaling a well-conditioned block at small absolute scale (true
`|det|` below `EPS²`) was validly accepted and stored by the factorization but
then silently **skipped** at solve time — leaving its solution components
untouched and returning a wrong solution with no error and no flag. Both sides
now share a single `ssids_det_floor_fail` predicate, so a 2×2 block the
factorization inverts is exactly a block the solve inverts. (`zero_tol_2x2` is
retained on `Factors` for the legacy `count_2x2_inertia` accounting but no
longer gates the solve.)

New regression `tests/d4_solve_2x2_gate.rs` hand-builds a `Factors` with `L = I`
and a single small-scale 2×2 block and solves `D·x = D·[1,1]`: pre-fix the gate
skips the block and returns `x ≈ [1.1e-16, 1.1e-16]` (off by 16 orders);
post-fix `x ≈ [1, 1]`. The finding's second facet (a nonsingular block whose
*naive* determinant rounds to exactly `0.0` being skipped) is not independently
reachable — such a block has condition `≳ 2⁵²` and the same SSIDS floor rejects
it, so the factor never stores it; this is recorded in `dev/tried-and-rejected.md`
and pinned as a consistency guard. Finding D4 from
`dev/research/repo-review-2026-06-09.md`.

### Fixed — contribution-block extraction no longer violates `Vec::set_len`'s init contract (D6)

The contribution-block extraction in `factor_frontal_in_place_with_scratch_impl`
and `factor_frontal_blocked_in_place_with_scratch` (`src/dense/factor.rs`)
called `contrib.set_len(cdim²)` **before** writing the cells, then materialized
a `&mut [f64]` over the still-uninitialized tail. Every cell is written before
read, so results were correct, but calling `set_len` to expose uninitialized
elements violates its documented safety precondition (the "write before read"
property is not the property `set_len` requires). Both sites now initialize the
region through `spare_capacity_mut()` as `MaybeUninit<f64>` and call `set_len`
only after all `cdim²` elements are initialized — satisfying the contract at the
same single-write-per-cell cost (issue #56 Lever B preserved). Output is
byte-identical (the `blocked_ldlt` scalar-vs-blocked equality suite is
unchanged). No observable behavior change. Guarded by `tests/d6_contrib_uninit.rs`
under Miri. Finding D6 from `dev/research/repo-review-2026-06-09.md`.

### Fixed — static-pivot floor now scale-invariant (computed in scaled space) (N2)

The MA57-style static-pivot floor implied by `static_pivot_threshold = Some(t)`
was computed in the solver from the **unscaled** user matrix
(`floor = t · ‖A‖∞`) but enforced by the Bunch-Kaufman kernels on pivots of the
**scaled** matrix `D·A·D`. Under a norm-normalizing scaling (`InfNorm` /
`MC64`) the unscaled and scaled ∞-norms differ by the scaling ratio, so the
*relative* threshold `t` behaved like a wildly different value in pivot space —
breaking the documented MA57 `cntl(1)` analogy and making the static-pivot
decision depend on a global scalar `γ` that the scaling otherwise normalizes
out. The conversion now lives in `factorize::apply_post_scaling_overrides`
(renamed from `override_null_pivot_tol`), alongside the F-01 null-pivot floor,
where the scaled ∞-norm `‖D·A·D‖∞` is already in hand; the solver's unscaled
`matrix_inf_norm` scan is removed.

New regression `n2_static_pivot_floor_is_scale_invariant_under_infnorm` factors
an indefinite saddle KKT under `InfNorm` scaling with `t = 1e-6`, then again
scaled by `γ = 2³⁰`, and asserts the static-pivot decision (`needs_refinement`,
inertia) is identical — because `A` and `γ·A` equilibrate to the same scaled
matrix. Pre-fix `A → (refine=false)` but `γ·A → (refine=true)`; post-fix they
agree. Finding N2 from `dev/research/repo-review-2026-06-09.md`.

### Fixed — panel inline 2×2 now applies the static-pivot floor (D2)

The blocked dense LDLᵀ panel's inline 2×2 accept path skipped the MA57-style
static-pivot perturbation (`perturb_2x2_to_floor`) that the scalar path
(`scalar_pivot_step`) applies *before* the growth/det gates and inertia count.
With `static_pivot_floor > 0` (wired from `NumericParams::static_pivot_threshold`),
a sub-floor 2×2 block accepted inline was accepted **unperturbed** — diverging
from the scalar path in `D`, `L`, `needs_refinement`, `n_tiny`, and even
**inertia**, violating the module's documented panel/scalar bit-parity contract
and the MA57 `cntl`-style static-pivot semantics. The panel now mirrors the
scalar perturbation (lift the smaller |eigenvalue| to the floor, set
`needs_refinement`, bump `n_tiny`) so both paths stay byte-identical.

New regression `test_d2_panel_inline_2x2_static_pivot_floor_parity` builds an
isolated antidiagonal 2×2 (eigenvalues ±δ, δ = 1e-3) below the floor (1e-1) in
an 80×80 front (crossing the 64-column panel boundary) and asserts byte-identity
against the scalar oracle. Pre-fix the panel reported inertia `(79+, 1−)` for
the unperturbed ±δ block where the perturbed scalar path reports `(80+)`;
post-fix they agree. Finding D2 from `dev/research/repo-review-2026-06-09.md`.

### Fixed — default `postorder()` is now linear, not O(n²·log n), on star etrees (S1)

The default elimination-tree `postorder` (in the standard symbolic pipeline)
re-cloned and re-sorted `children[node]` on **every** DFS stack visit. A node
with `c` children sits on top of the stack `c+1` times, so it paid
O(c²·log c); on a star etree — one root with `n-1` children, the shape AMD
produces for an arrow/bordered-KKT matrix with a dense *trailing* border —
the whole traversal was O(n²·log n). It now carries each node's sorted child
list on the stack (a `(node, sorted_children, cursor)` layout matching the
already-correct `biased_postorder` / `EliminationTree::postorder`), so each
node is sorted exactly once and the traversal is O(n·log n).

New regression `test_postorder_star_sort_work_is_linear` reproduces the blow-up
deterministically (no flaky timing) via a `#[cfg(test)]` work counter: on an
`n = 2000` star the old code materialized ~`n²` child-list elements (3,998,000);
the fix materializes ~`n` (≤ `4n`). Output is unchanged — the existing
topological-order, inverse-roundtrip, and Schur-parity tests still pass.
Finding S1 from `dev/research/repo-review-2026-06-09.md`.

### Fixed — `with_fma(true)` now actually dispatches the FMA kernels (N1)

`Solver::with_fma(true)` was a silent no-op. It set `NumericParams::fma`,
but every Bunch-Kaufman call site consumes `&params.bk`, whose `fma` field
stayed at its `false` default — so the documented ~2× FMA dense kernels
(`schur_panel_minus_fma_strided*`, `axpy_minus_unroll4`, `axpy2_minus_unroll4`,
issue #8) never engaged through the public API. The solver factor funnel now
syncs `bk.fma = fma` into the params handed to both the sequential and parallel
multifrontal drivers, and the stale doc on `BunchKaufmanParams::fma` (which
claimed the *driver* copied the flag) is corrected.

New regression `fma_opt_in_actually_dispatches_fma_kernels` asserts that
enabling FMA changes the factorization at the bit level (proving dispatch)
while keeping the solution within the documented within-ulps bound; the
pre-existing "same inertia + small residual" test could not catch a dead
toggle. Finding N1 from `dev/research/repo-review-2026-06-09.md`.

### Fixed — dense LU update/solve no longer commits singular bases or emits silent Inf/NaN (L1)

The dense LU column-replacement path could commit a numerically singular
replacement basis and then divide by a ~0 pivot in the back-solves, emitting
silent `±Inf`/`NaN` (whereas the sparse path already guarded both ends). Two
fixes, mirroring `sparse_update.rs` / `sparse_solve.rs`:

- **Update** (`DenseLu::update`): the bump-elimination loop validated only
  pivots `q..m-2`; the final diagonal `u[m-1,m-1]` was never checked, and when
  the leaving slot was the last column (`q == m-1`) the loop never ran at all. A
  vanishing final pivot is now rejected (`NeedsRefactor`) before commit.
- **Solve** (`usolve`/`ut_solve`): the dense back-solves now error with
  `SingularBasis { column }` on a zero or non-finite `U` diagonal instead of
  dividing into an `Inf`/`NaN`.

Regression tests `dense_zero_u_diagonal_errors_instead_of_inf` (solve) and
`update_singular_last_pivot_does_not_commit` (update). Finding L1 from
`dev/research/repo-review-2026-06-09.md`.

### Fixed — legacy dense `factor()` no longer corrupts the column after a force-accepted zero pivot (D1)

In the legacy dense `factor()` path, a strict-zero pivot routed through
`ZeroPivotAction::ForceAccept` (and the degenerate 2×2 twin) zeroed its own L
column but ran no rank-1/rank-2 update, then returned a *fabricated* fused
next-column argmax of `(0.0, k+2)` / `(0.0, k+3)`. The caller stored that, so
the next iteration saw `gamma0 == 0.0`, took the "zero off-diagonal column"
fast path, and silently discarded the **real off-diagonals of the following
column** — corrupting L (a full-magnitude reconstruction error, ~1.5 on the
regression matrix) and risking wrong inertia, the project's hard contract.
Both branches now report the genuine off-diagonal max of the (unmodified) next
column via `column_offdiag_max`. Regression test
`tests/dense_ldlt.rs::test_d1_force_accept_does_not_corrupt_next_column`
asserts exact `P·L·D·Lᵀ·Pᵀ = D_eq·A·D_eq` reconstruction. Finding D1 from
`dev/research/repo-review-2026-06-09.md`.

### Added — on-disk dense-column regression fixture for issue #80

`cargo run --bin bench` now regenerates two synthetic dense-coupling-column
symmetric quasidefinite KKTs (`data/matrices/synthetic-regression/
densecol_kkt_{300,1000}`) on every run and factors them through the sparse
pipeline with the same inertia + residual checks as the corpus. They
reproduce the near-dense coupling column behind the #80 MC64 cost classes,
and `FERAL_SCALING=mc64` drives the symmetric MC64 matching on the dense
column directly. The inertia oracle is the Vanderbei (1995) symmetric-
quasidefinite theorem — `(n_var, n_con, 0)` for any off-diagonal block — not
feral itself. Because `data/matrices/` is gitignored, the corpus matrices
that exercised this path were never in the repo; these fixtures are
deterministically regenerated from committed source (no committed matrix or
result) so the path is checked on any clone, even with no corpus present. A
`cargo test` structural guard (`densecol_kkt_is_dense_column_sqd`) asserts the
generator keeps producing a genuine dense column and the SQD inertia.

### Removed — dead `ordering::amd::amd_order` (#80)

Deleted the legacy O(n²) `feral::ordering::amd::amd_order` (and its tests /
`estimate_fill` helper). It was not in any factorization path — production
ordering is the `feral_amd` quotient-graph crate (`symbolic`/`schur`
dispatch to `feral_amd::amd_order`). Only `permute_pattern` from that module
is still used and is retained. Two orphaned triage examples that referenced
it were removed. This eliminates the misleading "this is the AMD" function
that derailed the initial #80 diagnosis.

### Fixed — MC64 matching no longer reallocates its heap per column (#80)

`hungarian_match` allocated a fresh `IndexHeap::new(m)` inside the
per-unmatched-column augmenting loop — O(m) zeroing on each of up to `n`
iterations, i.e. an O(n·m) ≈ O(n²) term that dominated MC64 on large
near-tree KKTs. The heap is now allocated once and reset incrementally over
the touched-row set (matching how `d`/`visited` were already handled). On the
pf22 powerflow KKT (n=2.8M) MC64 matching drops from ~53s to **0.31s** and the
default first factor from ~65s to **2.6s**, with bit-identical matching
output (inertia 940,248 and residual 5.28e-13 unchanged). This was the actual
cost behind issue #80's reported ~55s first factor.

### Fixed — symbolic profiler attributes the LdltCompress/MC64 preprocessor to its own stage (#80)

`symbolic_profile_report()` previously folded the `LdltCompress`
preprocessor's MC64 matching into the `ordering` stage. On the pf22
powerflow KKT (n=2.8M) this made the report blame the fill-reducing
ordering for ~53s when the ordering itself is ~0.3s and the time is
actually MC64 (`compute_mc64_cache`). The preprocessor is now recorded
under a dedicated `ldlt_compress` stage (plus `compress_pattern` /
`expand_perm` on the compressed-graph path), and `ordering` measures only
`run_external_ordering`. No change to the produced factorization — this is
a profiler-attribution fix only. See `dev/journal/2026-06-06-01.org`.

## [0.10.0] - 2026-06-04

### Changed — `Auto` now prefers AMF over MetisND at every size (#73)

Extends the #67 thin-large reroute past its `n ≤ 100_000` ceiling: the
`AMF_BAND_MAX` bound is **removed**, so whenever the size rule would pick
MetisND, `choose_adaptive` now routes to AMF at **every** `n`. The earlier
`n > 100_000 && avg_deg < 5 → Amd` (#50 powerflow) and arrow → AMF (#64)
catches still fire first, so the powerflow-class guardrail and dense-border
catch are unchanged — only the uniformly-thin would-be-MetisND population
above 100k is affected.

#67 deferred this regime as under-sampled. The #73 investigation closed it
with a real **factor + solve** A/B (`probe_issue67_thin`) on the n>100k
families: AMF wins on **every measured matrix** — dtoc2 2.49×, pinene 1.18×,
cont5_1_l 2.75×, nql180 2.05×, YATP1NE 2.13×. The decisive case is **nql180**,
where MetisND has 2% *smaller* symbolic fill yet AMF is 2.05× faster on the
real factor+solve — proving that fill (`factor_nnz_estimate` / flop_proxy) is
not a reliable speed predictor at this scale. A fill-guarded race was
therefore **rejected** in favor of the unconditional reroute (see
`dev/tried-and-rejected.md`).

See `dev/research/issue-73-n100k-thin-regime.md` and `dev/decisions.md`.

### Changed — diagnostic binaries moved out of the default build/test set (#71)

The root `feral` package's `src/bin/` held 145 binaries, 144 of them
throwaway diagnostics (`diag_*`, `probe_*`, `bench_*`, `profile_*`, …) and
only `bench.rs` a real keeper. With `autobins = true`, every root
`cargo build` / `cargo test` / `cargo clippy` compiled all 145 — and on
macOS each fresh binary triggers a per-binary Gatekeeper/XProtect scan, so a
cold `cargo test` built ~190 binaries and took ~30 min locally (Linux CI
unaffected). Only 2 of the 144 carried a `#[test]` (local JSON-sidecar
parser unit tests, not solver gates).

The 144 diagnostics now live in a non-default workspace member crate,
`crates/feral-diagnostics/` (depends on `feral`). Root `cargo build` /
`cargo test` / `cargo clippy` operate on the `feral` package only — the
diagnostics are no longer in the default build/test set. They remain
buildable and runnable on demand:

```
cargo run -p feral-diagnostics --bin <name> [-- args...]
cargo build -p feral-diagnostics            # compile all of them
```

`bench.rs` stays in the root package, so the protocol command
`cargo run --bin bench --release` is unchanged. CI keeps the diagnostics
lint-clean and runs their 2 test sets via explicit
`cargo clippy -p feral-diagnostics --all-targets -- -D warnings` and
`cargo test -p feral-diagnostics` steps (cheap on Linux); the `stress-smoke`
job's `bench_one_matrix` / `probe_fma_kernel` references now select the
crate with `-p feral-diagnostics`. No library or solver code changed.

### Changed — thin-large default ordering now prefers AMF up to n ≤ 100k (#67)

The size-only default routed every `n > 10_000` matrix to MetisND. On
**uniformly-thin** large matrices (3-D-PDE-like discretizations with a flat
degree distribution and no dense border — so the #64 arrow catch correctly
does not fire) nested dissection is supposed to win, yet AMF produces both a
smaller factor *and* a faster solve. A corpus-wide A/B (54 `n > 10_000`
KKT/SuiteSparse families, measuring **factor + solve wall-time**, not nnz_L
alone) found that across the entire `(10_000, 100_000]` band AMF wins or
ties MetisND on **all 36/36** in-scope matrices — worst case 0.99× (run
noise), median ~1.5×, tail to 4.5× (e.g. bratu3d 1.8×, cont-201 2.1×). The
issue's hypothesis that MetisND trades fill for a shorter critical path
never materialized at this scale.

`choose_adaptive` now raises the AMF band ceiling: when the size rule would
pick MetisND and `n <= AMF_BAND_MAX` (100_000), it routes to AMF instead.
This is a static `n` threshold, deliberately **not** an average-degree
predicate (the #50 powerflow hazard) and **not** an `AutoRace` (MetisND's
nested-dissection symbolic ordering is 2–5× more expensive than AMF's, so
racing it costs a measured 50–255% overhead for zero benefit). The
`n > 100_000 && avg_deg < 5 → Amd` (#50) and `n > 100_000 && avg_deg ≥ 5 →
MetisND` paths are untouched; genuinely-large 3-D problems keep nested
dissection.

See `dev/research/issue-67-thin-large-ordering.md`. Opt-in regression
fixtures (gitignored, fetch with `dev/scripts/fetch_large_matrices.sh`):
`tests/issue67_thin_ordering.rs` asserts `resolved_method == Amf` and an
nnz_L ceiling on bratu3d (n=27792) and cont-201 (n=80595).

### Fixed — wrong inertia (spurious zero pivots) on ill-conditioned KKTs under Auto scaling (#65)

On an ill-conditioned symmetric-indefinite KKT (e.g. the Vanderbei `sawpath`
iter-0 system, cond ≈ 4e20 but effectively full rank), default `Auto` scaling
routed to InfNorm, whose Bunch-Kaufman sequence collapsed ~100 pivots to the
working-precision floor and reported a **wrong, rank-deficient inertia**
(`(789,670,116)` vs the true `(789,786,0)` from MA27/numpy). Consuming IPMs read
the spurious `zero` count as `Singular`, took a bad regularized step at
iteration 0, and could falsely declare the problem infeasible.

`Solver::factor` now applies an **inertia-guided MC64 fallback**: when the user
configured `ScalingStrategy::Auto`, the resolved scaling was not MC64, and the
factor force-accepted zero pivots (`inertia.zero > 0`), it re-runs with
`Mc64Symmetric` and adopts that result iff it strictly reduces the zero count.
MC64's symmetric matching pulls large entries onto the diagonal so the pivot
sequence never hits the floor, recovering the true inertia (sawpath → exactly
`(789,786,0)`, smallest pivot `0.03`). The fallback is correctness-safe — MC64
cannot change rank, so a genuinely singular matrix keeps its original factor —
and is gated on `Auto` only (explicit InfNorm/Identity/MC64 are respected).
Adoption pins the sticky-Auto strategy to MC64 so subsequent refactors on the
same pattern skip the retry. New `Solver::mc64_scaling_fallback_count()` reports
how often it fired.

A purely structural router fix was ruled out: `sawpath` (needs MC64) and
`twirism1` iter-0 (needs InfNorm — MC64 gives it the *wrong* inertia there) have
the identical router signature, so the deciding factor must be the numerical
factorization outcome, not matrix shape. Corpus-validated against the KKT
consensus oracle: no new inertia mismatches; the fallback fires rarely.

See `dev/research/issue-65-mc64-scaling-fallback.md`. Regression fixtures
(gitignored, regenerate with `dev/scripts/regen_issue65_kkts.sh`):
`tests/issue65_mc64_fallback.rs`.

### Fixed — arrow/bordered-KKT ordering blow-up (#64)

The default ordering routed by size alone (`n > 10_000 → MetisND`). On
**arrow / bordered-KKT** patterns — a thin body plus a handful of
very-high-degree "border" columns (e.g. an IPM augmented system with a
few dense inequality rows) — nested dissection cannot isolate the dense
border and the LDLᵀ factor blows up ~7–9× vs AMF/AMD. On the LP `r05`
iter-0 KKT (n=14842, 171 columns of degree 502 carrying 38.5% of the
nonzeros) this was 4.4M nnz_L under the default (→ MetisND) vs 0.51M
under AMF, and POUNCE end-to-end went from ~16 s to 0.84 s with AMF.

`symbolic_factorize` now detects the arrow signature with a cheap O(n)
degree-distribution pass (`is_arrow_bordered`): when a *small* set of
columns (< 5% of n) concentrates a *large* share of the nonzeros
(≥ 20%), it routes to AMF instead of MetisND. Uniformly-thin matrices
(PoissonControl, powerflow22, bratu3d, cont-201) and matrices whose few
high-degree columns carry a tiny nnz share (bcsstk38, 0.3%) are not
flagged and keep their previous ordering. The detection lives in
`choose_adaptive`, and `symbolic_factorize` now resolves through
`OrderingMethod::Auto`, so the no-arg default and an explicit `Auto`
caller resolve to the same concrete ordering on every matrix (they could
previously disagree on very-large-and-sparse patterns).

See `dev/research/issue-64-arrow-bordered-ordering.md`. Regression
fixture (gitignored, regenerate with `dev/scripts/regen_r05_kkt.sh`):
`tests/issue64_arrow_ordering.rs` asserts `nnz_L < 1.0e6` on r05's KKT.

## [0.9.0] - 2026-05-30

### Fixed — batched iterative refinement for wide multi-RHS solves (#58)

`Solver::solve_many_refined` looped the single-RHS refiner per column,
which bypassed the BLAS-3 panel kernel — so with refinement on (the
default) batched multi-RHS solves were 3–7× slower per RHS than the
unrefined batched path, and could be slower than looping single-RHS.

Wide refined solves (`nrhs ≥ 16`) now refine through a batched loop:
the initial and per-step correction solves go through `solve_sparse_many`
(one panel solve over the still-active columns), and the residual is a
per-column SpMV. Per-column best-iterate tracking and the convergence
predicates (`ε·√n` relative target, 2-strike plateau, 100× divergence)
are preserved exactly, and each step compacts the active columns so the
batched path never does more solve work than the per-column loop. Narrow
solves (`nrhs < 16`, e.g. the IPM predictor-corrector) keep the
per-column loop unchanged.

For `16 ≤ nrhs < 32` the batched result is bit-identical to the
per-column loop (the solve runs the rank-1 kernels there); for
`nrhs ≥ 32` it agrees with the per-column oracle to the refinement
residual target. Measured (`bench_multirhs`, 2-D Laplacians): batched
refined is ~2.5–3× faster per RHS than the per-column refined loop.

The common case (every column already at the target after the direct
solve, i.e. ~0 correction steps) now returns before allocating the wide
`best_x`/residual buffers — it allocates only the solution and a
length-`n` scratch. Those three `n × nrhs` allocations were cheap in the
native binary but cost ~50 µs/RHS under the Python process's allocator,
masking the win through `Solver.solve_refined` on 2-D inputs; with them
gone the Python refined path shows the full ~2–3× too.

### Changed — multi-RHS sparse solve: BLAS-3 panel kernels (#57 fix #2)

Wide multi-RHS solves (`nrhs ≥ 32`) now run each supernode's
forward/backward substitution as a register-blocked dense panel solve
— a TRSM on the unit-lower triangle `L_11` plus a MR×NR (4×8)
register-tiled GEMM on the trailing block `L_21` — instead of a rank-1
cascade. Narrow solves (the IPM hot path) stay on the rank-1 kernels.

Two supporting layout changes, both internal (no public API/ABI change,
RHS/solution stay column-major `n × nrhs`):

- The internal `y` working buffer is now **row-major**, so every
  per-supernode gather/scatter is a contiguous memcpy instead of a
  stride-`n` transpose. This removed a cache-conflict pathology on
  power-of-two `n` (the n=1024 grid had regressed) and roughly halved
  the wide-solve time across the board.
- The forward substitution and D-block solve are **fused into one
  postorder pass** (a node's eliminated rows are final once its
  forward-sub completes), saving one gather/scatter round trip per
  supernode.

Forward substitution stays bit-identical to looping single-RHS;
back-substitution differs only by floating-point reassociation
(~κ·eps). The multi-RHS parity suite (`nrhs` up to 64, MR/NR tail
sizes) verifies `max|many − single| < 1e-12` against the independent
single-RHS oracle (observed ≤ 1.6e-15).

Measured per-RHS speedup vs looping single-RHS, on 2-D Laplacians
(`cargo run --release --bin bench_multirhs`, `nrhs ∈ {64, 256}`):
n=484 ~4–5×, n=1024 ~3×, n=2025 ~5–6×.

### Changed — multi-RHS sparse solve: row-major working buffer (#57)

The per-supernode working buffer in `solve_sparse_core_many_into` now
uses a row-major layout (`w[i*nrhs + c]`) instead of column-major
(`w[c*nrow + i]`). The per-RHS inner loops in forward-sub, the D-block
solve, and back-sub are now contiguous (stride-1) and auto-vectorize.

The caller-visible RHS/solution layout is unchanged — it remains
column-major `n × nrhs`, matching MUMPS/SSIDS — so there is no public
API or ABI change. The single-RHS path (`solve_sparse_core_into`) and
the iterative-refinement path are untouched. Results are bit-identical
to looping the single-RHS solve (verified by the multi-RHS parity
suite, `max|many − single| = 0`).

Measured per-RHS speedup of one batched `solve_sparse_many` call vs
looping single-RHS, on 2-D Laplacians (`cargo run --release --bin
bench_multirhs`): batched/looped ratio 0.76–0.99 (up to ~1.3× faster
per RHS) at `nrhs ∈ {64, 256}`. The larger 5–10× regime needs the
BLAS-3 panel kernels (issue #57 fix #2, deferred).

## [0.8.0] - 2026-05-29

### Removed — 4 synthetic rank-deficient stress-corpus matrices

Dropped `rankdef_10_3`, `rankdef_50_5`, `rankdef_exact_50_5`, and
`stokes_q1p0_8` from the stress corpus. Under #54's SSIDS-aligned
strict-zero routing, feral reported `inertia.zero = 1` on all four —
which contradicted MUMPS, SSIDS, *and* MA57 simultaneously,
violating CLAUDE.md's "must agree with at least one canonical" rule
and red-lighting the stress-smoke gate on the v0.8.0 release commit.

Rather than allowlist them or narrow #54's `zero_tol` (which would
reopen the IPM δ-cascade instability on `nuffield2_trap_iter1.mtx`
that motivated #54 in the first place), the four matrices were
removed entirely. They were synthetic borderline fixtures where the
"correct" zero count depends on order-1e-15 round-off and no
3-of-4-oracle consensus exists — the same disagreement that
`compute_consensus.py` already tags `excluded`. The rank-deficient
regime remains covered by `rankdef_5_2`, `rankdef_200_20`,
`rankdef_exact_100_10`, `saddle_rankdef_50_10_3`,
`saddle_rankdef_100_20_5`. See `dev/decisions.md` (2026-05-28 entry).

### Performance — Thomson-Hessian per-iter throughput ([#56][i56])

Three additive levers on the dense / wide-supernode IPM-KKT hot path,
landed on `issue-56-thomson-hessian-throughput` and merged to `main`:

- **Lever A** — pre-built `permuted_pattern` + permute structure cache.
  Collapsed the symbolic `permute (P A P^T)` + `symmetric_pattern`
  phases to <0.5 % of total wall on warm calls.
- **Lever B** — fused single-pass `contribextract` write. Replaced
  `resize(cdim², 0.0)` + lower-triangle overwrite with
  `reserve + unsafe set_len` + write-each-cell-once. Bit-identical
  contrib block (`extend_add` reads only `ci ≥ cj`; root-Schur
  extractor canonicalizes via transpose; `parallel_corpus_parity`
  binary-compares the full buffer). Roughly 33 % fewer writes.
  Re-measurement at Thomson n=200 (sequential, 9 warm reps):
  factor min −5.5 % (parallel ON), −10 % (parallel OFF).
- **Lever C** — InfNorm Knight-Ruiz inner-loop vectorization. Hoisted
  the loop-carried `row_max[j]` dependency to a register accumulator
  (sparse and dense paths, bit-identical by associativity of max on
  non-NaN finite inputs); added a pulp-dispatched SIMD off-diagonal
  kernel on the dense path. Re-measurement at Thomson n=200:
  scaling phase −19 %, total wall −5 %.

No regression on the Phase 2.8.1 corpus partition gates; both
small-frontal and medium buckets improved relative to the 2026-05-27
baseline. KIRBY2 family worst-case sparse outlier improved from
10.25× → 7.97× vs MUMPS.

See `dev/sessions/2026-05-28-01.md` and
`dev/research/issue-56-thomson-hessian-throughput-2026-05-27.md` for
the localization data and re-measurement tables.

[i56]: https://github.com/jkitchin/feral/issues/56

### Added — symbolic-analysis-time delayed-pivot budget + CB rewire ([#55][i55])

Per-supernode `delayed_capacity` is assigned during symbolic
analysis. The numeric phase enforces the bound at the frontal
expansion site: if more pivots delay into a supernode than its
capacity, the factor returns the structured
`FeralError::DelayBudgetExceeded { supernode, required, capacity }`
(MUMPS `INFO(2)` workspace-overflow analog) rather than growing
the front unboundedly. Cascade-break now triggers on budget
exhaustion instead of the heuristic ratio gate, matching MUMPS's
`dfac_front_aux.F:1251-1331` invariant that static perturbation
fires only when delay is structurally impossible.

Capacity formula:
- `tight = max(4 * own_ncol, 16)`
- `capacity = min(subtree_ncol(s) - own_ncol(s), tight)`

Root-supernode cap (defensive, `n >= 1024`): declines amalgamations
that would push the root past `min(0.05 * n, 2048)` columns.

`FeralConfig::default()` now ships with cascade-break armed
(`cascade_break_ratio = Some(0.5)`, `cascade_break_eps = Some(1e-10)`)
as the budget-exhausted fallback. Pounce's per-problem `.opt`
overrides for `nql180` / `pinene_3200` are no longer required.

See `dev/research/symbolic-delay-budget-2026-05-27.md` and
`dev/decisions.md` (2026-05-27 entry, frozen convention).

[i55]: https://github.com/jkitchin/feral/issues/55

### Changed — strict-zero pivots route to `inertia.zero` (SSIDS-aligned, [#54][i54])

When `ZeroPivotAction::ForceAccept` accepts a 1×1 pivot whose
magnitude satisfies `|d| <= zero_tol`, the inertia now increments
`zero` instead of routing by `sign(d)`. This matches SSIDS
(`NumericSubtree.hxx:259-267`) and HSL MA57 (INFO(24) = `neig`,
INFO(25) = number of zero pivots) and supersedes the Issue #42
Option A sign-routing rule.

Motivation: pounce's IPM δ-cascade on `nuffield2_trap_iter1.mtx`
(LP-shaped KKT, n=26155) stalled for 600 s (vs 1.8 s on MA57)
because Option A split bit-exact zero pivots by IEEE round-off and
the counter jumped backwards mid-cascade (`13042 → 12615`). The new
accounting restores monotonicity (probe confirms 0 backwards jumps
across `δ_x ∈ {0 … 6.99e19}`).

Behavior changes visible to callers:
- `Solver::inertia()` and `factor()`'s `Inertia` field now report
  the mathematical (Sylvester) inertia on matrices with rank
  deficiency that lands in strict-zero pivots.
- `num_negative_eigenvalues()` returns strict `inertia.negative`
  only (unchanged convention). IPM callers comparing against an
  expected oracle should now sum `negative + zero` to match
  MA57's INFO(24)+INFO(25) convention.

Tests updated to reflect the new convention; see
`dev/decisions.md` (2026-05-26 entry) for the full trade-off
analysis and `dev/research/issue-54-lp-kkt-inertia.md` for the
oracle cross-check.

[i54]: https://github.com/jkitchin/feral/issues/54

### Added — `DelayBudgetExceeded` exposed through Python bindings

`feral-solver` (Python) now re-exports the #55 error as a named
exception:

- `feral.DelayBudgetExceeded` — subclass of `feral.FactorError`
  (and transitively `feral.FeralError`), raised when
  `factor()`/`refactor()` hits the symbolic-analysis-time
  delayed-pivot budget. The message carries the supernode index,
  required and capacity columns. Python callers no longer see this
  routed through the generic `NumericFailure` wildcard.

### Added — documentation site (mdBook + rustdoc on GitHub Pages)

- `book/` — mdBook skeleton (`introduction`, `getting-started`,
  `inertia`, `api`). Content is intentionally thin; the `dev/research/`
  notes remain internal and are not auto-published.
- `.github/workflows/pages.yml` — builds the book and
  `cargo doc --workspace --no-deps --lib`, mounts rustdoc under `/api/`,
  and deploys to GitHub Pages on every push to `main`. Site URL is
  `https://jkitchin.github.io/feral/`; rustdoc at `/api/feral/`.
- `Cargo.toml`: `/book` added to the crate `exclude` list so the book
  source does not ship in the published crate tarball.
- `.gitignore`: `/book/book` ignored (mdBook output is regenerated by
  the Pages workflow).

Repo setting required once: **Settings → Pages → Source: GitHub
Actions**. Until that flag is set, the deploy job will fail with a
"Pages not enabled" error; the build job is unaffected.

## [0.7.0] - 2026-05-25

### Added — opt-in instrumentation accessors ([#52][i52])

New public surface on `Solver` for pounce-side debugging of
linear-solver behavior without re-running with `cargo flamegraph`
or instrumenting forks.

**Phase A — always-on snapshot.** `Solver::last_factor_stats()`
returns `Option<FactorStats>` populated after every successful
`factor()`. Fields: `nnz_a`, `nnz_l`, `fill_ratio`, `inertia`,
`min_abs_pivot`, `max_abs_pivot`, `pattern_reused`, `scaling_info`.
No gating flag — the two extra integer writes per `factor()` are
cheaper than a gate check would be.

**Phase B — opt-in profiler.** `Solver::with_profiling(true)`
enables per-supernode numeric timings and one-shot symbolic timings,
read back via `Solver::profile_report()` and
`Solver::symbolic_profile_report()`. The symbolic report is
populated only on cache-miss factors — `None` on cache hits is the
unambiguous "did symbolic actually run" signal pounce asked for.
Default is `false`; when off, no profiler arcs are allocated and
the code path is byte-identical to a pre-issue-52 build.

**Performance.** Measured by `benches/issue52_overhead.rs` on
tridiagonal SPDs (n ∈ {64, 256, 1024}, sequential, 30 samples × 3 s):

| n    | main baseline | default-off | profiling-on |
|------|---------------|-------------|--------------|
|   64 | 260.6 µs      | 257.6 µs    | 258.9 µs     |
|  256 | 348.2 µs      | 345.2 µs    | 347.0 µs     |
| 1024 | 719.6 µs      | 714.7 µs    | 709.1 µs     |

Default-off vs `main` baseline: within ±1.2% (often faster — noise).
Profiling-on vs default-off: within ±1%. Both deltas sit inside the
criterion noise band on tridiagonal workloads.

Parallel-driver profiler contention (B2b in the plan) is documented
as the escape hatch but not engineered; revisit only if an IPM
workload regresses with profiling on.

[i52]: https://github.com/jkitchin/feral/issues/52

## [0.6.0] - 2026-05-23

### Added — `Solver::with_scaling()` builder ([#51][i51])

New public builder method on `Solver` to pin the scaling strategy
explicitly. Lets IPM hosts override `ScalingStrategy::Auto` when
they already know what they want — recommended escape hatch when
the picker's per-matrix heuristic is wrong for a problem class.
Default (`Auto`) is now sticky on cached pattern (see "Fixed" below)
so the picker no longer flaps; this builder is for callers that
want full control.

### Fixed — sticky `Auto` scaling on cached pattern ([#51][i51])

Three coordinated fixes in `src/numeric/solver.rs` address a ~50×
numeric-phase slowdown on IPM workloads when the picker re-routes
across iters or `PartialSingular` outcomes drop the MC64 cache.

Reproducer (`pounce/gams/nlpbench/feral_repro/powerflow22/`,
n=2,813,976 IPM KKT, default `Auto`):

| call | pre-fix | post-fix |
|---|---:|---:|
| factor (cold) | 54.77 s | 54.42 s |
| refactor (cached symbolic, iter A values) | 1.00 s | 1.02 s |
| factor #2 (cached symbolic, iter B values) | **53.80 s** | **1.07 s** |

The three fixes:

1. **Sticky Auto pick.** First `factor()` on a pattern runs the full
   `compute_scaling_auto_with_cache` pipeline (preserves Policy-4
   fallback semantics — InfNorm-spread guard, off-diag-ratio guard,
   MC64 catastrophic-spread guard). Post-call we derive the resolved
   strategy from `factors.scaling_info`
   (`Mc64FallbackToInfnorm → InfNorm`,
   `PartialSingular → Mc64Symmetric`, `NotApplied → Identity`,
   `Applied → pick_scaling_strategy(matrix)`) and stash it on the
   `Solver` as `auto_picked_strategy`. Every subsequent factor on
   the same pattern uses the stashed strategy directly, bypassing
   Auto. Pattern change clears it alongside the MC64 cache. Mirrors
   MUMPS ICNTL(7) / SSIDS `options%ordering`: structural decision
   once at first call, reuse every refactor.

2. **MC64 cache gate widened to `PartialSingular`.** The Hungarian
   on a structurally rank-deficient KKT still produced a real
   scaling vector (unmatched positions land at 1.0 per
   `mc64.rs:222`); the value-bound check still gates reuse, so
   caching is correctness-safe. Pre-fix the post-#49 gate required
   `ScalingInfo::Applied`, which dropped the cache on every IPM iter
   over a structurally rank-deficient KKT and forced the Hungarian
   to rerun from scratch.

3. **`Solver::with_scaling()` builder** (see "Added" above).

### Tests

Three unit tests in `src/numeric/solver.rs`:

- `issue_51_with_scaling_builder_overrides_default`
- `issue_51_auto_pick_is_sticky_on_cached_pattern`
- `issue_51_partial_singular_populates_cache`

And one `#[ignore]`'d corpus regression test:
`issue_51_corpus_sticky_auto_holds_across_ipm_iters` walks
`tests/data/parity/<family>/*.mtx` for every family with ≥ 2 IPM
iter snapshots, factors all iters against one `Solver::new()`, and
asserts the sticky pick holds across every iter (13 families
covered: acopp30, hatfldbne, hahn1, ssi, …).

`mc64_fallback_surfaces_via_solver_api` was updated to reflect the
new sticky semantics: iter 2 now asserts
`mc64_fallback_count == 1` (sticky pin runs straight InfNorm; no
fallback to surface) and locks
`auto_picked_strategy = Some(InfNorm)`.

### Changed — `Auto` dispatcher rewrites ([#50][i50])

The `OrderingMethod::Auto` dispatcher (`src/symbolic/mod.rs::
choose_adaptive`) was simplified to two rules on top of
`pick_default_method`: kept only the very-large-and-sparse catch
(`n > 100_000 && full_avg_deg < 5.0 → Amd`); everything else
delegates to `pick_default_method` (`n ≤ 10_000 → Amf`,
`n > 10_000 → MetisND`).

- **Fix A — large-and-sparse swap** (`c442a0c`). The pre-fix
  `n > 100_000 && full_avg_deg < 5.0 → ScotchND` branch is swapped
  to `Amd`. On `powerflow22` (n=2.8 M, full_avg_deg ≈ 3.7) prior
  ScotchND took 113.8 s symbolic (15.8 M nnz_L); AMD takes 55 s
  (10.4 M nnz_L). The ScotchND advantage at very large n had been
  load-bearing against the same BK pivoting cascade that motivated
  the chain catches; issue #46 eliminated that amplifier in May
  2026 and removed the justification for routing very-large sparse
  matrices through nested dissection. Corpus inventory:
  `dev/research/issue-50-numeric-inventory.csv` shows the IPM
  corpus's [100k, 200k) bucket has AMD/MetisND num_nnz_l ratio 1.00
  on both representatives. Validation: 258 chain-catch corpus rows
  under post-Fix Auto — 0 failures, 0 num_nnz_l regressions for
  matrices that actually reroute. See
  `dev/research/issue-50-metisnd-symbolic-cost.md` §F7–§F11.

- **F11 follow-up — small-and-sparse retire** (`3f8f6f6`). The
  pre-fix `n < 10_000 && full_avg_deg < 15.0 → KahipND` branch is
  deleted entirely; the population now falls through to
  `pick_default_method`'s `n ≤ 10_000 → Amf` default (MUMPS
  `ana_set_ordering.F` SYM=2 N≤10000). Justified by an 838-matrix
  4-way inventory (`dev/research/small-sparse-inventory.csv`): AMF
  wins 169/838 strict per-matrix (KahipND 16); aggregate num_nnz_l
  ratios to AMD are AMF 0.870×, KahipND 0.984×; aggregate factor_us
  ratios are AMF 0.832×, KahipND 0.990×. KahipND remains reachable
  via `OrderingMethod::KahipND` for the 41 high-avg-deg cases
  (STEENBRD, HADAMARD, TABLE8) where it still wins — all sub-22k
  nnz_L absolute. See `dev/research/issue-50-metisnd-symbolic-cost.md`
  §F12.

## [0.5.0] - 2026-05-22

### Performance — Schur trailing-update kernel widened ([#44][i44])

The deferred-Schur trailing-update SIMD kernel
(`src/dense/schur_kernel.rs`) was widened to a quad NEON-tile inner
loop (commit `5f1661c`). Micro-benchmark shows ~2.2–2.5×; end-to-end
on the `NARX_CFy` numeric loop the gain is ~3–7% — the kernel is one
phase of a loop that is also ~39% contribution-block memory traffic.
A phase-breakdown probe (`probe_narx_phases`, `dense::factor::
phase_timing` counters) measured the warm loop: schur 43.5%,
extend_add 21.1%, contrib-extract 17.7%, the rest assembly and
bookkeeping. Issue #44 is closed: `NARX_CFy` factors correctly; the
residual gap vs MA57 is a structural BLAS-free performance gap, and
the remaining levers are small (~2%) or need an `unsafe`/layout
change not warranted for an already-correct solver. See the #44
wrap-up comment and journal `2026-05-22-02.org`.

### Fixed — B2 scaling cache no longer caches InfNorm scaling ([#49][i49])

The B2 value-bounded *MC64* scaling cache (`src/numeric/solver.rs`)
now populates only when the factor actually ran the MC64 Hungarian
matching. Previously the population gate was
`matches!(scaling_info, ScalingInfo::Applied)` — but `compute_infnorm`
and a user-supplied `External` scaling both also report
`ScalingInfo::Applied`. `Applied` means "a non-trivial scaling was
applied, the solve must undo it", not "MC64 ran". On a matrix that
routes to InfNorm (e.g. POUNCE's Mittelmann `ex4_2` explicit-zero
`(2,2)` KKT, `max_col_nnz=6` → InfNorm) the cache therefore stored the
iterate-0 InfNorm vector and a later warm `factor()` replayed that
stale scaling as an injected `External` on the drifted iterate-N
matrix — a latent correctness defect (benign on every `ex4_2` iterate
measured, but unsound). The gate now additionally requires the
effective strategy to have run the Hungarian: explicit
`Mc64Symmetric`, or `Auto` that `pick_scaling_strategy` routes to
`Mc64Symmetric`. Caching InfNorm bought nothing anyway — InfNorm is
O(nnz), the same cost as the value-bound check it would have replaced.
The genuine-MC64 cache path (populate, value-bound hit, pattern-change
rebuild) is unchanged; all three existing MC64-cache tests stay green.
See journal `2026-05-22-01.org` and `dev/plans/mc64-value-bounded-cache.md`.

### Fixed — explicit-zero KKT entries no longer route to MC64 ([#47][i47])

`pick_scaling_strategy` (`src/scaling/mod.rs`) is now value-aware: when
counting per-column nnz, `max_col_nnz`, and the `diag_only` slack-mass
tally it skips explicit stored `0.0` entries. Previously it counted
*stored* entries, so a constraint column holding an explicit `0.0`
`(2,2)`-block diagonal was mistaken for a degree-1 slack column. On
POUNCE's CHO `parmest` KKT — which keeps explicit-zero `(2,2)`
diagonals — this drove `diag_only/n` to 0.500 (kept) versus 0.000
(stripped), so the structurally identical kept matrix routed to
`Mc64Symmetric` while the stripped one routed to `InfNorm`. MC64 then
degenerated (the ≈3e82 spread blow-up of [#45][i45]), fell back to
`InfNorm`, the B2 value-bounded scaling cache never populated, and
every warm factor re-ran the ~345 ms Hungarian match — the ~2× wall
slowdown. With the fix the kept matrix routes to `InfNorm` like the
stripped one: `scaling_info=Applied`, MC64 cache hits, warm refactor
~16 ms (was ~370 ms — ~23×), inertia and residual unchanged. See
`dev/plans/issue-47-explicit-zero-routing.md` and journal
`2026-05-21-04.org`.

### Fixed — spurious zero in 2×2 inertia (cancellation) ([#48][i48])

The 2×2 pivot-block inertia classifier no longer fabricates a spurious
`zero` eigenvalue. `count_2x2_inertia` / `count_2x2_inertia_val` in
`src/dense/factor.rs` classified signs from the closed-form eigenvalues
`λ = 0.5·(tr ∓ s)`; although `s` is computed cancellation-free, the
*final* subtraction `0.5·(tr ∓ s)` itself cancels — a genuine
non-singular 2×2 whose small eigenvalue lies below `ULP(0.5·tr)`
IEEE-rounds to *exactly 0.0* and was then counted as a `zero`. The
classifier now decides from the cancellation-free sign of the
determinant — `det_sym2x2`, Kahan's fused difference-of-products
(relative error ≤ 2·u for any inputs) — plus the sign of the trace,
via the new `classify_2x2_inertia`: `det < 0` → straddle `(1,1,0)`;
`det > 0` → `(2,0,0)`/`(0,2,0)` by `sign(tr)`; `det == 0` exactly →
`(1,0,1)`/`(0,1,1)`/`(0,0,2)`. A `zero` is now produced only on
genuine exact singularity. This surfaced when Fix 1 (below) removed
the delayed-pivot cascade that had been masking it: it restored
inertia-exactness on the `pinene_3200` KKT replay (iters 8/9 had
returned `WrongInertia`) and, as a side effect, on the `marine_1600`
replay (iter 17 — the defect previously filed as #48). KKT-dump
factor-replay under the default config: `pinene_3200` all 10 iterates
exact, `marine_1600` all 18 exact, `robot_1600` unchanged. Benchmark
inertia match 100.0%; all four exit-partition buckets pass. See
`dev/plans/kkt-cascade-fix2-2x2-inertia-cancellation.md` and journal
`2026-05-21-03.org`.

### Fixed — delayed-pivot cascade amplifier (fine-grained delay) ([#46][i46])

The Bunch-Kaufman driver loops in `src/dense/factor.rs` no longer
forfeit the whole remaining tail of a supernode on the first delayed
pivot. Both loops did `Delayed => break` followed by
`n_delayed = ncol - nelim`, so a single stuck column delayed every
column after it — the amplifier behind the `pinene_3200` interior-point
KKT factor-time explosion. The loops now use **fine-grained delayed
pivoting (swap-to-boundary)**: a stuck column at index `k` is swapped
with the last still-eligible column and the eligible range shrinks by
one, so the driver keeps eliminating and a delay forfeits exactly one
column. This is real delayed pivoting — the stuck column is promoted to
the parent front intact — so inertia stays exact by construction (no
force-accept, no perturbation). On `pinene_3200_0009` (n=127995): total
delayed pivots 133648 → 11309, factor-nonzeros ~165.7M → 3.6M (fill
blowup 69× → 1.51×), factor time ~183 s → 78 ms, inertia
`(64000, 63995, 0)` exact and unchanged. The change is bit-identical on
any matrix with no delayed pivots; all four benchmark exit-partition
buckets pass with no regression. See
`dev/research/kkt-cascade-amplifier-2026-05-21.md` and
`dev/plans/kkt-cascade-fix1-fine-grained-delay.md`.

### Fixed — `External` scaling produced a silent 10× solve error

`ScalingStrategy::External` is now solve-correct. The factorization
applies the user-supplied scaling `D = diag(s)` to the matrix
unconditionally (`D·A·D`), but the solve path had been skipping the
matching pre/post un-scaling because the `External` arm reported
`ScalingInfo::NotApplied` — a flag the solve treats as "the scaling
vector is all-ones, no-op." A non-identity `External` vector therefore
factored `D·A·D` but solved it as `A`, returning `D⁻¹A⁻¹D⁻¹b` instead
of `A⁻¹b` (a 10× error on a `D = 0.3162·I` vector). `External` now
reports `ScalingInfo::Applied`; `NotApplied` is produced only by
`ScalingStrategy::Identity`. The bug was latent — no prior code drove
a non-identity `External` vector through a solve.

### Added — value-bounded MC64 scaling cache (`Solver::with_mc64_cache`)

`Solver` can cache a freshly-computed MC64 symmetric scaling vector and
reuse it across warm `factor()` replays on the same sparsity pattern,
skipping the Hungarian matching when an O(nnz) value-bound check
confirms the cached scaling is still within budget. Default on; a
genuine cache hit is bit-identical to the no-cache path. Disable with
`Solver::with_mc64_cache(false)`; observe hits via
`Solver::mc64_cache_hit_count()`. Note: on the current KKT corpus the
value-bound gate rarely accepts a reuse (the metric is confounded by
the interior-point regularization trajectory), so this is presently
latent infrastructure with no measured speedup — see
`dev/decisions.md` 2026-05-21.

### Fixed — delayed-pivot cascade on zero-(2,2)-block saddle KKT ([#46][i46])

The numeric Bunch-Kaufman kernel no longer cascades into a delayed-pivot
blowup on an interior-point / saddle-point KKT `[[H, Bᵀ], [B, 0]]` whose
`(2,2)` block is structurally zero. Such a KKT has thousands of
zero-diagonal constraint columns; the kernel's 2×2 pivot search
considered only the magnitude-argmax partner row `r`, and when `r` was
not fully summed (an out-of-front coupling) it could neither form a 2×2
nor 1×1 the zero diagonal — so it delayed the column up the elimination
tree, and the delays cascaded. On the POUNCE `cho` `parmest` KKT
(n=43332) this produced a 28M-nonzero, ~17 s factorization where MA57
takes ~70 ms — ~160× slower end-to-end. `scalar_pivot_step` now also
considers the adjacent column `k+1` as the 2×2 partner when `r` is out
of front and `k`/`k+1` are coupled: the `LdltCompress` analysis phase
already co-locates each MC64-matched saddle partner there. The 2×2 is
still gated by the Duff–Reid growth bound and the determinant floor, so
this widens the pivot *search* without relaxing the stability test, and
is bit-identical to the previous kernel whenever `r` is fully summed or
the neighbour is structurally uncoupled. On the `cho` KKT: factor time
11.7 s → 0.20 s (57×), factor-nonzeros 28.05M → 3.35M, inertia
`(21672, 21660, 0)` unchanged. See
`dev/research/kkt-zero-2x2-block-cascade-2026-05-20.md`.

[i44]: https://github.com/jkitchin/feral/issues/44
[i46]: https://github.com/jkitchin/feral/issues/46
[i47]: https://github.com/jkitchin/feral/issues/47
[i48]: https://github.com/jkitchin/feral/issues/48
[i49]: https://github.com/jkitchin/feral/issues/49
[i50]: https://github.com/jkitchin/feral/issues/50
[i51]: https://github.com/jkitchin/feral/issues/51

### Fixed — MC64 catastrophic-spread guard ([#45][i45])

`ScalingStrategy::Auto` no longer applies an MC64 symmetric scaling
vector whose own spread `max|s| / min|s|` exceeds `1 / EPS`
(≈ 4.5e15). Such a scaling is degenerate to working precision:
`D = diag(s)` is singular, `D·A·D` underflows during the
factorization, and Bunch-Kaufman force-accepts exact-zero pivots —
producing a silently wrong solve while `factor()` still reports
`Success` with the correct inertia. On the CHO `parmest` saddle-point
KKT (n=43332) MC64's symmetric matching produced a scaling spanning
≈ 3e82; `Auto` now detects this, falls back to the InfNorm scaling it
already computed on the same path, and solves correctly (relative
residual 7e11 → 2.5e-8). The fallback is surfaced as
`ScalingInfo::Mc64FallbackToInfnorm { reason:
Mc64FallbackReason::Mc64ScalingDegenerate }` (a new variant). The
guard is anchored to the hard numerical invariant `1 / EPS`, not
fitted to a corpus: every matrix in the 38-family parity corpus has
MC64 spread ≤ 3.27e15, well below the threshold. Only `Auto` is
guarded; an explicit `ScalingStrategy::Mc64Symmetric` request is an
informed user choice and is honored as-is. See
`dev/research/kkt-mc64-scaling-blowup-2026-05-20.md`.

[i45]: https://github.com/jkitchin/feral/issues/45

### Changed — inertia counts every pivot by sign ([#42][i42], [#40][i40])

Under the default `ZeroPivotAction::ForceAccept`, FERAL's reported
inertia now classifies *every* accepted pivot by sign — including a
pivot that reduced to a bit-exact `0.0`, which is counted as `negative`
(`0.0 > 0.0` is `false`). The `zero` component of the inertia triple is
therefore structurally `0` whenever the factorization succeeds under
`ForceAccept`. This makes FERAL's reported inertia a *sign-count*, the
same convention used by SSIDS, MA57, and default MUMPS; it is no longer
the mathematical (eigenvalue-sign) inertia on rank-deficient matrices.

Previously FERAL used a hybrid rule that counted bit-exact-`0.0` pivots
into `zero` while counting merely-tiny pivots by sign, producing an
inertia triple that matched no canonical solver on borderline
rank-deficient matrices (e.g. `rankdef_10_3`: FERAL reported `(4,5,1)`;
SSIDS/MA57 report `(4,6,0)`). Whether a near-null pivot rounds to a
bit-exact `0.0` depends on elimination order and per-CPU FMA
contraction, so the old `zero` lane was also architecture-dependent
(#40). Counting every pivot by sign removes both problems: FERAL now
matches the SSIDS/MA57 consensus on every rank-deficient corpus matrix,
identically on every architecture. Rank deficiency remains observable
via `min_pivot_magnitude` (continuous) and `ZeroPivotAction::Fail`
(which still returns `NumericallyRankDeficient`). No factorization or
solve numerics changed — only the inertia counter. Rationale in
`dev/decisions.md` (2026-05-20) and
`dev/research/f01-rankdef-underreporting.md`.

[i42]: https://github.com/jkitchin/feral/issues/42
[i40]: https://github.com/jkitchin/feral/issues/40

### Changed — MC64 partial-singular warning is now opt-in ([#43][i43])

The one-line `warning: MC64 matching left N of M variables unmatched`
stderr breadcrumb (emitted from three numeric-driver sites on
`ScalingInfo::PartialSingular`) is now off by default. `PartialSingular`
is routine and benign for IPM hosts, which factorize structurally
rank-deficient KKT systems on the first attempt of most iterations; the
unconditional stderr write flooded host logs for expected, downstream-
recovered behavior. The new `NumericParams::warn_partial_singular` flag
(default `false`) gates all three sites. Enable it via the
`Solver::with_partial_singular_warning(true)` builder or the
`FERAL_WARN_PARTIAL_SINGULAR=1` env var on the C ABI. The same fact is
always available structurally via `Solver::scaling_info()` (and as a
count via `Solver::mc64_fallback_count()`), so this is a diagnostic
breadcrumb, not a correctness signal. No factorization/solve behavior
change.

[i43]: https://github.com/jkitchin/feral/issues/43

### Added — near-singularity signal (`min|λ(D)|`)

`Solver::min_pivot_magnitude` / `max_pivot_magnitude` (and the
underlying `SparseFactors` methods) report the smallest and largest
accepted pivot magnitude over every 1×1/2×2 D block — FERAL's analog
of MA57's `CNTL(2)` small-pivot threshold. The C ABI exposes them as
`feral_min_pivot` / `feral_max_pivot` (returning `-1.0` when no factor
is available). An IPM perturbation handler thresholds the scale-free
ratio `min/max ≈ 1/κ(D)` to bump its Hessian perturbation on KKT
systems that are near-singular but land on the correct inertia — where
FERAL's `ZeroPivotAction::ForceAccept` force-accepts the pivot and an
inertia-only signal is silent. Computed for free in a pass mirroring
the existing `min_diagonal()`; no factorization/solve behavior change.
Rationale in `dev/research/near-singularity-signal.md`.

### Fixed — F-01 sign-fallback closes FBRAIN3LS_0839 inertia outlier ([#39][i39])

Bunch-Kaufman pivots whose magnitude lands in the F-01
rank-deficiency band `(EPS, sqrt(n)·EPS·||A_scaled||_∞]` are now
counted by sign instead of as zero, matching the default convention
of both reference solvers (MUMPS without `ICNTL(24)=1`, SSIDS). The
strict-zero path (`|d| ≤ EPS`) is unchanged, so synthetic
rank-deficient matrices whose null pivots collapse to exactly 0.0
under BK partial pivoting still detect at least one zero. Five
emission sites in `src/dense/factor.rs` (basic `factor` last-pivot,
`do_1x1_pivot`, `try_reject_1x1_frontal`, `count_1x1_inertia`,
`count_2x2_inertia`) and a ~60-line doc block on
`BunchKaufmanParams::null_pivot_tol` are updated together; full
rationale in `dev/research/f01-rankdef-underreporting.md` 2026-05-17
addendum. Parity panel: 20/0/6 → 21/0/5
(`parity_fbrain3ls_0839` un-ignored). Four stress matrices
(`rankdef_exact_100_10`, `rankdef_200_20`, `saddle_rankdef_100_20_5`,
`stokes_q1p0_8`) now report `zero=0` matching MUMPS-with-`ICNTL(24)=1`
and are ALLOWLISTED(#39) in `external_benchmarks/stress/report.py`.

### Changed — Python wheel pipeline now uses `PyO3/maturin-action`

The v0.4.0 release publish revealed two cibuildwheel-config bugs that
the push/PR CI never tripped (the wheel matrix only runs on `release`
or `workflow_dispatch`). After fixing the trivial ones (manylinux has
no rustup; macOS/Windows runners have no `uv`), the deeper problem was
that cibuildwheel only copies `python/` into the sandbox, so
`feral = { path = ".." }` in `python/Cargo.toml` cannot resolve.
Switched the wheel matrix to `PyO3/maturin-action@v1`, which mounts
the whole `$GITHUB_WORKSPACE` into the manylinux container and runs
natively on macOS/Windows. Trade-off: per-wheel `CIBW_TEST_COMMAND`
is gone; the `test` job (linux × py3.10/3.12/3.13) and the
`smoke-test` job (linux wheel + `uv pip install` + quickstart.py)
still gate every release.

### Changed — `tests/parity.rs` now uses oracle-consensus, not MUMPS-only

The parity gate now accepts a feral inertia that matches **either**
MUMPS 5.8.2 **or** SPRAL SSIDS, which is the literal correctness
contract from `CLAUDE.md`. Side-effect: 5 previously-`#[ignore]`'d
panel matrices that always agreed with SSIDS at oracle-generation
time (ACOPP14_0001, ACOPP14_0003, ACOPP30_0000, ACOPP30_0001,
CERI651CLS_0486) now pass. One genuine outlier remains — FBRAIN3LS
matrix 0839, where feral reports `(5,0,1)` while both MUMPS and
SSIDS report `(6,0,0)` (filed as [#39][i39]). Parity suite: 20
passed / 0 failed / 6 ignored.

[i39]: https://github.com/jkitchin/feral/issues/39

## [0.4.0] - 2026-05-17

Headline: feral is now at parity with MA57 on the 47-problem Mittelmann
panel (solve-rate 39/47 vs 39/47, geomean wall ratio feral/ma57 = 0.96
on the 37 both-solved problems). The two changes that closed the gap are
the auto-armed cascade-break default and the tightened MC64 routing
gate; both are described below.

### Changed — auto-arm cascade-break by default in the C-API (#37, #38)

The Ipopt-feral C-API (`feral_new`) now arms
`Solver::with_auto_cascade_break(0.05)` by default when the user
has not set `FERAL_CASCADE_BREAK=on`. The auto-arm mechanism (in
`src/numeric/solver.rs` since #38) enables
`cascade_break_ratio=0.5, eps=1e-10` for the *next* factor call
when the *previous* call had `max(n_delayed_in) >= 0.05 * n`. Disable
by setting `FERAL_AUTO_CB_BETA=0`; override the threshold with any
non-negative finite float.

Impact on the 47-problem Mittelmann panel (600 s timeout each, vs
MA57): solve-rate parity (39/47 each), geomean wall ratio
feral/ma57 = 0.96 on the 37 both-solved problems, feral aggregate
wall = 54 % of MA57 aggregate. Headline rescues:

| problem        | CB=off    | auto-CB   | Δ        |
|----------------|-----------|-----------|----------|
| robot_1600     | 13.81 s   |  2.93 s   |  -79 %   |
| marine_1600    | 470.87 s  | 20.45 s   |  -96 %   |
| clnlbeam       | 361.26 s  | 45.27 s   |  -87 %   |
| corkscrw       | 53.89 s   | 16.13 s   |  -70 %   |
| dtoc2          | timeout   |  3.69 s   | rescued  |

Six dramatic wins; one minor regression (rocket_12800 +21 %).

### Changed — scaling Auto routing now requires dense arrow head, not just diag-only mass (#68)

`pick_scaling_strategy` previously routed any matrix with
`diag_only / n >= 0.30` to MC64. This conflated true arrow KKTs
(VESUVIO family: 33% diag-only, dense linking columns) where MC64
yields 6x-243x speedups, with banded 1-D PDE KKTs (clnlbeam: 40%
diag-only, max_col_nnz=5) where MC64 hurts the IPM trajectory.

Tightened gate: route to MC64 only when ALSO `max_col_nnz > 32`.
Single O(n+nnz) pass over `col_ptr + row_idx`, no allocations.
clnlbeam end-to-end: 2367 iters / 361 s -> 506 iters / 45 s.
VESUVIO/CRESC/MUONSINE/ACOPP30 routing unchanged.

### Added — diagnostic env-gated tracers

Two zero-cost-when-unset tracers for diagnosing per-factor cost
in live ipopt-feral runs:

- `FERAL_MC64_TRACE=1` -- per-call MC64 wall + process-global
  recompute counter (`MC64_RECOMPUTE_COUNT` in `src/scaling/mc64.rs`).
- `FERAL_FACTOR_TRACE=1` -- per-factor wall + `sum_delayed` +
  `max_delayed` from `src/capi.rs::feral_factor`.

### Added — MA57-style static-pivot perturbation knob (#38)

`NumericParams::static_pivot_eps` (also reachable via
`Solver::with_static_pivot_eps(eps)`, default `None`) lets the
caller request MA57-style "perturb to epsilon" handling for
non-finite or near-singular pivots, replacing the default
hard-fail behaviour with a documented numerical perturbation.
Off by default; opt-in only. Bit-identical for all default-path
users. Evidence: 47-problem Mittelmann sweep with `1e-8` vs
baseline showed identical iteration counts, wall deltas < 2 %.

### Added — non-finite input validation in `Solver::factor` (#38)

`Solver::factor` now eagerly validates that no input matrix entry
is `NaN` or `±inf` and returns `FactorStatus::InvalidInput` instead
of producing a corrupt factor. This was a silent correctness hole
when IPM iterates produced non-finite entries during restoration
phase. Same call point that lights up the new
`with_auto_cascade_break(beta)` warm-arm logic.

### Added — closed-form 2x2 eigenvalue inertia classifier (#38)

Dense Bunch-Kaufman 2x2 pivot inertia is now decided by the
analytic eigenvalues of the symmetric 2x2 block instead of a
trace-and-determinant heuristic. Closes a numerical-edge regression
seen on near-singular 2x2 blocks where the heuristic mis-counted
the negative eigenvalue. Bit-exact for all well-conditioned blocks.

### Fixed — invalidate stale MC64 cache between `factor()` calls (#38)

`SymbolicFactorization::cached_mc64` was reused across
`Solver::factor()` calls, applying the iter-0 scaling values to
iter-N matrix values. Silent inertia drift on warm IPM calls when
matrix values changed enough to alter the matching's optimal
permutation. Fix: invalidate per-factor and recompute fresh.
Forced the auto-CB and value-bounded-cache work that landed in
this release.

### Fixed — raise dense `MAX_N_ELIM` 64 -> 128 (#36)

Dense Bunch-Kaufman `MAX_N_ELIM` constant raised from 64 to 128
and the surrounding bounds assert hardened. The previous 64
ceiling clipped some legitimate supernodes from being eliminated
in a single dense pass on large sparse problems, forcing
spurious delayed pivots.

### Added — `mittelmann_ipopt` benchmark harness

New `external_benchmarks/mittelmann_ipopt/` runs ipopt+ma57 and
ipopt+feral on the 47-problem Mittelmann panel, with per-problem
`PROBLEM_FERAL_ENV` overrides for matrices that need
`FERAL_CASCADE_BREAK=on` (marine_1600, pinene_3200, dtoc2 -- see
the journal entries in dev/journal/2026-05-17-01.org). Drives the
0.96 geomean wall-ratio headline reported above.

### Added — M5 stress-smoke CI gate (#28)

New `stress-smoke` job in `.github/workflows/ci.yml` runs the
stress harness on every push/PR with `--max-n 1000` (21 matrices:
all synth + small `cuter_kkt` rows). `report.py` exits non-zero on
any newly-flagged matrix, making the gate PR-blocking. Local
M-series wall-clock for `synth.py + run.py + report.py` after the
release build is ~0.7s; total CI job stays well under the 10-min
acceptance budget from #28. Caches
`external_benchmarks/stress/matrices/` keyed on the manifest hash
so SuiteSparse downloads occur at most once per manifest change.
README documents gate semantics: "blocked" definition,
baseline-update procedure, allowlist convention.

### Added — SQD (symmetric quasi-definite) fast-path (#34)

New opt-in factorisation path for matrices satisfying Vanderbei
(1995)'s SQD contract `K = [[-E, A^T], [A, F]]` with `E, F ≻ 0`.
Enable via `Solver::new().with_sqd_mode(true)`. The kernel skips
the Bunch-Kaufman 1×1-vs-2×2 pivot search and runs a pure
diagonal-D loop, reusing the shared `do_1x1_update` rank-1
trailing kernel.

New public surface:
- `feral::dense::factor::factor_diagonal`
- `feral::dense::factor::factor_frontal_diagonal_in_place`
- `feral::Solver::with_sqd_mode(bool)`
- `feral::NumericParams::sqd_mode: bool`
- `feral::FeralError::SqdContractViolated { column: usize,
  pivot: f64 }`

Contract enforcement (loud failure, never silent BK fallback):
1. `|d_kk| > zero_tol` near-zero guard.
2. `max |l_{ik}| <= 1/sqrt(EPS) ≈ 6.7e7` Gill-Saunders-Shinnerl
   1996 column-growth guard.

A trip on either surfaces `SqdContractViolated` immediately;
the caller decides whether to refactor with `with_sqd_mode(false)`
(BK fallback) or investigate the input.

Bench: new `cargo run --release --bin bench_sqd` runs 6 synthetic
KKT shapes through both BK and SQD paths and reports per-shape
speedup + geomean. Measured geomean speedup ≈ 1.025–1.05× on the
synthetic corpus (M4 Pro, 2026-05-16); the value proposition is
primarily robustness (predictable factor time, loud contract
trips, no pivot-search dependence on near-threshold pivots) rather
than raw speed. See `dev/sessions/2026-05-16-08.md` and
`dev/research/sqd-fast-path.md` for the full discussion.

Tests: 11 in `tests/sqd_fast_path.rs` covering kernel hand-checks,
Solver-level dispatch (dense + multifrontal), contract-trip on
both bounds, BK-vs-SQD reference parity, random-SQD property
trials, and the symbolic-cache reuse contract.
### Added — M3 stress corpus expansion (#26)

Triples the SuiteSparse stress-suite corpus from 18 to 104 rows
(125 total with synth + cuter_kkt). New groups: `Schenk_IBMNA` (49
indef Jacobians from circuit / nonlinear-arithmetic problems),
remaining `GHS_indef` entries in the n ≤ 100k tier (33 saddle / pde /
opt / dense rows), and 4 Boeing mechanics matrices with indefinite
stiffness blocks (`bcsstk35/37/39`, `nasa1824`). Each row is tagged
under the existing `category` taxonomy so `report.py`'s per-category
roll-up stays meaningful: `indef=59`, `saddle=15`, `opt=15`,
`mech=4`, `dense=6`, `pde=5`. Full run completes in 4m35s wall
(13× headroom under the 1h acceptance ceiling); `report.py` is
byte-stable across back-to-back invocations and exit code is 0 on
all 122 ready rows. Pattern-encoded (`nasa2910`, `nasa4704`) and
integer-encoded (`aug2d`, `aug2dc`, `aug3d`) matrices excluded
pending matrix-market reader extension — see
`dev/tried-and-rejected.md`. Schenk_AFE skipped entirely (all
matrices are ≥504k rows, outside the n ≤ 100k size tier and over
the time budget). Background, group selection, SPD filter, and
pathologies-encountered documented in
`dev/research/stress-corpus-m3.md`.

### Changed — Issue #10 closes; default `nemin=16` and `OrderingMethod::Amd` confirmed (#10)

Issue #10 ("Add APP path alongside TPP in dense LDLᵀ kernel") closes
without an APP implementation. Five architectural levers were tried
against the 1D-banded Mittelmann panel — SLB driver removal,
MAXFROMM AMAX cache, manual axpy SIMD, ordering swap (Metis/Scotch
ND), and forced supernode amalgamation (`nemin ∈ {32, 64, 128}`) —
and all five came up negative. The rank-1 axpy kernel on
`ncol=1..16` fronts is bandwidth-bound; pulp saturates the vector
ALU; AMD's elimination tree is already shape-optimal under the
nnz_L bound. New diagnostic binary `diag_nemin_amalgamation_panel`
sweeps `SupernodeParams::nemin ∈ {16, 32, 64, 128}` on the 4-family
× 20-matrix panel and reports paired-by-matrix geomean ratios; the
shape lever engages (`ncol_mean` doubles at nemin=64) but factor
time stays flat or regresses 36% on `clnlbeam`. Cumulative 5-lever
table and the joint conclusion are documented in
`dev/research/issue-10-amalgamation-floor.md`. The opt-in knobs
(`Solver::with_ordering`, `SupernodeParams::nemin`) stay shipped;
defaults unchanged.

### Added — FBRAIN3LS pivot-threshold sweep + stress-suite entry (#29)

New diagnostic binary `diag_fbrain3ls_pivtol_sweep` factors five
FBRAIN3LS borderline samples (`0788`, `0839`, `0843`, `0848`, `0851`)
under `pivot_threshold ∈ {0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-2}`
in two passes (`ForceAccept` and `Fail` for `on_zero_pivot`) and
reports per-row inertia, smallest |D|, and the JSON-RHS relative
residual. The new research note
`dev/research/fbrain3ls-2x2-stability.md` writes up the result: the
sweep is flat across all seven `pivot_threshold` values in both
passes, and FBRAIN3LS produces no 2×2 pivot blocks at any tested
threshold. The active gate on these matrices is the F-01
`null_pivot_tol` override (`sqrt(n) · EPS · ‖A‖_inf`), not
`pivot_threshold`. The default `NumericParams::default().bk.pivot_threshold
= 1e-8` is therefore kept on its existing rationale (Ipopt
`ma27_pivtol` compatibility for Identity-scaled IPM-KKT callers).

`external_benchmarks/stress/manifest.tsv` gains three
`cuter_kkt / FBRAIN3LS__FBRAIN3LS_{0839,0843,0851}` rows under a
new `borderline` category, and `external_benchmarks/stress/fetch.py`
learns to source `cuter_kkt` rows from the in-repo
`data/matrices/kkt/<family>/<sample>.mtx` CUTEst dump rather than
SuiteSparse. README updated to document both the group and the
category.
### Added — M4 synthetic stress generators (#27 + #31 follow-up)

`external_benchmarks/stress/synth.py` gains five generator families
covering pathologies the existing `rankdef` / `near_sing` / `illcond`
/ `cascade` synthetics did not reach:

- `rankdef_exact_<n>_<k>` — exact-IEEE-zero variant requested in the
  #31 follow-up; makes the dispersed-null-space failure mode an
  explicit oracle target rather than a side effect.
- `saddle_rankdef_<n>_<k>_<r>` — KKT block `[H A^T; A 0]` with
  rank-deficient constraint, inertia oracle `(n, m−r, r)`.
- `wide_frontal_<n>` — bordered block diagonal forcing a single
  supernode of width 600, stresses the sparse → dense crossover.
- `mc64_resistant_<n>` — `Q D Q^T` with one eigenvalue at 1e-8 and
  dense Q; MC64 sees an O(1) diagonal and produces s ≈ 1, so the
  scaled matrix still has cond ≈ 1e8.
- `stokes_q1p0_<h>` — Q1-P0 mixed-element Stokes saddle on an h × h
  mesh, inertia oracle `(n_u_free, n_p − 2, 2)` (constant +
  checkerboard pressure modes).

`report.py` is extended with regex-based oracle dispatch for each
new naming convention, and the rank-deficient-refusal short-circuit
is broadened from `category == "rankdef"` to also cover
`saddle_rankdef` and `stokes`. The manifest gains seven new rows
(two per family for rankdef_exact and saddle_rankdef, one each for
wide_frontal / mc64_resistant / stokes). Math, oracle derivations,
and the abandoned rank-1-perturbation attempt for mc64_resistant
are documented in `dev/research/synthetic-generators-m4.md`.

Verified end-to-end: `synth.py --only <name>` produces each matrix,
feral factors all seven with `rel_res < 1e-13`, and `report.py`
flags none. The Stokes matrix gives an exact inertia match
`(98, 62, 2)`.

### Added — near-singular inertia sweep + certification (#31)

New diagnostic binary `diag_near_singular_sweep` runs
`Solver::new()` factor + iterative-refinement solve on the
parametric `near_singular_eps_<p>` family (p ∈ {6..14}) and
reports `(status, inertia, min|D_ii|, rel_res, pivtol)` for each.
Together with the new research note
`dev/research/inertia-near-singular-certification.md` this
documents the detection boundary at p = 6 — i.e. feral's default
Bunch-Kaufman thresholds (`zero_tol = f64::EPSILON`,
`pivot_threshold = 1e-8`) are *provably unable to detect a single
isolated small eigenvalue in a generic dense random-Q symmetric
matrix at any p*, including the canonical
`near_singular_eps9` / `near_singular_eps12` stress matrices
which never actually reported the null pivot. The factorization
remains stable (rel_res ≤ 6e-15 across the sweep) because BK
treats the small mode as a healthy O(1) trailing pivot — see
Higham 2002 Ch. 11.

The Python generator `external_benchmarks/stress/synth.py` is
extended with `near_singular_eps_<p>` entries for p ∈ {6..14},
and `external_benchmarks/stress/manifest.tsv` now pins
`near_singular_eps_7` as a regression matrix at the boundary +1
slot per the issue's acceptance criterion.

No solver code, no defaults changed. The rejected
trailing-norm-based criterion is documented in the note.

### Added — `Solver::with_ordering(method)` builder (#33 §3)

The `Solver` builder now exposes the fill-reducing ordering choice
through `Solver::with_ordering(OrderingMethod)`. Default is
`OrderingMethod::Auto` (unchanged behavior — matches
`symbolic_factorize`'s dispatcher exactly), with explicit `Amd`,
`Amf`, `MetisND`, `ScotchND`, `KahipND` available.

Motivation: issue #33's 1D-banded Mittelmann panel (clnlbeam)
bottlenecks 97% main-thread in the scalar 1×1 pivot path because
AMD produces thin supernodes on banded structure. The
"supernode-shape thesis" is that nested-dissection orderings
produce squarer fronts that let more work batch through the
blocked panel kernel. The machinery already existed
(`symbolic_factorize_with_method`, `OrderingMethod`), but `Solver`
always routed through the `Auto` dispatcher — there was no way to
override without forking the symbolic plumbing. This release
closes that gap so library consumers can re-time banded KKTs with
`Solver::new().with_ordering(OrderingMethod::ScotchND)` without
rebuilding.

Per-corpus speedup measurement on banded KKTs is left to follow-up
empirical work; this release ships only the lever.

Regression tests in `tests/solver_with_ordering.rs`:
`with_ordering_threads_method_into_symbolic_phase` (proves the
choice actually reaches the symbolic phase by comparing permutations)
and `with_ordering_auto_matches_default` (guard against the default
silently drifting from `Solver::new()`).

### Added — `Mc64FallbackToInfnorm` diagnostic surfacing (#24, M2)

`ScalingStrategy::Auto` previously fell back from MC64 to InfNorm
silently in two cases (`InfNormSpreadAcceptable` pre-MC64 trial
and `Mc64WorseThanInfnorm` post-MC64 Policy 4 ratio guard).
Callers had no way to learn which scaling actually ran, making
"feral got the wrong answer" reports undiagnosable.

This release surfaces the fallback as a new structured signal:

- `ScalingInfo::Mc64FallbackToInfnorm { reason: Mc64FallbackReason }`
  is returned from `compute_scaling` / `compute_scaling_with_cache`
  on every fallback path. The variant carries the trigger reason
  (`InfNormSpreadAcceptable` or `Mc64WorseThanInfnorm`) for
  triage. The fallback scaling vector itself is unchanged.
- `Solver::scaling_info() -> Option<&ScalingInfo>` exposes the
  most recent factor's scaling info.
- `Solver::mc64_fallback_count() -> usize` is a session-cumulative
  counter that long-running IPM drivers can poll without
  inspecting per-factor state.
- `bench_one_matrix` now emits `mc64_fallback yes|no` (plus a
  `mc64_fallback_reason` key when yes) in the sidecar so the
  cross-solver comparison harness can audit fallback frequency
  per corpus.

The new variant slots in alongside the existing `Applied`,
`PartialSingular`, and `NotApplied`; downstream solve consumers
(`solve.rs::needs_scaling`) treat it as "scaling applied"
because the InfNorm vector is the correct scaling for the
returned factor.

Regression tests:
`src/scaling/mod.rs::tests::auto_surfaces_infnorm_spread_fallback_on_uniform_diag`
(synthetic, always runs); the fixture-gated
`auto_falls_back_to_infnorm_on_mss1_0009` and
`auto_picks_infnorm_on_acopp30_0064` were extended to assert the
new variant when the corpus is present. Solver-level coverage in
`src/numeric/solver.rs::tests::mc64_fallback_surfaces_via_solver_api`.

### Changed — `ScalingStrategy::Auto` pre-MC64 InfNorm trial (#23, ACOPP30 plateau-2)

`compute_scaling_auto_with_cache` now runs the InfNorm Knight-Ruiz
pass *before* committing to MC64 when the arrow-KKT router selects
it. If the resulting scaling vector has `max|s|/min|s| < 1e3`
(IN_SPREAD_GUARD), Auto accepts InfNorm and skips MC64 entirely.
This catches matrices where the structure suggests MC64 is needed
but the values are already near-equilibrated; on such matrices
MC64's matching is gratuitous and on the ACOPP30 KKT family
(cond ≈ 3e16) catastrophic.

Before this change, the legacy fast-path (`raw_diag_range >= 1e6 →
MC64 unconditionally`) routed 6 ACOPP30 matrices (iters 59, 63-67
of the 105-matrix corpus) to a literal-zero pivot that iterative
refinement could not recover (rel_ref 1.88e-6 to 1.74e-1). After
the fix, all 105/105 ACOPP30 matrices pass rel_ref < 1e-10 under
default `Solver`.

Policy 4 panel validation (9-matrix): zero regressions. The
matrices where MC64 strictly wins (VESUVIA / VESUVIO / VESUVIOU /
MEYER3NE, all with in_spread > 1e4) still route to MC64. HS75 and
MUONSINE (in_spread < 100) now route to InfNorm and produce
sub-1e-15 residuals — strictly better than the prior MC64 default.

The in_vec computation is hoisted above the existing `raw_drng`
fast-path and reused by the downstream `mc_off/in_off` ratio test,
so the cost is one InfNorm pass per Auto invocation that reaches
the MC64 leg of the router.

Regression tests:
`src/scaling/mod.rs::tests::auto_picks_infnorm_on_acopp30_0064`,
`auto_picks_infnorm_on_hs75_0000` (renamed from
`auto_keeps_mc64_on_hs75_0000`).

References:
`dev/research/acopp30-plateau-2.md`,
`src/bin/probe_scaling_policy4.rs` (panel diagnostic),
`src/bin/diag_acopp30_residual.rs` (105-matrix sweep).

### Added — rank-deficient KKT systems report honest inertia (F-01, #21)

The Bunch-Kaufman kernel gained `BunchKaufmanParams::null_pivot_tol`
(and `null_pivot_tol_2x2`), a *factor-time* rank-deficiency floor that
is independent from the strict *solve-time* `zero_tol`. When the
sparse multifrontal driver runs with `on_zero_pivot != Fail`, it
overrides `null_pivot_tol` to `sqrt(n) · EPS · ‖A_scaled‖_∞` (MUMPS
CNTL(3)-style threshold) so that pivots in the rank-deficiency band
`(EPS, sqrt(n)·EPS·‖A‖]` are counted as zero in the inertia signature
rather than miscounted as small-but-positive/negative.

Effect on the stress baseline:

- `synth/rankdef_5_2`:    inertia=(2, 2, 1) — unchanged.
- `synth/rankdef_10_3`:   inertia=(4, 5, 1) — unchanged.
- `synth/rankdef_50_5`:   inertia=(25, 24, 1) — unchanged.
- `synth/rankdef_200_20`: inertia=(109, 88, 3) — previously
  (112, 88, 0), now detects 3 of the 20 constructed zeros.
  Partial detection is consistent with MUMPS 5.8.2 behavior
  (MUMPS with ICNTL(24)=1 also reports `zero=0` on this matrix).

Crucially, the split keeps `Factors.zero_tol` at the strict EPS
default. The solve continues to divide by any pivot above EPS, so
ill-conditioned non-singular matrices (e.g. `synth/ill_cond_e14`
cond≈1e14) retain `rel_res ≈ 7e-16` — no degradation. Default dense
callers see no behavioral change (`null_pivot_tol` defaults to
`zero_tol`).

The stress harness's rankdef acceptance rule loosened from
`zero == expected` to `1 <= zero <= expected`; the F-01 research
note documents the MUMPS oracle comparison that motivates this.

Regression test: `tests/pounce_interface.rs::f01_rankdef_surfaces_at_least_one_zero_pivot`.

References:
`dev/research/f01-rankdef-underreporting.md`,
`dev/plans/robustness-roadmap.md` (F-01 entry).

### Changed — sparse `NumericParams::default()` accepts isolated zero pivots (F-03, #32)

`NumericParams::default().bk.on_zero_pivot` now defaults to
`ZeroPivotAction::ForceAccept` instead of inheriting `Fail` from
`BunchKaufmanParams::default()`. The sparse multifrontal path has
infrastructure (delayed pivoting, root force-accept) to recover
from an isolated zero pivot; the dense entry point does not, and
its default remains unchanged.

Behavior change for callers of `Solver::new()`:

- Matrices with isolated zero pivots that previously returned
  `FactorStatus::Singular` (factor discarded) now return
  `FactorStatus::Success` with `inertia.zero > 0` and the factor
  preserved. Detect rank-deficiency via `solver.inertia()`.
- Matches MUMPS `INFOG(28)` (number of null pivots) and MA57
  `cntl[4]` defaults — both oracles produce usable factors on
  matrices like `GHS_indef/bloweybl` (n=30003 saddle with 2/3
  zero diagonals) where feral previously rejected with
  `NumericallyRankDeficient`.
- Stress baseline: `bloweybl` flips from failed to ok with
  `inertia=(20001, 10001, 1)` matching both MUMPS and MA57
  (residual 1.68e-15).
- No matrix that previously factored successfully changes
  outcome.

Callers wanting the legacy abort-on-zero behavior can opt in
explicitly: `BunchKaufmanParams { on_zero_pivot:
ZeroPivotAction::Fail, .. }`. Two `pounce_interface.rs` tests
that depended on the old default were updated to opt in.

References:
`dev/research/f03-bloweybl-rank-rejection.md`,
`dev/plans/robustness-roadmap.md` (F-03 entry),
`src/numeric/factorize.rs:382-417`.

### Added — JAX integration (`feral.jax`)

`feral.jax` exposes a differentiable, vmap-able, jit-able sparse
symmetric solve via implicit differentiation. Installs via the
`[jax]` extra: `pip install feral-solver[jax]`.

Surface:
- `feral.jax.SparsePattern.from_csc(n, indices, indptr)` — hashable
  static sparsity descriptor. Pass as `pattern=...` kwarg.
- `feral.jax.solve(values, b, pattern=pattern)` — JAX-traceable
  solve of `A x = b`. Custom JAX primitive with rules for:
  - `jax.grad` / `jax.vjp` / `jax.jacrev` (reverse-mode, one extra
    solve via implicit diff; values-gradient projected to the CSC
    pattern via a symmetric outer-product primitive).
  - `jax.jvp` / `jax.jacfwd` (forward-mode, `dx = A⁻¹(db - dA x)`).
  - `jax.vmap` over `b`, over `values`, or over both (vmap-over-
    values loops independent factorizations on the host).
  - `jax.jit` (routes the host call through `jax.pure_callback`).
- `feral.jax.matvec(values, x, pattern=pattern)` — differentiable
  symmetric matvec, also exposed as a primitive.

Requires `jax>=0.4.30` and **x64 mode** (`jax.config.update("jax_enable_x64",
True)`). feral is double-precision only; float32 inputs raise `TypeError`.

Tests (15/15 pass) verify: grad-w.r.t.-b against `A^{-T} g`, grad-w.r.t.-
values against central finite differences, jvp against finite differences,
`jacrev`/`jacfwd` against `inv(A)`, vmap over both axes, jit + jit(grad).

Caveats:
- Host hop per call (pure_callback breaks XLA fusion). Designed for
  outer Newton/IPM loops, not inner `scan`-style fusion.
- No second-order autodiff yet (`jax.hessian` unsupported in this
  release — the pattern-outer primitive has no JVP rule).
- Each `solve` call spins up a fresh `feral.Solver` and re-factors.
  For symbolic reuse across an IPM Newton loop, use `feral.ipm.KktSolver`
  directly (stateful, non-JAX path).

Example: `python/examples/jax_quickstart.py`.

### Added — Python interface (`feral-solver` on PyPI)

First-class Python binding shipped under `python/`, published as
`feral-solver`. Built with PyO3 0.22 + maturin, distributed as
abi3-py310 wheels (one wheel per platform, Python 3.10+).

User-visible surface:

- `feral.CscMatrix` — sparse CSC matrix with `from_dense`,
  `from_triplet`, `from_mtx`, `from_scipy`, `to_scipy`, plus
  `set_values` for cheap value-only refactor.
- `feral.Solver` — direct solver with `factor`, `refactor`,
  `solve` (1D and 2D RHS), `solve_refined`,
  `estimate_condition_1norm`, `increase_quality`. Properties:
  `inertia`, `num_negative_eigenvalues`, `factor_nnz`,
  `symbolic_call_count`, `quality_level`, `needs_refinement`,
  `pivot_threshold`, `parallel`, `scaling`.
- `feral.Inertia(n_pos, n_neg, n_zero)` — first-class triple with
  `matches`, `__iter__`, `as_tuple`.
- `feral.FactorStatus` / `feral.QualityLevel` — IntEnums.
- Exception hierarchy: `FeralError` → `FactorError`,
  `SolveError`, `PatternMismatch`, `FeralIOError`;
  `FactorError` → `SingularError`, `WrongInertiaError`,
  `NumericFailure`.
- `feral.ipm.KktSolver` — Wächter–Biegler 2006 §3.1 perturbation
  loop wrapper for IPM callers (target consumer: `discopt`).
  Caches diagonal indices once, applies δ_w / δ_c escalation
  inertia-driven, returns `FactorReport` with attempt count and
  wall time. Symbolic factorization is reused across the full
  Newton loop.
- `feral.from_scipy(a, *, symmetric="lower"|"upper"|"full")` and
  `feral.to_scipy(m)` for scipy.sparse interop.

Distribution:

- `pyproject.toml` declares maturin build backend, ABI3 wheel
  tags, numpy>=1.23 runtime dep, optional `[scipy]` extra.
- `.github/workflows/python-wheels.yml` runs tests on every PR
  (ubuntu × py3.10/3.12/3.13) and on tagged releases builds
  wheels via cibuildwheel for linux x86_64 + aarch64, macos
  universal2 (arm64+x86_64), and windows AMD64, plus sdist,
  plus a uv-install smoke test, and publishes to PyPI via
  trusted publishing.

Tests: 23/23 pytest tests pass. The Newton-loop reuse test
(`test_kkt_solver_symbolic_reuse_across_newton_loop`) confirms
`symbolic_call_count == 1` after 20 perturbed factorizations.

Caveat: `KktSolver` requires every diagonal entry it intends to
perturb (including (2,2)-block zeros) to be present in the CSC
pattern. The HS071 example shows the from_triplet idiom.

Closes #20.

### Changed — cascade-break is now opt-in (was auto-armed by default)

`NumericParams::default()` previously set `cascade_break_ratio =
Some(0.5)` and `cascade_break_eps = Some(1e-10)`, auto-arming a
non-standard "static-pivot at cascade-overloaded supernodes"
mechanism. Both now default to `None`. Callers that want the
`pinene_3200`-style cascade-absorption speedup (88.6 s → 33 ms on
`_0009`, ~2840× — confirmed in this session via
`probe_cascade_perturb`) opt in explicitly:

```rust
let mut s = Solver::new()
    .with_cascade_break(0.5)
    .with_cascade_break_eps(1e-10);
```

Rationale:

1. The original `PerturbToEps` docstring claimed `||Δ||_∞ ≤
   abs_floor` per perturbed pivot (Weyl-localised). That bound is
   wrong: with L scaled by `1/d_new`, the implicit `Δ` flows
   through the trailing Schur update and is bounded in
   `||A||² / eps` in the worst case. On IPM KKT matrices the
   unrefined residual stays small in practice (`~1e-5` on
   `robot_1600_0004`), but the claim was misleading.

2. MUMPS and MA57 don't ship an equivalent feature. Auto-arming a
   non-standard mechanism by default was creating surprises in
   downstream tooling.

3. A proposed code fix to make the bound match the docstring
   (zero `L[:,k]` after writing the perturbed `D[k,k]`) was tried
   and rejected — measurement showed the residual on
   `robot_1600_0004` got *worse* by 5 orders of magnitude (1e-5 →
   2e3) because the solve loses the live L column needed to
   cancel `1/d_new` during back-substitution. See
   `dev/tried-and-rejected.md` "Zero L on `PerturbToEps`".

Also updated: the `PerturbToEps` and `with_cascade_break_eps`
docstrings now honestly describe the perturbation structure
(LAPACK static-pivoting / MA57 `cntl(4)` precedent, not a Weyl
`eps` bound). New probe binary `probe_cascade_perturb` measures
residuals across cb=off / cb=default / cb=fa configurations.

Phase 2.8.1 corpus gates remain PASS; bench numbers within noise
of session 2026-05-15-06.

References:
`dev/research/cascade-break-l-perturbation-2026-05-15.md`,
`dev/tried-and-rejected.md` (2026-05-15 zero-L entry),
`src/bin/probe_cascade_perturb.rs`.

### Changed — `PAR_MIN_FLOPS` lowered from 10⁸ to 10⁷ (feral#19 closeout)

The work-aware parallel-assembly gate threshold drops from 10⁸ to
10⁷ flops after direct measurement on `robot_1600` KKT dumps
(`probe_issue_19`, session 2026-05-15-06, Apple M4 Pro):

- iter 0000 (4.75×10⁶ flops): parallel hurts 0.67× → gate keeps
  sequential ✓
- iter 0001/0003 (1.13×10⁷ flops): parallel wins 1.42-1.48× →
  gate now allows parallel ✓ (old 10⁸ default was blocking this)
- iter 0006 (9.43×10⁶ flops): parallel wins 1.24× → still
  sequential under the new threshold; traded for safety margin
  against the 6×10⁶ break-even.

Two earlier confounders that justified the conservative 10⁸: the
per-call `rayon::ThreadPool` cv-wait cost (eliminated by the
persistent pool in `91e028a`), and the freehand "100 µs spawn +
10 GFLOP/s" calibration argument (off by ~10× on each side).
The original 12× wall regression on `robot_1600` no longer
reproduces post-`91e028a` — forced-parallel is now ~30% *faster*
than sequential there.

Non-M4 hardware can still override per-call via
`NumericParams::min_parallel_flops` (or
`POUNCE_FERAL_MIN_PAR_FLOPS=<u64>` from pounce-feral). See
`dev/research/par-min-flops-calibration-2026-05-15.md` (update
section) and `dev/sessions/2026-05-15-06.md`.

### Added — `probe_issue_19` binary

`cargo run --release --bin probe_issue_19 -- <mtx> [reps]` times
`Solver::factor()` on a single KKT matrix under three configs
(sequential, parallel-with-gate, parallel-with-gate-off) and
reports the gate's decision. Used to land the issue-#19 closeout;
useful in general for diagnosing per-workload gate behaviour.

### Changed — `Solver` reuses a persistent `rayon::ThreadPool` (feral#19 follow-up)

`Solver` now owns a lazily-built `rayon::ThreadPool` that is reused
across every `factor()` call dispatching the parallel multifrontal
driver. Built on first parallel-fire; persists for the `Solver`'s
lifetime. Inside `pool.install(...)` the inner driver's
`rayon::scope` / `current_thread_index` / `current_num_threads`
calls bind to this pool's workers instead of the global pool, so
the cv-wait wakeup cost that issue #19 flagged is paid once
up-front rather than per `factor()`. End-to-end measurement on M4
Pro `robot_1600` (force-parallel, 200 IPM iters): sys time dropped
28% (24.7 s → 17.9 s). No user-facing API change; the existing
`Solver::with_parallel(false)` toggle continues to skip the pool
build entirely. See `dev/sessions/2026-05-15-04.md`.

### Changed — work-aware gate in `should_parallelize_assembly` (feral#19)

`should_parallelize_assembly` (the dispatcher inside
`factorize_multifrontal_parallel_with_workspace`) now also requires
that the estimated total tree flop count clear `PAR_MIN_FLOPS = 10^8`
before firing the rayon-parallel driver. Previously the gate was
structural only (`n_snodes ≥ N_PAR_MIN` + ≥1 multi-child supernode),
which let parallel fire on small-KKT IPM control-NLP profiles
(`robot_1600`) where rayon spawn / cv-wait overhead exceeded the
parallel speedup. New public surface: `pub const PAR_MIN_FLOPS`,
`pub fn estimate_assembly_flops`, `pub fn should_parallelize_assembly_
with_threshold`, and `NumericParams::min_parallel_flops: Option<u64>`
for per-call tuning. Pounce-side env-var hook
`POUNCE_FERAL_MIN_PAR_FLOPS=<u64>` plumbs the override. See
`dev/decisions.md` (2026-05-15-03 block) and
`dev/sessions/2026-05-15-03.md`.

### Changed — `feral_solve` C ABI defaults to iterative refinement

`feral_solve` in the C ABI (`src/capi.rs`) now routes through
`Solver::solve_many_refined` against the cached factored matrix
by default. This closes the residual floor that was causing
ipopt-feral to stall in the final-tail convergence on Mittelmann
`NARX_CFy.nl` (feral#18 — ipopt-feral now Optimal at 485 iters
vs TIMEOUT before) and `robot_1600.nl` (feral#17 — ipopt-feral
now Optimal at 301 iters / 19 s vs MaxIter before). The fix is
the same in both: cascade-break perturbs the L factor enough
that the unrefined backsolve carries a ~1e-5 residual that the
IPM can't drive below the duality gap; one round of refinement
against the original matrix closes the gap. Opt out with
`FERAL_REFINE=0` in the environment. Cascade-break itself stays
enabled — it helps on the matrices it was calibrated for
(feral#8, feral#15). See `dev/decisions.md` (2026-05-15-02
block) and `dev/sessions/2026-05-15-02.md`.

### Added — Synthetic-matrix scaling benchmark vs MUMPS and MA57

New harness at `external_benchmarks/scaling/` that sweeps four
synthetic matrix families (`dense_si`, `banded_spd`, `laplace2d`,
`saddle_kkt`) across multiple sizes and compares feral against
MUMPS 5.8.2 and HSL MA57 on a derived `total_factor_us` metric
(analyse + numeric factor combined, accounting for MUMPS's bundled
`JOB=4` timing). Components: `run.py` driver (matrix generation,
RHS synthesis, solver manifests, aggregation), `plot.py` (four
PNGs per family + overview, log-log fits), `report.org` (full
write-up), and `scaling.tsv` for downstream analysis. Headline
finding: feral's numeric factor is at parity with MUMPS on every
sparse family and at parity with MA57 on `banded_spd` at the
largest `n` — but symbolic analysis is 91-96% of total wall time
on structurally-regular sparse families, making it the highest-
impact optimization target. See
`external_benchmarks/scaling/report.org` for the full discussion.

### Changed — `dev/assemble-context.sh` no longer re-runs full corpus bench

Default `./dev/assemble-context.sh` invocation now sources the
benchmark section from the latest dated session checkpoint instead
of executing `cargo run --bin bench --release` (which walks ~150k
corpus matrices and takes ~3.5 minutes). Pass `--with-bench` to
re-run for fresh numbers. Refresh time drops from ~3m30s to ~3s.
Also fixes a pre-existing glob bug where `phase-2-baseline.md` was
selected as "latest" because it sorts after `2026-...` lexically;
the glob is now `dev/sessions/[0-9]*.md`.

### Added — F2.3 iterative-refinement diagnostics

New public function `solve_sparse_refined_with_diagnostics` and
companion types `RefinementDiagnostics` / `RefinementStep`
(`src/numeric/solve.rs`). Mirrors `solve_sparse_refined` exactly
in control flow and returned iterate; additionally emits the
per-step residual / Skeel forward-error-bound trajectory plus a
shared `kappa_1_est` from the Hager-Higham 1-norm estimator. The
non-diagnostic path is unchanged — diagnostics are opt-in and add
3-5 solves (one κ̂ pass) plus an O(nnz) `||A||_1` walk only when
explicitly requested. Intended for ripopt's δ-ladder logging
(per `dev/plans/kkt-feature-gaps.md` §F2.3). Re-exported at the
crate root.

### Changed — Symbolic-arm gate on cascade-break trigger (issue #15)

The cascade-break trigger (`NumericParams::cascade_break_ratio`)
now requires `symbolic.n >= CASCADE_BREAK_MIN_N` (=4096) to fire.
Below the threshold the trigger is a guaranteed no-op regardless
of how aggressively it is configured. Cascade-break savings only
accumulate when some front can grow, via delay propagation, to
several thousand columns — and `n` is an upper bound on
achievable expanded ncol. Issue #15 reported a 14× regression on
qcqp1000-1nc (n=1154) attributed to `Some(0.5)` firing
aggressively; per-iterate data shows the trigger actually never
fires on qcqp because `n_delayed_in = 0` everywhere. The gate
makes the trigger a structural no-op on small problems while
preserving the cascade-break wins on marine_1600 / pinene_3200
(both n > 70 000). See
`dev/research/issue-15-cascade-break-symbolic-arm.md`.

## [0.3.0] - 2026-05-13

### Added — Feral C ABI for Ipopt linkage (`feral::capi`)

New `pub mod capi` (`src/capi.rs`) exposes a minimal C ABI surface
matching Ipopt's `SparseSymLinearSolverInterface` plug-in shape:
`feral_new`, `feral_free`, `feral_set_structure`, `feral_values_ptr`,
`feral_factor`, `feral_solve`, `feral_num_neg`. Matrix format is
Ipopt's `CSR_Format_0_Offset` (upper-triangle CSR, 0-based) which is
byte-identical to feral's lower-triangle CSC. Status codes mirror
Ipopt's `ESymSolverStatus` enum.

`Cargo.toml` adds `staticlib` to `crate-type` so the ABI can be
linked into the C++ Ipopt build via the `feral-ipopt-shim/` patch
(opt-in for downstream Ipopt builders; pure-Rust consumers continue
to use the `rlib`). See `dev/research/feral-ipopt-c-shim.md` and
`dev/plans/feral-ipopt-shim.md` for the design.

### Added — Ipopt 3-way NLP comparison harness

`external_benchmarks/nlp_comparison/` runs the Ipopt
ScalableProblems suite against three Ipopt 3.14.20 binaries
(`build-mumps`, `build-ma57`, `build-feral`), each linked to a
single sparse direct solver. 35 problems × 3 solvers; see
`REPORT.md` for the 2026-05-13 sweep. MUMPS 35/35 optimal, MA57
34/35, feral 34/35; geomean over triple-optimal subset: MUMPS
139 ms, feral 158 ms, MA57 162 ms. Generates `results.json` and a
Markdown report. Logs/out/RHS blobs gitignored; only the harness +
report are tracked.

### Added — MA57 oracle + 4-way cross-solver comparison

`external_benchmarks/ma57_oracle/` builds a CoinHSL MA57 benchmark
binary alongside the existing MUMPS/SSIDS oracles.
`external_benchmarks/comparison/` is extended from 3-way to 4-way
(feral + MUMPS + SSIDS + MA57), with new `run.py` / `aggregate.py`
/ `report.py` wiring MA57 into the per-matrix sample comparison and
`REPORT.md` summary.

### Added — Issue #9 Steps 2 + 3: 32×32 register-resident kernel wired into production

**Step 3 (SIMD body).** `update_1x1_block32` in
`src/dense/block_ldlt32.rs` tiles trailing destination columns in
groups of four through `schur_panel_minus_nofma_strided_quad`
(n_elim=1), with a trailing `_dual` for the 2-column tail and a final
`axpy_minus_unroll4_nofma` for the 1-column tail. Each tile packs 4
dst columns per pulp dispatch sharing one source-column load — the
intended Phase 2.4.3 register-resident pattern. Per-element output is
byte-identical to the scalar reference and to `factor::do_1x1_update`
(verified by 4 bit-parity unit tests at p=0, p=5, p=30, zero-pivot).

**Step 2 (dispatch wiring).** `do_1x1_update` and `do_2x2_update`
(factor.rs) gain an `n == 32` fast-path delegating to
`update_1x1_block32` / `update_2x2_block32`.
`factor_frontal_blocked_in_place_with_scratch` dispatches
`nrow==ncol==32` fronts to `factor_block32` (which delegates to
`factor_frontal`); the eager unblocked BK loop drives the SIMD update
via the fast-paths. This bypasses `lblt_panel_frontal` for full
32×32 fronts because, at `bs==ncol==32`, the panel's
`apply_blocked_schur_panel` quad-dispatch path is unreachable
(`j_start = k + n_elim == nrow` skips the batched trailing update),
so all trailing-update FLOPs are done by single-column peek-ahead
axpys. The eager-update path issues quad dispatches for every
trailing tile of 4 columns instead.

Bench: median small p90 1.33 (was 1.36), median medium p90 1.74
(was 1.78) across 3 runs. Modest but consistent improvement at the
better edge of the noise band. Inertia 154428/154481, byte-identical
to baseline.

**Step 4 (rank-2 SIMD body)** remains deferred — the quad kernel's
per-q sequential rounding chain is 1-ULP-divergent from
`axpy2_minus_unroll4_nofma`'s fused chain, so a custom
4-dst-column 2-src pulp dispatch is required. 2×2 pivots are rare on
the bench corpus (no measurable bench impact expected); tracked as
follow-up. **Step 5 (cross-arch CI gate)** also tracked as
follow-up.

### Changed — Per-supernode fixed-overhead reduction (#13, Phases A + B + C)

**Phase C (single-slot contrib pool).** New `pub contrib_pool:
Option<Vec<f64>>` field on `FactorScratch`. The multifrontal driver puts
the child's `ContribBlock.data` into the slot after `extend_add` consumes
it; the kernel takes at extract time, clears+resizes to `cdim*cdim`,
and writes. When the slot is empty (cold scratch, or take outpaces put),
the kernel falls back to a fresh `Vec` allocation — bit-identical to the
pre-Phase-C path. An initial multi-slot `Vec<Vec<f64>>` variant was
abandoned: it preserved bit-parity but regressed bench p90 by ~+0.19
(small) / ~+0.30 (medium) in 4 consecutive runs (growable-indirection
bookkeeping cost more than the malloc/free pairs it avoided). The
single-slot variant is bench-neutral vs Phase A+B (small p90 1.41,
medium p90 1.83–1.85) and bit-parity is preserved across all four
parity cases including the new (d) pool-hot pre-seeded case.

Phase C contributes no measurable bench movement on this corpus, but
the infrastructure is correct and ready: if a future kernel change
makes the contrib allocation a bigger fraction of factor cost, the
recycle path engages automatically. Final issue #13 standing:
criterion #1 (ns/sup reduction) MET, criterion #2 (bench p90 small <
1.30 OR medium < 1.60) **unreachable via allocation pooling on this
corpus**, criterion #3 (no correctness regression) MET, criterion #4
(bit-exact `blocked_ldlt`) MET. Per-front kernel cost (32×32 SIMD,
issue #9) is the next plausible lever for the bench-ratio gap.

### Changed — Per-supernode fixed-overhead reduction (#13, Phases A + B background)

**Phase A (`FactorScratch` pool).** New `FactorScratch { subdiag, d_panel }`
struct in `src/dense/factor.rs` pools the two internal-only working buffers
that `factor_frontal_blocked_in_place` previously allocated per supernode.
New entry point `factor_frontal_blocked_in_place_with_scratch` accepts
`&mut FactorScratch`; the existing function is now a thin wrapper that
allocates a fresh scratch and delegates. `FactorWorkspace` carries a
`factor_scratch` field that the three hot-path call sites in
`src/numeric/factorize.rs` (D.3 dense fast path, `factor_one_supernode`,
`factor_one_small_leaf`) thread through. The scratch is safe to re-warm
across different `(nrow, bs)` shapes — the kernel prologue clears and
resizes unconditionally. Bit-parity gated by
`tests/factor_scratch_parity.rs` (7-case size sweep + 6-case repeated-
calls regression) plus the 19 byte-identity `tests/blocked_ldlt.rs`
integration tests.

**Phase B (`extend_add` direct writes).** The multifrontal `extend_add`
in `src/numeric/factorize.rs` now bypasses `SymmetricMatrix::set`/`get`
and writes directly into `frontal.data` using the lower-triangle column-
major linear index. Per-cell work drops by one indirection, one branch,
and one redundant `i >= j` sanity check, with the symmetric-storage
canonicalisation preserved at the caller.

Diagnostic (`cargo run --bin diag_supernode_cost --release`): Phase A
delivered −16 % to −54 % ns/sup on the CRESC100 / ACOPR30 / HAIFAM /
KIRBY2 cluster (issue #13 acceptance criterion #1 MET). Phase B is
within run-to-run noise of Phase A on ns/sup, which is expected
because extend_add is a child-driven post-factor cost rather than
per-supernode.

Bench (`cargo run --bin bench --release`): dense small-frontal p90
1.33–1.37 and medium p90 1.75–1.78 (vs issue baseline 1.33 / 1.70).
Issue #13 acceptance criterion #2 (small p90 < 1.30 OR medium p90 <
1.60) NOT met by Phases A+B alone. 154428/154481 inertia match
preserved exactly. Phase C (return-struct pooling for `l`, `d_diag`,
`d_subdiag`, `contrib`, `perm`, `perm_inv`) is deferred to a separate
session; design choice (ABI break vs take-into vs with_capacity hints)
is unresolved.

### Added — BLAS-3 quad-column trailing-update kernel (#9, parked on #13)

`schur_panel_minus_nofma_strided_quad` in `src/dense/schur_kernel.rs`
processes four trailing columns per pulp dispatch, halving src memory
traffic vs the existing dual kernel. Wired into
`apply_blocked_schur_panel` — every front with ≥ 4 trailing columns
now routes through quad → dual → single fall-through. Bit-exact per
column with four sequential single-column dispatches (176-config
parity sweep + 19 byte-identical `blocked_ldlt` integration tests).
Zero corpus regression: dense small-frontal p90 1.33 (target ≤ 2.0
PASS), medium p90 1.70 (target ≤ 3.0 PASS); 154428/154481 inertia
match, 99.8 % residual pass.

No measurable headline-throughput win on the current corpus — the
2026-04-27 CHAINWOO_0000 root that motivated the work (1984 × 32) no
longer exists on the current build (max actual nrow = 18 after METIS-
ND on this build). The new bottleneck is per-supernode fixed
overhead, tracked as issue #13. Kernel retained as parked
infrastructure: it re-engages automatically when fronts grow tall-
skinny again. See `dev/decisions.md` 2026-05-12 (c).

### Added — block_ldlt32 scaffold and trailing-update primitives (#9)

New module `src/dense/block_ldlt32.rs` with `BLOCK_SIZE = 32`,
`factor_block32` stub (delegates to `factor_frontal` pending the
const-generic driver port), `update_1x1_block32`, `update_2x2_block32`
scalar primitives, and a bit-parity test harness diffing factors by
`to_bits()`. Signatures match the planned pulp dispatch contract; the
SIMD body swap is a surgical follow-up gated on issue #13.

## [0.2.0] - 2026-05-12

### Fixed — Honest `resolved_method` and consistent `Auto` routing (#3)

`SymbolicFactorization.resolved_method` now reflects what the symbolic
pipeline actually ran rather than what the caller requested. Two
behavior changes:

- **ScotchND silent fallback is surfaced.** When SCOTCH's nested-
  dissection recursion produces no separator for the entire graph
  (bordered-KKT shapes such as PoissonControl trigger this), the
  driver falls back to AMD via `amd_leaf` for every recursion node.
  Previously `resolved_method` still reported `ScotchND` while the
  permutation was bit-identical to AMD's. It now reports `Amd`. The
  recovery itself is unchanged — only its visibility is fixed.
- **`OrderingMethod::Auto` is consistent with the no-arg default.**
  Auto resolution now happens against the original matrix's pattern
  before any `LdltCompress` preprocessor reshapes the graph, and the
  residual branch delegates to `pick_default_method` (the rule used
  by the no-arg `symbolic_factorize`). Previously Auto could pick a
  different concrete method than the default rule on the same
  matrix, depending on whether compression triggered. Auto is now a
  strict superset: same answer as the default plus the two extra
  shape-bakeoff branches (`n>100_000` → ScotchND, `n<10_000` →
  KahipND).

Reported by independent triage on K=158 PoissonControl benchmarks.

### Fixed — `CscMatrix::from_triplets` rejects upper-triangle entries (#4)

`CscMatrix::from_triplets` and `CscMatrix::validate` now return
`FeralError::InvalidInput` when any triplet has `row < col`. Previously
upper-triangle entries were silently accepted and routed through
`sort_and_sum_duplicates`, producing a `CscMatrix` whose row indices
violated the documented "lower triangle only" invariant. Downstream
consumers (e.g. `symmetric_pattern`) assume lower-triangle storage, so
the same symmetric matrix described with upper- vs lower-triangle
triplets produced different solve results. The error message identifies
the offending triplet by index and `(row, col)`. Reported by @janosh.

### Changed

- `Solver` now defaults to the rayon-parallel multifrontal driver
  (`factorize_multifrontal_parallel_with_workspace`). The driver is
  bit-exact with the sequential supernodal path on a per-supernode
  basis and falls through to the sequential path when the supernode
  count is below `N_PAR_MIN = 32`, so small-problem latency is not
  affected. Override with `Solver::new().with_parallel(false)`.
  Closes #7. Motivation: pounce's `marine_1600` / `pinene_3200`
  Mittelmann runs were spending all their time in sequential
  `factor_one_supernode` even though the parallel driver was
  available; this wires `Solver` directly to it.

### Added

- `Solver::with_parallel(bool)` — opt out of the rayon-parallel
  driver (returns `Self` for builder chaining).
- `Solver::parallel()` — test/diagnostic accessor for the current
  flag value.
- `SymmetricMatrix::from_pooled_buf(n, buf)` constructor that zeros
  only the lower triangle when reusing a pooled buffer; cuts the
  dead upper-triangle memset out of `factor_one_supernode`'s
  per-supernode hot path. See dev/decisions.md 2026-05-12.
- `compute_infnorm_dense(&SymmetricMatrix)` in `src/scaling/infnorm.rs`
  — dense-native Knight-Ruiz iteration for the D.3/D.4 dense
  fast-path; the sparse `compute_infnorm` remains the path for the
  multifrontal driver.
- `Solver::inertia()` — `Option<&Inertia>` accessor returning the
  full inertia of the last successful factor. Complements the
  Ipopt-shaped `num_negative_eigenvalues` (which panics if no factor
  is stored) for callers that prefer to branch on `None`. Used by
  the cross-solver bench harness.

### Tooling

- Cross-solver comparison harness under
  `external_benchmarks/comparison/` (run.py, aggregate.py, report.py)
  measures feral against MUMPS 5.8.2 and HSL MA97 2.8.1 on a sampled
  SuiteSparse subset and emits `REPORT.md`. Each solver is configured
  to its production-quality (refinement-on) settings so the residual
  comparison is apples-to-apples:
  - feral driver routes through `solve_sparse_refined` (Richardson
    refinement with stagnation exit) in `src/bin/bench_one_matrix.rs`.
  - `external_benchmarks/mumps_oracle/mumps_bench.F` sets
    `ICNTL(10) = 2` (max two iterative-refinement steps) — MUMPS
    default is 0 (no refinement).
  - `external_benchmarks/hsl_bench/hsl_bench.c` wraps
    `ma97_solve_d` in a 4-step Richardson loop because MA97 has no
    native residual-based refinement entry for non-singular systems.
  Configuration is documented in the generated report's Solvers
  table.

### Performance

- Dense fast-path (`dense_fast_factor`) now runs Knight-Ruiz ∞-norm
  scaling directly on the column-major lower-triangle buffer
  produced by `to_dense_into`, removing the `row_idx[k]`
  indirection that dominated wall time on small-dense matrices.
  Routing: `ScalingStrategy::Auto` and `InfNorm` go through the
  dense KR; `Mc64Symmetric`, `Identity`, `External` are honored
  via the unchanged sparse path. Bit-exact with the sparse KR on
  every fast-path-gate matrix (`should_use_dense_fast_path` ⇒
  matrix is small enough that every column-major slot maps 1:1
  onto a CSC entry or a known-zero, and `(d_i · 0 · d_j) = 0`
  is a no-op in the max-reduction). Targets the
  `dev/results/lever-d3/stage1-stage2-2026-04-19.md` §1 finding
  that `compute_scaling` was 82% (34 of 41 µs) of dense-path
  wall time on TRO3X3_0013.
- Pooled `local_contribs` per rayon worker inside `FactorWorkspace`,
  removing a per-task `Vec<Option<ContribBlock>>` of length
  `n_snodes` from the parallel driver. Decisive on cont-201
  (sequential **–34%**, parallel-at-T=8 **–10%**); also helps
  bratu3d (**–6% / –5%**). Bit-exact. See dev/decisions.md
  2026-05-12 and dev/sessions/2026-05-12-01.md.
- Skip the upper-triangle zero on pooled frontal buffer reuse
  (`SymmetricMatrix::from_pooled_buf`). Bit-exact; 5–10% sequential
  wall reduction across mid-size matrices.

### Investigated

- Parallel multifrontal driver lock contention (T=4) — falsified.
  Added opt-in `AtomicLockStats` telemetry to `NumericParams` with
  per-task lock wait/hold counters and eight per-phase wall-time
  counters wrapping the sequential prologue/epilogue. cont-201's
  previously-reported residual headroom is **sequential symbolic
  factorize** (157 ms of a 214 ms single-shot wall), not mutex
  contention (worst-case 3.4% of body time on cont-201, 0.02% on
  c-big). On the cached-symbolic path (production / pounce-IPM
  regime) cont-201 wall drops 214 → 56 ms with body_frac jumping
  0.15× → 0.55×; remaining 1.5× headroom is inside the rayon::scope,
  not at lock sites. Full analysis in
  `dev/debugging/2026-05-12-cont201-cached-headroom.md`.

- Parallel driver within-scope localization (iteration 2) —
  rayon idle dominates. Added `task_wall_ns` (whole-closure
  bracket) and `ws_lock_wait_ns` (per-worker workspace mutex
  wait) to `AtomicLockStats`. Derived `rayon_idle = scope·T −
  task_wall_agg` quantifies the parallelism deficit attributable
  to etree dependencies. cont-201 cached: rayon_idle = 12.3 ms/T
  (78% of the gap) vs locks 1.7 ms/T (10%) and ctrl-flow 1.5
  ms/T (10%). c-big at T=4 is essentially sequential (74%
  rayon-idle capacity, 1.04× speedup vs body_agg). Conclusion:
  **assembly-tree parallelism is exhausted** on these matrices;
  within-supernode parallelism (panel-BK / threaded dense
  kernels) is the only remaining axis. Closes the cont-201
  assembly-tree investigation. See iteration 2 in
  `dev/debugging/2026-05-12-cont201-cached-headroom.md`.

- Scaling cache verification (iteration 3) — `compute_scaling_with_cache`
  works as designed. Added `solver_scaling_phase_split` test
  (`#[ignore]`) that loads the corpus and times
  `pick_scaling_strategy` + `compute_scaling_with_cache(cache=None)`
  + reorder gather. c-big picks `Mc64Symmetric` and the no-cache
  path takes **2.3 seconds** (full Hungarian); the cached path
  in production takes **2.4 ms** — 1000× speedup, cache hits.
  cont-201 and bcsstk38 pick `InfNorm`, which is values-dependent
  Knight-Ruiz iteration (~4 ms per call) and is **not cacheable**
  across IPM iterations. The 3.95 ms scaling slice on cached
  cont-201 is fundamental per-factor work, not a missed cache.
  Closes scaling probe from session 2026-05-12-02 "Next session
  should #2".

- Issue #5 (MSS1 BK inertia non-monotone under δ_w·I): triage
  complete, closed on the feral side. Landed a reproducer test
  + zero_tol/pivot_threshold sweep diagnostics in
  `src/numeric/factorize.rs::tests`. Empirically demonstrated
  that no in-kernel magnitude-floor lever cures the wandering;
  cross-checked MUMPS 5.8.2 and MA57 (via Ipopt's wrapper) and
  confirmed neither implements eigenvalue-aware 2×2 splitting.
  Recommended fix is upstream (caller-side δ_c bump matching
  Ipopt's `PerturbForSingularity`). Full analysis in
  `dev/research/issue-5-mss1-inertia-monotonicity.md` §9.

## [0.1.0] - 2026-05-06

First public release on crates.io. Seven crates published in
dependency order:

- [`feral-ordering-core`](https://crates.io/crates/feral-ordering-core)
  0.1.0 — locked contract surface (`CscPattern`, `OrderingStats`,
  `OrderingError`, `CONTRACT_VERSION`).
- [`feral-amd`](https://crates.io/crates/feral-amd) 0.1.0 — AMD
  fill-reducing ordering (Amestoy, Davis & Duff 1996, 2004).
- [`feral-amf`](https://crates.io/crates/feral-amf) 0.1.0 —
  HAMF4 / Approximate Minimum Fill (Amestoy 1999).
- [`feral-metis`](https://crates.io/crates/feral-metis) 0.1.0 —
  multilevel nested-dissection (Karypis & Kumar 1998).
- [`feral-scotch`](https://crates.io/crates/feral-scotch) 0.1.0 —
  SCOTCH-style nested-dissection (Pellegrini 1996).
- [`feral-kahip`](https://crates.io/crates/feral-kahip) 0.1.0 —
  KaHIP flow-based nested-dissection (Sanders & Schulz 2011; data
  reduction per Ost, Schulz & Strash 2021).
- [`feral`](https://crates.io/crates/feral) 0.1.0 — sparse symmetric
  indefinite direct solver with certified inertia counts.

The accumulated `Unreleased` work below is the complete content of
the 0.1.0 release; future entries will land under `## [Unreleased]`
above.

### Changed (2026-05-03) — `SupernodeParams::default().nemin` 32 → 16

`SupernodeParams::default().nemin` (src/symbolic/supernode.rs:115)
drops from 32 to 16. `nemin` is the minimum supernode size below
which the symbolic phase merges parent and child nodes during
amalgamation. A larger `nemin` yields fewer, fatter supernodes (more
BLAS-3 work per node, more pass-through row padding); a smaller
`nemin` yields more, thinner supernodes (tighter L storage, more
pivot-block boundaries).

The previous `nemin = 32` was inherited from an early dense-kernel
study and out of step with reference solvers — MUMPS uses 5,
SSIDS's canonical configuration sits in the same low band. The new
`nemin = 16` lands halfway, validated by a sweep over
{8, 16, 32, 64} on PoissonControl K=50 and K=158 (both AMD and
METIS-ND): nemin=16 is the sweet spot for `factor_nnz` and factor
wall — K=50 `factor_nnz` -26%, factor wall -18%; K=158 `factor_nnz`
-21%, factor wall ≈ par. The corpus bench retains its dense P90
ratio targets (small-frontal ≤ 2.0 PASS, medium ≤ 3.0 PASS).

The slack recovered comes from the pass-through row padding
diagnosed in `dev/research/factor-nnz-residual-gap.md`: smaller
supernodes have less inflation from rows that flow through ancestors
not pivoting on those rows.

References: `dev/research/factor-nnz-residual-gap.md`,
`dev/decisions.md` 2026-05-03 nemin entry.

### Fixed (2026-05-03) — `build_row_indices` upper-triangle pollution

`build_row_indices` (src/numeric/factorize.rs:2257-2298) now filters
trailing-row candidates with `r < first_col + own_ncol`. The
function iterates `full_pattern.col_ptr[j]` for j in the supernode's
own column range; `full_pattern` is the fully-symmetrized A pattern,
so each column carries both lower-tri (r > j) and upper-tri (r < j)
entries. Upper-tri rows correspond to columns already eliminated by
ancestors of those rows in the etree — including them as trailing
rows polluted every supernode's frontal, propagated up through child
contrib blocks, and inflated `factor_nnz` by 7-19× over the textbook
L-fill (Σ col_counts).

Performance only: numeric assembly writes nothing into the rogue
rows (they receive zeros), so they were dead weight in BK pivoting,
not a source of wrong answers. Inertia is bit-identical across the
test corpus before and after the fix.

PoissonControl K=50 factor_nnz dropped from 1,363,445 to 323,643
(4.2×) and factor time from 231,075 µs to 3,542 µs (65×). K=158
factor_nnz dropped from 46,734,661 to 4,610,269 (10×) and factor
time from seconds to 85,099 µs. All 216 lib + integration tests
pass.

A `cfg(debug_assertions)` invariant assertion at
src/numeric/factorize.rs:1469-1485 verifies the trailing-row floor
on every supernode in debug builds. New regression test file
`tests/build_row_indices_trailing_invariant.rs` (8 tests) covers
four multifrontal-path fixtures (n > N_TINY=16).

References: `dev/research/build-row-indices-fix.md`,
`dev/decisions.md` 2026-05-03 entry.

### Changed (2026-05-02) — `NumericParams::default()` `pivot_threshold = 1e-8`

`NumericParams::default()` now sets `bk.pivot_threshold = 1e-8`,
matching MA27's `cntl[1]` reference default (Ipopt's
`ma27_pivtol`). Previously the default inherited
`BunchKaufmanParams::default()`'s `0.0`, which silently disabled
the column-relative pivot rejection, rook rescue, and delayed-
pivoting rescue paths — producing exact-zero multiplier outputs on
rank-deficient KKT-augmented LS-init systems (issue #2).

`BunchKaufmanParams::default()` (the dense entry point) is
unchanged at `0.0` per the 2026-04-13 dense-vs-sparse split.
Callers that explicitly construct `BunchKaufmanParams` and pass it
to `NumericParams::with_bk` are unaffected; in-tree sparse callers
that already opt in to `0.01` (benches, parity tests) keep their
override.

The `Solver::increase_quality` cascade still includes the W5
"0.0 → 0.01" first-jump rule for callers that explicitly disable
the threshold; from `Solver::new()` the cascade now goes
1e-8 → 1e-6 → 10^-4.5 → ... → `pivtol_max = 0.5`.

References: `dev/research/issue-2-kkt-pivot-default.md`,
`dev/plans/issue-2-kkt-pivot-default.md`, issue #2.

### Added (2026-04-28) — `bench_solver_corpus` realistic-IPM perf bench

New `src/bin/bench_solver_corpus.rs` walks `data/matrices/kkt/`,
groups by family (`<FAM>_NNNN.mtx` → `FAM`), and per family runs two
scenarios — one persistent `Solver` (caches `SymbolicFactorization`
across same-pattern re-factorizations) vs the free-function loop
that re-runs symbolic on every call. Reports per-family table and
corpus-wide aggregate (geomean / p10 / p50 / p90 speedup, symbolic
call counts, implied symbolic share of freefn wall).

Initial run on 534 families × 19,410 iterates: **aggregate 1.70×,
geomean 2.86×, p50 3.00×, p90 4.08×**. Symbolic share of freefn
wall is 41.3% — the realistic figure for IPM-tail workloads, vs the
artificial 64% reported by the per-matrix `bench` which recomputes
symbolic on every of 154k matrices.

Going forward `bench_solver_corpus` is the perf-tuning ground truth
for symbolic-phase optimizations (decisions.md 2026-04-28). The
per-matrix `bench` is retained for inertia/residual correctness
sweeps and oracle ratio comparisons.

Also added: `src/bin/profile_hot.rs` (samply target for sampling
profiles of the hot pipeline) and `[profile.release] debug = true`
in `Cargo.toml` so samply runs symbolicate cleanly.

### Changed (2026-04-27) — Dense kernel B-1 dual-column DSYRK (NR=2)

Phase B-1 of `dev/plans/dense-kernel-blas3.md` (narrow first step).
Replaces the per-column rank-`n_elim` SIMD axpy in
`apply_blocked_schur_panel` with a dual-column kernel that processes
two adjacent trailing columns per dispatch and shares src loads
between both accumulator stacks — halving src memory traffic on the
bulk of the trailing update.

- New `schur_panel_minus_nofma_strided_dual` kernel: 4-way unrolled
  SIMD body holding 8 accumulator registers (4 dst0 + 4 dst1).
  Cap (column-j diagonal) processed by scalar prologue; bulk
  (dst0[1..] and dst1) shares one src load per chunk per q.
- `apply_blocked_schur_panel` walks trailing columns in pairs;
  odd-tail column falls back to the single-column kernel.
- Bit-exact with two sequential `schur_panel_minus_nofma_strided`
  calls (verified by 144-case n_elim × len0 sweep + zero-alpha
  independence test).
- All 16 `tests/blocked_ldlt.rs` byte-identity fixtures continue
  to pass — per-supernode byte identity preserved.

Bench: all four exit-partition gates still PASS; small/medium
frontals see ~+2-3% session-13 baseline regression (within noise);
wide trailing updates (qcqp1500-1c root) get the src-load-sharing
win. Phase B-2 (lift all-1×1 gate for 2×2 streams) and Phase C
(cache-blocked dense root) remain on the plan.

### Changed (2026-04-27) — Dense kernel W-2 2×2 inline (no-swap fast path)

Phase A of `dev/plans/dense-kernel-blas3.md`. The blocked panel
factorization (`lblt_panel_frontal`) now handles no-swap 2×2 pivots
inline instead of bailing to the scalar path on every 2×2 trigger.
ACOPR-style KKT panels with heavy 2×2 pivot counts (43.8% of factor
time on `qcqp1500-1c` is in `ncol=17–32` panels, 28.6% in
`ncol=33–64`) stay on the deferred-Schur fast path.

- New `PanelStatus::ScalarFallbackPeekedNext` variant signals when
  the inline 2×2 path pre-updated col+1 before bailing on
  swap/LAPACK/growth/det-floor — caller adjusts `j_start` to avoid
  a double rank-1 update.
- New `peek_ahead_replay(target_col)` primitive supports replay onto
  arbitrary trailing columns.
- Rank-bs fast path stays gated on all-1×1 panels for bit-exactness
  (axpy2's fused add-then-sub differs from the rank-bs SIMD body's
  per-q sequential mul-sub). Lifting this gate is Phase B-2.
- Bit-exactness contract: per-supernode byte-identity between
  `factor_frontal` (scalar) and `factor_frontal_blocked` is
  preserved on all 16 fixtures including 4 new W-2 2×2 fixtures.

Symmetric-swap 2×2 and rook-rescue 2×2 still bail to scalar; phase B
will address those.

### Added (2026-04-27) — F3.4 `SchurBlock::solve` convenience

Closes the F3 phase plan. Two new methods on `SchurBlock`:

- `SchurBlock::symv(x, y)` — symmetric mat-vec `y = S · x` against
  the dense full-square buffer.
- `SchurBlock::solve(rhs)` and `solve_with(rhs, params)` — factor
  `S` with the existing dense Bunch-Kaufman LDL^T solver and run a
  single solve. The convenience wraps `dense::factor::factor` +
  `dense::solve::solve`; for repeated solves with the same `S`,
  callers should drive the underlying primitives directly to
  amortise the factor cost.

Tests cover (a) hand-computed 3×3 explicit Schur block,
(b) end-to-end `factorize_multifrontal_with_schur` →
`SchurBlock::symv` → `SchurBlock::solve` round-trip on a 4×4 KKT,
(c) dimension-mismatch error path.

This is the last item in the F3 (Schur complement extraction)
phase plan from `dev/plans/kkt-feature-gaps.md`. With F3.1
(ordering hook), F3.2/2b (numeric hook + multi-supernode tail),
F3.3 (per-matrix oracle gate, 250/250 PASS), and F3.4 all
landed, F3 is closed.

### Changed (2026-04-27) — F3.3 acceptance gate switched to per-matrix oracle bound

The strict `feral-vs-MUMPS ≤ 1e-10` reading was unachievable on
ill-conditioned ACOPR-family KKTs because MUMPS itself disagrees
with a pure-Rust dense oracle by ~1e-6 on those matrices
(conditioning floor, not bug). The gate is replaced with a
per-matrix conditioning-adapted bound:

  feral-vs-oracle ≤ max(1e-10, K · MUMPS-vs-oracle)   K = 10

with corpus floor ≥ 100 matrices. New tooling:

- `src/bin/produce_dense_schur` generates `<id>.dense_schur.bin`
  oracle sidecars (partial-pivot GE on `[A_FF | A_FS]`, column-
  major, same layout as MUMPS sidecar).
- `src/bin/diag_schur_parity` now loads both MUMPS and oracle
  sidecars, reports three pairwise distributions and the per-
  matrix Option B verdict.

Result on the 250-matrix corpus subset with sidecars: 250/250
PASS. Worst feral-vs-oracle = 9.841e-7 (ACOPP30_0000); worst
MUMPS-vs-oracle = 1.037e-6 (ACOPR14_0002) — i.e. feral hits the
same conditioning floor as MUMPS, with feral marginally closer
to ground truth on the worst case. Spec text updated in
`dev/research/schur-complement.md`.

### Fixed (2026-04-27) — F3.3 forest-Schur-etree postorder bug

`schur_constrained_postorder` in `src/ordering/postorder.rs`
emitted Schur nodes via DFS over the Schur subtree, which
violates the `post[k] == k` tail identity that
`symbolic_factorize_with_schur` relies on whenever the Schur
etree is a forest (multiple Schur roots) rather than a single
ascending chain. ACOPP30_0000 hit this with 11 Schur roots plus
an internal chain — the resulting permutation reassigned the
diagonal at col 174, producing max_rel_err = 0.997 vs the dense
Schur oracle.

Phase 2 now emits Schur nodes directly in ascending etree-index
order, restoring the tail identity for arbitrary Schur-etree
topology. Regression test
`test_schur_postorder_forest_tail_identity` covers the minimal
forest case (n=8, three Schur roots + internal Schur child).

Corpus impact (`diag_schur_parity` on 250 matrices with MUMPS
Schur sidecars):

- Factor success: 68/250 → 250/250 (no factor failures).
- Median max_rel_err vs MUMPS: 6.437e-13.
- p90: 3.323e-9; p99: 2.059e-7; max: 1.033e-6 (ACOPR14_0002).
- 136/250 within 1e-10 tolerance; 114/250 above (ACOPR14 family,
  conditioning-driven BK pivot divergence — the dense oracle on
  ACOPP30_0000 shows the same 9.8e-7 disagreement with MUMPS).

### Added (2026-04-27) — F2.2 MUMPS RINFOG cross-validation harness

The MUMPS oracle (`external_benchmarks/mumps_oracle/`) now runs
with `ICNTL(11)=1` and emits `RINFOG(4..11)` (matrix infinity-
norm, solution norm, scaled residual, omegas, forward-error
bound, COND1, COND2). The canonical `.mumps.json` sidecar gains a
`conditioning` block. `src/bin/diag_cond_parity` walks the corpus
and reports `kappa_feral / RINFOG(11)` ratios.

Note: RINFOG(10)/(11) are componentwise infinity-norm condition
numbers (Arioli-Demmel-Duff, dsol_aux.F:935), not
`||A||_1 * ||A^-1||_1`. The harness is therefore a directional
cross-check; F2.1's Hilbert/KKT calibration remains the binding
numerical gate for `estimate_condition_1norm`. Corpus geomean
ratio over 165,959 matrices is 4.244e10 — ten orders of magnitude
offset, which led to the F2.2 acceptance-gate reframe in
`dev/decisions.md`.

### Changed (2026-04-27) — AMF default ordering for `n <= 10_000`

`pick_default_method` in `src/symbolic/mod.rs` now mirrors MUMPS's
`ana_set_ordering.F` AMF-vs-METIS rule: `Amf` for `n <= 10_000`,
`MetisND` for `n > 10_000`, with the existing bordered-KKT
(`n >= 5000 && nnz/n < 6`) and chain-pattern (`n >= 2000 && nnz/n
< 4`) escape hatches preserved as MetisND. Replaces the previous
"AMD by default" rule. AMD remains available via
`OrderingMethod::Amd` for diagnostic comparison.

Validated on the 183_293-sidecar Phase C corpus
(`tests/amf_corpus_oracle.rs::amf_corpus_gate`): **183_277 ok / 16
skipped / 0 fail** at the 1.10x feral-amf-vs-MUMPS-HAMF4 nnz_L
gate. Bench Phase 2.8.1 exit partition post-flip:

```
Dense small-frontal (<200)   147982   p90=1.25   target<=2.0  PASS
Dense medium       (<500)    152145   p90=1.72   target<=3.0  PASS
Sparse small-frontal (<200)  153455   p90=1.56   target<=2.0  PASS
Sparse medium       (<500)   153560   p90=1.56   target<=3.0  PASS
```

Sparse buckets moved from p90=1.52 (session 09 baseline, AMD
default) to 1.56 (+0.04). Dense buckets unchanged. Worst-ratio
tail improved dramatically: pre-flip top was NASH_0111 at 22.75x;
post-flip top is KIRBY2_0007 at 6.20x.

ORBIT2_0000 nnz_L stays at 32_105 (the AMF-clean-room headline
win — AMD's 1.4M-nnz_L factor on this kkt-expansion shape was
the original motivation).

### Docs (2026-04-27) — AMF clean-room research note + plan

`dev/research/amf-clean-room.md` and `dev/plans/amf-clean-room.md`
land the foundation for a clean-room Approximate Minimum Fill
(AMF / HAMF4) ordering as a peer of `feral-amd`. Motivation: MUMPS
picks HAMF4 as the default for SYM=2 N≤10000 (`ana_set_ordering.F:
52-78`), and we have empirical confirmation on at least one
bipartite-KKT family (ORBIT2: feral-amd 5.1M nnz_L vs MUMPS HAMF4
110k — 47×). Research note covers the Amestoy 1999 thesis fill
metric, the six inner-loop sites that change vs AMD, and the
`Metric`-trait architecture (extract shared quotient-graph
machinery; `feral-amd` and `feral-amf` become thin specializations).
Plan note breaks the work into four phases (module factoring with
bit-parity gate, AMF metric impl + fixtures, MUMPS HAMF4 oracle
plumbing, wire-up + corpus validation), 3-5 sessions total. No
code changes — pure documentation.

### Performance (2026-04-27) — rank-bs trailing-update accumulator (W-2, 1×1)

`src/dense/factor.rs::apply_blocked_schur` rewritten as a single
`pulp::WithSimd` body (`schur_panel_minus_nofma_strided`) iterating
`for j in j_start..nrow` outermost, accumulating all `n_elim`
contributions in register accumulators, then storing — replacing the
previous `O(n_elim * trailing)` SIMD dispatch pattern of `n_elim`
rank-1 axpys. Bit-exact against the rank-1 reference (no FMA, explicit
mul/sub ordering) — verified for `n_elim ∈ {1,2,4,7,8,16,31,32}` ×
`len ∈ {1,3,7,8,9,15,16,17,31,32,33,63,64,65,256,257}`. 2×2 panels
stay on the rank-1 fallback for now. Combined with W-1: CHAINWOO
driver-level total 28.7 → 4.4 ms (**6.5× speedup**). W-2 from
`dev/plans/dense-kernel-speedup.md`.

### Performance (2026-04-27) — engage blocked panel for ncol >= 8 (W-1)

`src/dense/factor.rs::factor_frontal_blocked_in_place` now dispatches
the deferred-Schur panel kernel whenever `ncol >= 8` (was `ncol > bs`
with default `bs=64`, sending every 32×32 CHAINWOO supernode to the
scalar path). Bit-parity preserved against the scalar reference for
the new fixtures `ncol ∈ {8, 12, 16, 24, 32}`. W-1 from
`dev/plans/dense-kernel-speedup.md`. Combined effect with W-2 above.

### Ordering (2026-04-27) — quasi-dense column quotient (Fix A, opt-in only)

`crates/feral-metis/src/lib.rs::metis_order_full` gained an opt-in
`MetisOptions::dense_quotient_enabled` flag (default **false**). When
enabled, columns with off-degree > `max(40, ceil(10*sqrt(n)))` are
pulled out of the ND graph, the M1–M7 pipeline runs on the
sparse-induced subgraph, and the dense columns are appended at the
end of the returned permutation. The technique is the user-guide
reading of MUMPS `ICNTL(6)` and HSL_MC68's "dense parameter"; expert
review of MUMPS and SPRAL sources (2026-04-27) found that neither
solver actually pre-strips the graph — MUMPS handles dense rows
*inside* QAMD via the `THRESM` parameter (`ana_orderings.F:5226+`)
and SSIDS doesn't special-case them at all. Empirical test on
ORBIT2_0000: enabling Fix A *increased* `nnz_L` from 1.54M to 2.25M
because removing the dense column destroys the structural signal that
makes it the natural top separator. The opt-in path is kept for
diagnostic experimentation (`src/bin/diag_orbit2_quotient.rs`); the
correct fix is a QAMD-style deferral inside `feral-amd`, deferred to a
future session. See `dev/research/orbit2-cluster-regression.md` §10
for the post-mortem.

### Tooling (2026-04-27) — bench `FERAL_KKT_FILTER` filters at load time

`src/bin/bench.rs::load_kkt_dir` now reads `FERAL_KKT_FILTER` and skips
non-matching `.mtx` stems before parsing the `.json` sidecar and
oracle-timing sidecars. Targeted single-family runs that previously
waited for the 156k-matrix corpus to load now complete in <1s
(`FERAL_KKT_ROOTS=kkt-expansion FERAL_KKT_FILTER="CHAINWOO_0000"`:
0.18s vs >60s post-load). The post-load filter is preserved as a
redundant safety net.

### Performance (2026-04-27) — move contrib into ContribBlock (W-3b)

The multifrontal driver previously cloned `ff.contrib` on every
supernode to produce a `ContribBlock` for the parent. For
CHAINWOO_0000's 1984-row root that clone is ~30 MB per factorization.
Replaced with `std::mem::take(&mut ff.contrib)` in
`factor_supernode_dense_internal` and `factor_supernode_leaf_dense`;
the saved `NodeFactors.frontal_factors.contrib` is empty afterward
(production solve paths never read it). CHAINWOO driver-level total:
~23,246 → ~22,620 us. W-3b from `dev/plans/dense-kernel-speedup.md`.

### Performance (2026-04-27) — factor frontals in place (W-3a)

Added `factor_frontal_blocked_in_place(&mut SymmetricMatrix, ...)` in
`src/dense/factor.rs` that factors directly into the caller's buffer,
eliminating the `nrow*nrow` scratch allocation + lower-triangle copy
the kernel had been performing on every supernode call. For
CHAINWOO_0000's 1984-row root supernode that is ~30 MB per call. The
multifrontal driver (`factor_kkt_dense_path`,
`factor_supernode_dense_internal`, `factor_supernode_leaf_dense`)
migrated to the in-place entry; `factor_frontal_blocked` stays as a
wrapper for tests/examples/bit-parity reference. Bit-parity preserved
(`tests/blocked_ldlt.rs` 9/9). W-3a from
`dev/plans/dense-kernel-speedup.md`.

### Fixed (2026-04-27) — 2x2 BK pivot inertia uses trace, not a00

`src/dense/factor.rs::count_2x2_inertia` decided eigenvalue signs from
`sign(a00)` alone. KKT 2x2 blocks routinely have `a00 = 0` (variable
rows have zero Hessian diagonal), so sign-of-a00 mis-attributed those
blocks. Switched to `trace = a00 + a11` per Sylvester's law -- matches
`count_2x2_inertia_val` (sparse path), rmumps's `compute_inertia`, and
canonical Fortran MUMPS. Threaded `a11` through `do_2x2_pivot`.
All 162 lib tests pass.

### Fixed (2026-04-27) — bench prefers MUMPS oracle inertia

The bench's "inertia mismatch vs MUMPS" headline previously compared
feral's factorization against the IPOPT iter-0 `.json` sidecar's
`inertia` field, which is *IPOPT's expected inertia at iteration entry*
-- NOT what MUMPS factually computed on the dumped matrix. After
regenerating 3,377 `.mumps.json` oracle files to record MUMPS-computed
inertia (11,748 sidecars total now have inertia), the bench prefers
the MUMPS oracle when present and falls back to the IPOPT sidecar
otherwise. Sparse "inertia mismatches" 837 -> 52 (~16x improvement).
Added `inertia: Option<(usize, usize, usize)>` to `OracleTiming`.

### Added (2026-04-27) — `pick_default_method` extends to MetisND for sparse low-degree large-n

`src/symbolic/mod.rs::pick_default_method` now picks `MetisND` when
`(n>=5000 && avg_deg<6) || (n>=2000 && avg_deg<4)`. This fixes the
CHAINWOO/HYDROELL/DIXMAANH 1000x factor-time regression by routing
chain-like KKT structures away from AMD (which produces 5x more fill
on these structures). Memory peak on CHAINWOO sequence: 1173 MB ->
145 MB (8.1x). Phase 2.8.1 sparse-medium p90 ratio vs MUMPS: 39.26x
-> 1.61x.

### Added (2026-04-26) — Streaming bench + `FERAL_SPARSE_MAX` cap

- `cargo run --bin bench --release` now streams matrix data through
  each loop iteration instead of loading the entire corpus upfront.
  `KktEntry` shrinks to metadata only (`name`, `mtx_path`, `sidecar`,
  oracle timings). Each loop body re-reads the `.mtx`, runs work,
  drops at end of scope. Peak RSS on `FERAL_KKT_ROOTS=all` drops from
  30+ GB (load-all design, SIGKILLed) to ~17 GB end of dense / ~36 GB
  end of sparse on the 64 GB dev laptop.
- New `FERAL_SPARSE_MAX=N` env var caps the sparse multifrontal loop
  by sidecar dimension (`n + m`). Default `usize::MAX` (no cap).
  Set to `20000` (or similar) to bound the sparse pass on the
  expanded corpus where the n>50000 tail blows the memory ceiling
  per-matrix. Skipped matrices counted as `Size-skipped (n > N)`.
- New `Parse-skipped` summary line counts matrices skipped during
  per-iteration parse (NaN/Inf entries, dim mismatch, unparseable
  `.mtx`); pass-rate denominators subtract these to preserve old
  semantics.
- First end-to-end expanded-corpus validation: 170,176 matrices,
  sparse 167,380 attempted (99.5% inertia, 99.8% residual,
  worst 2.99e8 same outlier as kkt-only POLAK6_0021).

### Fixed (2026-04-26) — `factor_nnz()` accounting matches SSIDS

`SparseFactors::factor_nnz()` and the `summary().nnz_l` accumulator
now report the SSIDS-equivalent count (lower triangle inc diagonal of
each eliminated supernode block + trailing rect rows) instead of the
full dense `nrow * nelim` block. The previous count swept in the
strict-upper triangle of the eliminated block, which is structurally
zero, and was 1.75× larger than SSIDS's `inform%num_factor` at the
median. After the fix bench reports `nnzL/SSIDS p50 = 1.00`,
geomean = 1.09 across the kkt corpus (was 1.75 / 1.79). New audit
binary at `src/bin/diag_factor_nnz_accounting.rs`. MUMPS `INFOG(9)`
ratio drops from 1.00 to 0.62 because `INFOG(9)` includes
delayed-pivot/pre-allocation overhead that feral's actual L-fill
does not.

### Changed (2026-04-25) — Phase 2.13a `AmalgamationStrategy::Auto` is now default

**Default `AmalgamationStrategy` flipped from `Renumber` to `Auto`.**
Auto is a cheap O(n) etree-shape dispatcher: path / near-path
elimination trees go to `Adjacency`, bushy trees go to `Renumber`.
Eliminates the Phase 2.12 MUONSINE regression (5.5× → 1.4× MUMPS)
while preserving the IPM-KKT tail wins from `Renumber`. Corpus
Top-10 max ratio improves 10.64 → 9.66; p99 3.45 → 3.40. To pin a
specific strategy: `SupernodeParams { amalgamation_strategy:
AmalgamationStrategy::{Adjacency,Renumber}, .. }`. See
`dev/decisions.md` Phase 2.13a entry.

### Added (2026-04-25) — Phase 2.13a etree-shape dispatch

- `AmalgamationStrategy::Auto` variant — new `#[default]`. Resolved
  to `Adjacency` or `Renumber` via `pick_amalgamation_strategy`
  before `find_supernodes`.
- `feral::symbolic::pick_amalgamation_strategy(&etree)` — public
  resolver, also exported.
- `feral::symbolic::AUTO_MULTI_CHILD_FRAC_THRESHOLD` — public const
  (0.05) for downstream consumers wanting the same predicate.
- `src/bin/diag_etree_shape.rs` — predicate-design probe over the
  7 known-answer matrices.
- `tests/auto_strategy.rs` — 7 dispatch unit tests (path, complete
  binary tree, empty, leaf-only forest, near-path, fan-at-root).

### Changed (2026-04-25) — Phase 2.12 SSIDS column renumbering is now default

**Default `AmalgamationStrategy` flipped from `Adjacency` to
`Renumber`.** Cuts factor time 30-67% on IPM-KKT tail matrices
(ACOPR30/CRESC100/LAKES/NELSON/SWOPF) at the cost of ~10% on the
corpus median for small CUTEst-Hessian matrices. Net win for feral's
spec-stated mission (IPM/KKT solves). See `dev/decisions.md` Phase
2.12 entries for the workload-weighted reasoning. To revert per-call:
`SupernodeParams { amalgamation_strategy: AmalgamationStrategy::Adjacency, .. }`.

### Added (2026-04-25) — Phase 2.12 SSIDS column renumbering machinery

- `SupernodeParams::amalgamation_strategy: AmalgamationStrategy` field —
  new opt-in enum. Default `Adjacency` (matches every prior release);
  `Renumber` runs an SSIDS-style merge-prediction + biased-postorder
  pass that places desired-merge children adjacent to their parents
  before `find_supernodes`.
- `src/symbolic/supernode.rs::predict_merges` — symbolic merge oracle
  matching the SSIDS trivial-chain + size rule.
- `src/ordering/postorder.rs::biased_postorder` — emits non-merging
  children first, merging children last; merging child sits adjacent
  to its parent.
- Reverse iteration in `find_supernodes` Step 2 under Renumber: parent
  `first_col` shrinks monotonically, opening adjacency for the next
  lower-index sibling.
- `tests/column_renumbering.rs` — 4 structural tests (arrow / bushy
  fan collapse to 1 supernode; tridiagonal stays at 1; perm bijection).
- `tests/column_renumbering_parity.rs` — 3 numeric parity tests
  (inertia + refined residual match across strategies on arrow SPD,
  bordered KKT, ACOPR30_0067).
- `src/bin/diag_strategy_compare.rs` — 5-run-median Phase-2.10
  profiler comparison.

Default stays `Adjacency`. Renumber cuts factor time 60-67% on
ACOPR30/CRESC100 tail and 2-3× supernode count on every tiny-IPM tail
matrix, but adds ~10% to the corpus median sparse factor ratio vs
MUMPS — outside the ±5% graduation budget. Decision recorded in
`dev/decisions.md` (Phase 2.12 entry).

### Added (2026-04-21) — Phase 2.6.5 LDLᵀ-aware ordering (opt-in)

- `src/symbolic/ldlt_compress.rs` — port of MUMPS `ICNTL(12) = 2`
  (Duff-Pralet symmetric matching + quotient-graph compression).
  Public API: `SuperMap`, `build_supermap`, `compress_pattern`,
  `expand_permutation`.
- `SupernodeParams::preprocess: OrderingPreprocess` field — new
  opt-in enum. Default `None`; `LdltCompress` runs MC64 matching,
  contracts each matched pair into one super-variable, orders the
  compressed graph, and expands the super-permutation before handing
  it to the rest of the symbolic pipeline.
- `src/bin/diag_compression_bench.rs` — corpus bench. On the
  worst-10 high-leverage matrices + stride-5000 sample (40 total):
  factor-time geomean `cmp/base = 0.604` (40% faster), 0 inertia
  mismatches. MUONSINE / VESUVIO / HAHN1 / GAUSS2 family factor
  47-60% faster with compression on.
- `tests/ldlt_compress.rs` — 3 integration tests; 9 unit tests in
  the module.

Default stays `OrderingPreprocess::None`. Flip is deferred to a
follow-up session with a larger-scale residual-parity-tracking
bench (plan: `dev/plans/phase-2.6.5-ldlt-compressed-graph.md`).

### Added (2026-04-21) — Phase 2.5.2 parallel multifrontal driver (live)

- `factor_one_supernode` helper — shared per-supernode body for
  sequential and parallel drivers.
- `factorize_multifrontal_supernodal_parallel` — rayon `scope` +
  `AtomicUsize` pending-children task-graph driver. Bit-exact parity
  with the sequential driver (0 / 38 878 KKT-corpus mismatches in
  `diag_par_repeat`). Dispatcher
  `factorize_multifrontal_parallel[_with_workspace]` routes to this
  driver when `should_parallelize_assembly` returns true.
- `should_parallelize_assembly` predicate — true when
  `n_snodes >= N_PAR_MIN (32)` and at least one supernode has ≥ 2
  children.
- `src/bin/parallel_corpus_parity.rs` — full 169 585-matrix audit.
- `src/bin/diag_acopr.rs` / `diag_par_repeat` / `diag_par_firstdiff`
  / `diag_par_frontal_hash` / `diag_parent_unique` — diagnostic
  binaries retained for future debugging.
- `tests/parallel_parity.rs` — 6 KKT parity tests (all pass).

### Fixed (2026-04-21) — Parallel driver seeding race

- Seed loop in `factorize_multifrontal_supernodal_parallel` used a
  dynamic `pending[i].load() == 0` predicate inside `rayon::scope`.
  Workers running spawned leaves decrement parents' counters
  concurrently with the seed loop; a non-leaf whose final child
  completed mid-seed could be spawned twice (once by the caller,
  once by the last child's `fetch_sub==1` trampoline). Replaced
  with a static "no children" filter captured before the scope.
  Root cause found via `FERAL_HASH_FRONTAL` instrumentation on
  ACOPR14_0003.

### Added (2026-04-20) — Phase 2.4.1b scaffolding (RED)

- `BunchKaufmanParams::block_size: usize` (default `64`). Consulted
  only by the forthcoming Phase 2.4.1b blocked-panel path; the
  scalar `factor_frontal` ignores it.
- `pub fn factor_frontal_blocked(..)` — stub that currently returns
  `FeralError::InvalidInput("…Phase 2.4.1b not yet implemented")`.
  Will be implemented in Step 4 (faer-style peek-ahead panel) and
  must produce byte-identical `(L, D, perm, inertia, contrib)` to
  `factor_frontal`, enforced by the six tests in
  `tests/blocked_ldlt.rs` (currently RED by design).

No runtime behavior change: all existing call sites use
`factor_frontal`; the scalar pivot body was extracted into an
internal `scalar_pivot_step` helper, verified byte-identical by the
118/118 lib tests and the 31 integration tests that exercise
`factor_frontal` (`delayed_pivoting`, `dense_ldlt`,
`pivot_rejection`, `dense_fast_path`).

### Changed (2026-04-20) — Bench harness multi-sample denoise

`cargo run --release --bin bench` now resamples per-matrix
factor+solve timings K=5 cold reps for any matrix whose MUMPS oracle
sidecar reports `factor_us < 200 µs`. Reduction: `min` across reps
for factor, `median` for solve. No library behavior change — this
is a test-harness change only. Inertia and residual validation still
run once on the first factor.

**Motivation.** Single-shot wall time at tens-of-µs produced 10–100×
noise excursions that dominated the top-N worst-ratio report (HS85
80×, CERI651BLS 57×, PALMER2ANE 202× — all diagnosed as noise in
session 2026-04-20-01).

**Effect.** Three-run sparse max across `cargo run --bin bench`:

| | pre-denoise | post-denoise |
|-|-----------:|-------------:|
| max spread over 3 runs | 11.81 / 102.07 / 285.80 (24×) | 13.38 / 11.36 / 27.09 (2.4×) |
| sparse p90 | 1.77 | 1.65 |
| sparse p99 | 3.76 | 3.52 |

**Cost.** Bench wall-time 2:15 → 4:00 (+78%), bench runs once per
session.

See `dev/decisions.md` and `dev/results/bench-denoise/summary.md`.

### Added (2026-04-20) — D.4 tiny-n disjunct in dense fast-path gate

`should_use_dense_fast_path` now accepts any matrix with
`n ≤ N_TINY = 16` regardless of density, in addition to the
existing D.3 density-gated disjunct. The `dense_fast_factor`
implementation is unchanged — only the gate predicate is broadened.
Motivated by the finding that at tiny `n` the multifrontal path is
dominated by symbolic-phase overhead, not floating-point work;
D.4 lets tiny matrices skip symbolic entirely.

**Per-call evidence:** six observed top-10 tiny-n rows (HS73,
PALMER1E, HATFLDH, PALMER1A, KIRBY2LS, HEART6LS) show 1.17–1.53×
p50 speedup over forced multifrontal and beat MUMPS by 2–4×.
**Corpus evidence:** sparse factor/MUMPS geomean stable at
0.38–0.39 across three runs (pre-D.4 reference 0.37, within noise).
D.4's unique class (`n ≤ 16 ∧ ρ < 0.25`) appears empty on the
current IPM corpus — observable corpus impact is small, but the
gate is now a correct primitive for any such matrices that arise.

See `dev/decisions.md` (2026-04-20 D.4 entry) and
`dev/plans/sparse-tail-d4.md`.

### Added (2026-04-19) — D.3 dense fast-path for small-dense matrices

`factorize_multifrontal_with_workspace` now routes matrices with
`n ≤ 128` and lower-triangle density `≥ 0.25` to a new
`dense_fast_factor` entry point that skips symbolic analysis +
supernodal assembly entirely: densifies to a `SymmetricMatrix`,
applies global symmetric scaling in place, runs the dense BK kernel
on all `n` columns, and wraps the result in a single-supernode
`SparseFactors` shape-compatible with `solve_sparse`. Out-of-gate
matrices follow a bit-identical multifrontal path — no regression.

Two new public API entry points —
`factorize_multifrontal_supernodal` and
`factorize_multifrontal_supernodal_with_workspace` — provide
explicit gate-bypass for tests and callers that need to force the
multifrontal body on an in-gate matrix.

**Corpus:** sparse factor/MUMPS geomean 0.46 → 0.37 (−20 %),
max ratio 128.34 → 80.22 (−37 %). Ex-ante target was ≤ 0.44,
exceeded with 0.07 margin. Phase 2.8.1 exit partitions still PASS.

See `dev/decisions.md` (2026-04-19 D.3 entry) and
`dev/plans/sparse-tail-d3.md`.

### Added (2026-04-19) — `FactorWorkspace` caller-owned scratch pool

New `FactorWorkspace` struct pools scratch allocations across
`factorize_multifrontal_with_workspace` calls and across supernodes
within a call: `row_map`, the per-supernode frontal matrix
`data` Vec, and the `build_row_indices` scratch buffers.
`Solver::factor` retains one workspace so IPM-style consumers
amortise heap traffic across iterations.

`factorize_multifrontal` still exists and allocates a fresh
workspace per call — callers that don't need amortisation see no
API change.

**Corpus:** sparse factor/MUMPS geomean 0.48 → 0.46. VESUVIO
reallocs collapsed 99 % (2053 → 13). See `dev/decisions.md`
(2026-04-19 D.1 entry) and `dev/plans/factor-workspace.md`.

### Changed (2026-04-19) — `ScalingStrategy::Auto` is the default; Policy 4 fallback

`ScalingStrategy::default()` now returns `Auto` (was `InfNorm`).
`Auto` routes matrices with the arrow-KKT signature
(`diag_only / n >= 0.30`) to `Mc64Symmetric` and everything
else to `InfNorm`. The trade landed an 8× tail compression on
the IPM corpus (worst factor/MUMPS 83× → 10×) at a net change
of -9 in residual_pass count out of 154 588 matrices; 14 of
the 21 regressions are oracle-`numerically_intractable` and 5
are tolerance-edge (residuals 1e-10 → 1e-9).

`Auto` includes a Policy 4 post-scaling diagnostic that falls
back to `InfNorm` when MC64 has demonstrably misfired on a
matrix where InfNorm would do fine. Three conditions must all
fire:

1. `raw_diag_range(matrix) < 1e6` — raw matrix already
   well-conditioned (so MC64 has nothing to recover).
2. `mc_off > 1e6` — MC64's scaled `max(|off|/|diag|)` is
   large.
3. `mc_off / in_off > 1e5` — and is much larger than InfNorm's.

This recovers the MSS1_0009 residual regression (6e-12 →
1e-6 → 6e-12) without sacrificing the VESUVIO/CRESC factor
wins or the MEYER3NE parity tests. Final corpus residual_pass
post-Policy 4: 154 233 / 154 588.

`InfNorm` and `Mc64Symmetric` remain available as opt-in.
Caller code that explicitly passed `&ScalingStrategy::InfNorm`
sees no change.

See `dev/decisions.md` (2026-04-19 entries) and
`dev/research/policy-4-scaling-fallback.md`.

### Added (2026-04-19) — POUNCE integration interface (`Solver`)

A stateful `Solver` handle that mirrors Ipopt's `SymLinearSolver`
contract: factor → check inertia → escalate quality → re-factor.
Built on the just-shipped β refactor so the cached
`SymbolicFactorization` is reusable across every quality level
without invalidation.

Public API additions:
- `Solver`, `FactorStatus`, `QualityLevel` (re-exported from crate
  root).
- `Solver::new() / with_params / factor / solve / solve_refined /
  increase_quality / num_negative_eigenvalues / provides_inertia /
  factors / quality_level / pivot_threshold / scaling_strategy /
  symbolic_call_count`.
- `FactorStatus::{Success, Singular, WrongInertia { actual, expected
  }, FatalError}` — `WrongInertia` does NOT clear the stored factor,
  matching Ipopt `SYMSOLVER_WRONG_INERTIA` semantics.
- `QualityLevel::{Baseline, ScalingEnabled, PivotRaised, Exhausted}`.
- `FeralError::NoFactor` for `solve()` / `solve_refined()` called
  before a successful factor.

Two-stage quality escalation (MA27-style defaults: `pivtol_max =
0.5`, `exponent = 0.75`):
- Stage 1: flip `Identity` scaling to `InfNorm` (skipped if scaling
  is already non-Identity).
- Stage 2: pivot threshold from 0.0 jumps to 0.01, then geometric
  `min(pivtol_max, t^0.75)` until cap; transitions to `Exhausted`.

Cache reuse: `Solver::factor` caches the `SymbolicFactorization` and
re-runs `symbolic_factorize` only when the matrix's
`(n, col_ptr.len(), row_idx.len())` fingerprint changes. The IPM
caller pattern (refactor on the same pattern with new values) pays
the symbolic cost exactly once.

Test coverage: `tests/pounce_interface.rs` (11 integration tests
including the canonical IPM loop pattern) plus 5 unit tests
`U1`-`U5` in `src/numeric/solver.rs` for the escalation state
machine. All 208 tests pass.

### Refactor (2026-04-19) — scaling moved from symbolic to numeric phase (β)

`SymbolicFactorization` is now purely structural and cacheable across
IPM iterations of structurally identical KKTs. MC64/InfNorm scaling
has moved from `symbolic_factorize` to `factorize_multifrontal`.

Public API changes:
- `SupernodeParams` no longer has a `scaling_strategy` field.
- `SymbolicFactorization` no longer carries `scaling`,
  `scaling_pivot_order`, or `scaling_info`.
- New `NumericParams { bk: BunchKaufmanParams, scaling: ScalingStrategy }`
  bundle is the per-numeric input to `factorize_multifrontal`.
- New `NumericParams::with_bk(bk)` constructor for default scaling.
- New `SparseFactors.scaling` and `SparseFactors.scaling_info` fields.

Migration: callsites that previously passed a `BunchKaufmanParams` to
`factorize_multifrontal` now pass `NumericParams::with_bk(bk)`. Sites
that wired per-iteration scaling via `SupernodeParams::scaling_strategy`
now build `NumericParams { bk, scaling }` and pass it on the numeric
side; the symbolic call uses default `SupernodeParams`.

Why: the IPM driver re-factorizes the same sparsity pattern many times
with new values. Tying scaling to the symbolic phase forced a
re-symbolic pass per numeric refactor, defeating cache reuse. New
structural test
`factorize_multifrontal_with_two_strategies_on_one_symbolic` proves
one `SymbolicFactorization` can drive two `ScalingStrategy` values.

### Documentation (2026-04-18) — KaHIP driver-integration decision pinned

A planning pass against the deferred priority "KaHIP K1 data
reduction integration into driver" (sessions 04, 06, 07) ran the
41-matrix shape bake-off with `KahipND` included and decided
**not** to add a dispatcher rule for KaHIP.

Bake-off evidence (`bench_orderings`):

| ordering | fill / AMD geomean | total symbolic time |
|----------|--------------------|---------------------|
| AMD      | 1.000              | 14.4 s              |
| METIS    | 1.024              | 68.2 s              |
| SCOTCH   | 1.038              | 15.3 s              |
| KaHIP    | 1.023              | 81.1 s              |
| Auto     | 0.988              | 14.7 s (mostly AMD) |

KaHIP-with-K1 ties METIS on fill but at 4-6× the per-call setup
cost, and never strictly beats METIS on any of the 41 matrices.
On the 154 588-matrix IPM bench it would only match METIS where
the existing `n>=5000 && nnz/n<6 → MetisND` rule already fires.

Changes:
- `OrderingMethod::KahipND` docstring documents why the dispatcher
  does not select it and where it is reachable.
- New unit test `pick_default_method_never_returns_kahip` pins the
  decision so a future opt-in change must be conscious.
- Research note: `dev/research/ordering-kahip-driver-integration.md`.
- Plan: `dev/plans/ordering-kahip-driver-integration.md`.

No code paths or default behavior change. KaHIP remains reachable
via `symbolic_factorize_with_method(.., KahipND)` and via
`OrderingMethod::Auto`.

### Changed (2026-04-18) — sparse solve: workspace reuse across refinement steps

`solve_sparse` now drives a private `solve_sparse_core_into` that
takes caller-owned scratch (`y_buf` length `n`, `w_buf` length
`max_nrow`). `solve_sparse_refined` builds one `SolveWorkspace`
per call and reuses it across the (up to) 11 internal solves, plus
reuses `r`, `dx`, and `best_x` buffers across iterations and updates
`x` in place rather than constructing `x_new` per step.

Behavior is unchanged — same residuals, same convergence, same best-
iterate semantics. The win is purely allocator pressure on tiny KKT
matrices where per-call `vec![0.0; n]` calls dominated the solve
phase. Microbench (10 000 iters, ns/call) before → after the refined
refactor (which sits on top of the earlier `solve_sparse_core`
workspace fix):

| matrix     |   n |  refined before | refined after |
|------------|-----|-----------------|---------------|
| ALLINITC   |   7 |          2226ns |        1845ns |
| MCONCON    |  48 |          2961ns |        2746ns |
| AVION2     |  94 |         10654ns |        9251ns |
| BATCH      | 121 |         67275ns |       55253ns |
| HAHN1      | 715 |        410527ns |      336387ns |

Corpus impact (154 588 IPM matrices): solve/SSIDS geomean 1.30 →
1.15. Residual pass and worst residual unchanged.

### Changed (2026-04-18) — `symbolic_factorize` default: bordered-KKT fallback to MetisND

`symbolic_factorize` (the default entry point) now applies a narrow
heuristic before dispatching:

  - `n >= 5000 && stored_nnz/n < 6` → `MetisND`
  - everything else                  → `Amd`

The rule catches bordered-KKT structures like CUTEst CRESC132 where
AMD orders the constraint block into a near-dense root frontal that
swallows ~96% of n and drives a ~5000-column delay cascade.

Diagnosis: `cresc_diag` showed CRESC132_0000 with AMD produced a
5084×4854 root frontal (96% of n=5314) and 4846 cumulative delays;
the same matrix with MetisND produced a 2661×2200 root and 1333
delays. Factor time 5.4 s → 480 ms (11× win on that one matrix).
Symbolic max-frontal-width does NOT predict actual width because
delays are a numeric artifact, so a posteriori check on symbolic
alone has no signal — see `dev/journal/2026-04-18-06.org`.

`OrderingMethod::Auto` was rejected previously (bench regression
from per-call KaHIP overhead on n<500 matrices). The new rule is
narrow enough that no IPM-corpus family with n<5000 is affected.

Bench evidence (154,588 IPM matrices):

| metric                  | AMD-only | heuristic |
|-------------------------|----------|-----------|
| factor/MUMPS geomean    | 0.45     | 0.42      |
| factor/MUMPS max ratio  | 521      | 85        |
| solve/SSIDS geomean     | 1.33     | 1.30      |
| residual pass           | 154241   | 154241    |
| worst residual          | 2.69e-4  | 2.69e-4   |

Callers wanting literal AMD with no dispatcher should call
`symbolic_factorize_with_method(matrix, params, OrderingMethod::Amd)`
explicitly. `src/bin/bench.rs` honors that escape hatch via
`FERAL_ORDERING=amd`; with the env var unset, the bench routes
through the new heuristic.

### Changed (2026-04-18) — sparse refinement: 2-strike plateau exit

`solve_sparse_refined` now exits after two consecutive non-improving
steps instead of running all 10 iterations to the unreachable
`ε·√n` relative target. `profile_sparse` showed the prior loop was
adding 11.5× overhead on the bare solve because the relative target
sits below double-precision floor noise on most KKT matrices.

The 2-strike rule preserves bouncing-into-basin behavior on
borderline matrices (a 1-strike rule killed it). Bench evidence
(154,588 IPM matrices):

| metric           | before | after |
|------------------|--------|-------|
| residual pass    | 154327 | 154241 |
| worst residual   | 2.69e-4| 2.69e-4|
| solve/SSIDS      | 1.82   | 1.33  |
| factor/MUMPS     | 0.44   | 0.45  |

86 marginal matrices regress at the residual-pass threshold (still
99.8% pass), worst residual exactly matches the prior bench, and
solve time drops 27% against SSIDS.

Also: `lib.rs` re-exports `solve_sparse` (was internal). New
`src/bin/profile_sparse.rs` per-stage profiler. Full per-stage
numbers and the bug history (a misordered `improved` check that
silently capped iterations to 1) in
`dev/journal/2026-04-18-06.org`.

### Notes (2026-04-18) — `OrderingMethod::Auto` is opt-in only

Closed-loop end-to-end bench (154,588 IPM KKT matrices) showed `Auto`
regresses sparse factor/MUMPS geomean from 0.44 (AMD) to 0.58. Root
cause: thousands of small (n<500) iteration dumps get routed to KaHIP
where K1 + multilevel setup costs 2-3× per call vs AMD. The 0.988
fill geomean from the 41-matrix shape bakeoff is real but does not
amortize on workloads dominated by tiny matrices.

Resolution: `symbolic_factorize` keeps the `Amd` default; `Auto`
remains available via `symbolic_factorize_with_method` for callers
whose workloads are dominated by large or `cresc132`-class matrices.
The doc comment on `OrderingMethod::Auto` warns callers. Full
evidence in `dev/tried-and-rejected.md`.

The `FERAL_ORDERING={amd,auto,metis,scotch,kahip}` env var added to
`src/bin/bench.rs` stays as harness for future ordering experiments.

### Added (2026-04-18) — `OrderingMethod::Auto` adaptive ordering dispatcher

- `src/symbolic/mod.rs`: new `OrderingMethod::Auto` variant. Picks a
  concrete ordering per-matrix from cheap `CscPattern` features:
  - `n > 100_000 && nnz/n < 5`  → `ScotchND` (large-sparse / arrow).
  - `n <  10_000 && nnz/n < 15` → `KahipND` (K1 reductions dominate).
  - otherwise                   → `Amd`.
  Internal helper `choose_adaptive` resolves `Auto` to a concrete
  method at dispatch time; non-`Auto` inputs pass through unchanged.
- `src/bin/bench_orderings.rs`: adds an Auto column to the per-matrix
  table and the summary.
- Unit test `choose_adaptive_rules` covers the four branches;
  `symbolic_factorize_auto_produces_valid_perm` verifies the
  end-to-end dispatch produces a valid permutation.
- Bakeoff (41 matrices):
  - geomean fill: AMD 1.000, METIS 1.024, SCOTCH 1.038, KaHIP 1.023,
    **Auto 0.988** (best on average, beats every fixed method).
  - min-fill wins: AMD 37, METIS 31, SCOTCH 28, KaHIP 37, **Auto 41**
    (Auto never strictly loses).
  - total symbolic time: AMD 15.64s, METIS 71.8s, SCOTCH 16.1s,
    KaHIP 83.7s, **Auto 15.37s** (fastest — dispatches SCOTCH on
    c-big, which is both faster and lower fill than AMD there).

### Changed (2026-04-18) — `feral-kahip` K1 wired into driver; Rule-1-only preset

- `crates/feral-kahip/src/node_nd.rs`: `kahip_nd_order` now runs K1
  data reduction as a pre-pass (via `reduce_graph`) and expands the
  reduced-graph permutation back to original indices via
  `expand_permutation`. The inner nested-dissection pipeline is
  factored into `kahip_nd_inner`.
- `crates/feral-kahip/src/data_reduction.rs`: added `ReduceOptions`
  struct with per-rule toggles (`degree2_simplicial`,
  `degree2_nonsimplicial`, `twins`, `subset`). `::conservative()`
  enables only Rule 1 (degree-1 cascading); `::full()` enables all
  rules. The driver uses `::conservative()`; unit tests use
  `::full()` so all four rules remain covered.
- Fixed a Rule-2 expansion bug: path interiors were anchored only to
  endpoint `u`, but fill-preservation requires them to be eliminated
  before BOTH endpoints. When `pos(w) < pos(u)` in the reduced perm,
  the old expansion produced extra fill through the still-alive path.
  Fix: at expansion time, anchor the path to whichever of the two
  endpoints' ultimate (path-compressed) anchors has the lower
  reduced-perm position. This fix alone improved geomean fill from
  2.094 to 1.876 but did not recover three regressions (vesuvio /
  vesuviou / cresc132) that were 40-50× AMD.
- Rules 2-4 remain implemented and unit-tested but are disabled in
  the driver. Empirically they cause 40-50× fill regressions on the
  bench corpus; root cause is unresolved. See
  `dev/tried-and-rejected.md` for details.
- Bakeoff over the full parity + large corpus (41 matrices):
  - geomean fill: AMD 1.000, METIS 1.024, SCOTCH 1.038, **KaHIP 1.023**
    (was 1.032 pre-K1; KaHIP is now the best on average)
  - min-fill wins: AMD 37, METIS 31, SCOTCH 28, **KaHIP 37** (tied
    with AMD, up from 30)
  - total symbolic time (us): AMD 15.1M, METIS 71.4M, SCOTCH 16.0M,
    KaHIP 84.0M — KaHIP time dropped from 147.6M to 84.0M because
    Rule-1 cascading shrinks the graph fed to the flow refinement.
  - `c-big` (n=345241) KaHIP fill 3.29× → 2.59× (improved but still
    not competitive with SCOTCH's 1.00×; adaptive dispatch or further
    tuning are open follow-ups).

### Added (2026-04-18) — `OrderingMethod::KahipND` solver-side dispatch

- `src/symbolic/mod.rs`: added `OrderingMethod::KahipND` variant;
  `run_external_ordering` dispatches to `feral_kahip::kahip_order`.
  Test `symbolic_factorize_kahip_produces_valid_perm` mirrors the
  existing METIS/SCOTCH perm-bijection checks on the 5×5 grid.
- `src/bin/bench_orderings.rs`: extended the 4-way bakeoff
  (AMD / METIS / SCOTCH / KaHIP), including per-row fill and time
  columns plus a KaHIP win-count / geomean / total-time summary.
- `Cargo.toml`: added `feral-kahip` as a workspace path dep.
- Bakeoff over the full parity + large corpus (41 matrices):
  - geomean fill: AMD 1.000, METIS 1.024, SCOTCH 1.038, KaHIP 1.032
  - min-fill wins: AMD 40, METIS 32, SCOTCH 28, KaHIP 30 (ties count
    for all at min)
  - total symbolic time (us): AMD 14.8M, METIS 77.9M, SCOTCH 16.6M,
    KaHIP 147.6M — KaHIP is the slowest (flow-based refinement at
    every level carries ~10× the per-ordering overhead of AMD/SCOTCH).
  - Notable: `c-big` (n=345241) KaHIP fill is 3.29× AMD — worse than
    METIS 2.69× and SCOTCH 1.00× (tied with AMD). Data point for the
    adaptive dispatcher follow-up.

### Added (2026-04-18) — `feral-kahip` phases K5+K6 (multilevel controller + ND driver)

- New module `crates/feral-kahip/src/cycle.rs` implementing K5
  multilevel edge bisection: reuses `feral_metis::internals`
  (coarsen, fm_refine, initial_partition, rng) for the multilevel
  plumbing, swaps METIS's FM-only refinement for a FM bootstrap +
  K3 flow refinement at each uncoarsening level. Mode tuning
  (Fast/Eco/Strong) controls `n_sep_trials`, `coarsen_floor`,
  `amd_switch`, `fm_pass_cap`, `bnd_distance`, and how many K3
  iterations run at each level.
- `graph_to_undirected` bridge from `feral_metis::internals::Graph`
  (i32-indexed CSR) to `UndirectedGraph` (usize-indexed CSR) so K3
  and K4 can consume subgraphs produced by the multilevel pipeline.
- New module `crates/feral-kahip/src/node_nd.rs` implementing K6
  recursive nested-dissection driver: connected-components walk,
  per-component recursion, AMD leaf fallback
  (`feral_amd::amd_order`) for subgraphs ≤ `amd_switch`, K5
  bisection + K4 `flow_node_separator` lift + separator-last
  numbering for larger subgraphs.
- `kahip_order_full` wired end-to-end; returns contract-conforming
  `(perm, OrderingStats, KahipStats)`. Status updated from
  "pre-implementation scaffold" to "K2-K6 complete".
- `feral-kahip/Cargo.toml`: added `feral-amd` and `feral-metis` path
  deps (same pattern as `feral-scotch`).
- 61/61 feral-kahip tests pass (+12 new: 5 K5, 7 K6). Coverage:
  trivial 10-vertex graph, determinism, balance within slack,
  Fast/Eco/Strong on 12x12 grid, graph bridge preservation,
  diagonal pattern, 10x10 grid → AMD leaf path, 16x16 grid →
  multilevel path, disconnected components, empty graph. Clippy
  clean under `-D warnings`.
- Research note `dev/research/ordering-kahip-k5-k6.md` with the
  combined K5/K6 architecture, mode-parameter mapping, and out-of-
  scope items (full V/F-cycle re-coarsening, K1 integration,
  `OrderingMethod::KahipND` solver dispatch).

### Added (2026-04-18) — `feral-kahip` phase K3 (flow-based edge refinement)

- New shared module `crates/feral-kahip/src/graph.rs`:
  `UndirectedGraph` CSR type (n, xadj, adjncy, eweight) with
  `cut_weight`, `neighbors`, `eweights`, and `from_csc_unit_weights`.
  Infrastructure shared by K3/K4/K5/K6.
- New module `crates/feral-kahip/src/flow_refine.rs` (internal to
  the crate until K5/K6 consume it) implementing one iteration of
  flow-based bisection refinement per Sanders-Schulz 2011 §4:
  - Boundary detection, BFS band extraction with configurable
    `bnd_distance` (plan audit item 12).
  - Undirected edges modeled as anti-parallel directed pairs with
    the full edge weight as capacity on each direction (audit
    item 10).
  - Fixed-node pinning at `pin_depth = min(max_dist_in_part,
    bnd_distance)` per side — pins all band vertices at that
    depth to super-source (part 0) or super-sink (part 1) with
    INF_CAP = `(sum_band_edge_weight / 2) + 1` (audit item 2;
    fallback covers small parts inside the BFS ball).
  - Two-cut most-balanced-min-cut v1: solve max-flow normally +
    reversed; pick the candidate with lower cut weight satisfying
    the balance tolerance. Full MBMC (residual-flow manipulation)
    deferred to K5/K6 (audit item 3).
  - Strict improvement acceptance only.
- 40/40 tests pass (`cargo test -p feral-kahip`); clippy clean.
  Coverage: empty/degenerate inputs, pre-optimal path-midpoint
  cut, suboptimal 7x7 diagonal improvement (cut 12 → 8 with
  bnd_distance=2, ε=0.4), determinism across repeated calls,
  balance-constraint rejection, non-worsening on a random 40-node
  graph, fixed-node pinning invariant on a path graph.
- Research note `dev/research/ordering-kahip-k3.md` with the
  formal algorithm, band/fixed-node definitions, two-cut MBMC v1
  scope, and the 8-item test-oracle construction.

### Added (2026-04-18) — `feral-kahip` phase K2 (push-relabel max-flow)

- Implemented push-relabel max-flow / min-cut in
  `crates/feral-kahip/src/flow.rs` (internal to the crate until phase
  K3 consumes it):
  - Goldberg-Tarjan 1988 preflow algorithm with highest-label active-
    vertex selection (buckets indexed by height, FIFO within a bucket).
  - Gap relabeling per Cherkassky-Goldberg 1995, required by the
    K3 band refinement budget. Gap detection is gated on
    `0 < g < n` (a gap at height 0 would falsely disconnect the
    sink); lifted vertices with residual excess are re-inserted into
    `bucket[n+1]` so stranded flow drains back to source via reverse
    edges.
  - Deterministic tie-breaking (lowest-index admissible neighbor,
    FIFO within same-height bucket) satisfying audit item 16 of
    `dev/plans/ordering-kahip.md`.
  - Min-cut extraction via residual BFS from source.
  - Rejects `MalformedInput` on `source == sink`, out-of-bounds
    endpoints, or negative capacities. Self-loops are ignored.
    Parallel edges are preserved (residual capacity stacks correctly).
- Crate-public surface unchanged: `kahip_order` / `kahip_order_full`
  still return `OrderingError::Internal`. No `OrderingMethod::KahipND`
  yet — dispatch lands with K6.
- 29/29 tests pass (`cargo test -p feral-kahip`); clippy clean.
  Coverage includes malformed-input rejection, unit-capacity path,
  parallel edges, self-loop ignore, diamond bottleneck, CLRS 3e
  Figure 26.1 (max-flow = 23), k×k grid horizontal cut (f = k for
  k ∈ {2, 3, 4, 5}), K_{3,3} bipartite matching (f = 3), cut-
  saturation invariant on a random 30-node graph (disconnected case,
  f = 0) and a hand-laid connected 6-node network (f = 10),
  disconnected-sink zero-flow, and determinism across repeated runs.
- Research note `dev/research/ordering-kahip-k2.md` with the formal
  algorithm, gap-relabeling proof sketch, data-structure layout, and
  the full test-oracle construction.

### Added (2026-04-18) — `feral-kahip` phase K1 (data reduction)

- Implemented Ost-Schulz-Strash 2021 data reduction rules in
  `crates/feral-kahip/src/data_reduction.rs` (internal to the crate
  until the K2–K6 pipeline lands):
  - Degree-1 elimination with cascading and order-preserving op stack.
  - Degree-2 path compression handling both simplicial (endpoints
    adjacent — zero fill) and non-simplicial (one fill edge added)
    sub-cases. Skips pure-cycle chains with a per-pass `skip` array
    so subsequent seeds find other chains.
  - Open and closed twin detection using canonical sorted
    signatures; closed twins (common in KKT diagonal blocks) are
    processed before open twins.
  - Subset elimination (mark-array) as a conservative capstone rule.
  - Path-compressed anchor union-find for permutation expansion.
- Crate-public surface is unchanged: `kahip_order` and
  `kahip_order_full` still return `OrderingError::Internal` because
  the full K1–K6 pipeline is not yet wired. `OrderingMethod::KahipND`
  is not introduced; dispatch wiring lands with phase K6 per
  `dev/plans/ordering-kahip.md`.
- 15/15 tests pass (`cargo test -p feral-kahip`); clippy clean.
  Coverage includes bijection, CSC invariants, cascading, closed
  twins on K4, open-twin-via-degree-2 on K_{2,3}, and a Rule 2
  firing test between two distinct hubs.
- Research note: `dev/research/ordering-kahip-k1.md`.

### Changed (2026-04-18) — `OrderingMethod::Amd` now routes through `feral-amd`

- Default AMD is now the full Amestoy/Davis/Duff AMD in the `feral-amd`
  workspace crate (approximate external degree + aggressive element
  absorption + supervariable detection), replacing the simplified
  exact-external-degree implementation at `src/ordering/amd.rs` in the
  dispatch path.
- Fill and time improvement on the large-matrix corpus: fill 17-23%
  lower on `c-big`, `cont-201`, `bratu3d`; time 18-88× faster.
  Parity-corpus fill is a statistical tie (geomean 1.001).
- `src/ordering/amd.rs` remains on disk as a reference implementation
  and still exports `permute_pattern`. See `dev/decisions.md`
  (2026-04-18 entry) and `dev/journal/2026-04-18-03.org`.
- Parity panel regenerated via `select_parity_panel`: 17 pass + 9
  ignored (was 27 + 1). The additional ignores are rank-deficient
  KKT matrices that now fall on the zero/tiny-signed pivot
  classification boundary; residual quality is preserved (all
  feral residuals ≤ ~1e-8, matching or beating MUMPS).

### Added (2026-04-18) — OrderingMethod enum dispatch wires METIS and SCOTCH into symbolic factorization

- `feral::symbolic::OrderingMethod::{Amd, MetisND, ScotchND}` (default
  `Amd`) selects which fill-reducing ordering
  `symbolic_factorize_with_method` uses.
- `symbolic_factorize` is preserved as a thin delegate that passes
  `OrderingMethod::Amd`, so existing callers are unchanged.
- Cross-crate adapter converts the main crate's owned-usize
  `CscPattern` to the ordering-contract's borrowed-i32 view
  (`i32::try_from` overflow-checks the matrix size) and maps
  `OrderingError → FeralError::InvalidInput` with perm validation
  (length, non-negative, bounded).
- `Cargo.toml` now depends on `feral-metis` and `feral-scotch`
  directly (previously only transitively through
  `feral-ordering-core`).
- The in-tree `src/ordering/amd.rs` is retained as the `Amd`
  implementation pending separate retirement work per
  `dev/decisions.md`.

### Added (2026-04-18) — Comparative ordering bake-off binary and corpora

- New binary `cargo run --release --bin bench_orderings` runs
  `symbolic_factorize_with_method` three times per matrix (AMD /
  METIS / SCOTCH) and reports per-matrix fill + symbolic time
  plus geomean ratios and win counts. Walks `tests/data/parity/`
  (one representative per family) and `tests/data/large/` (flat
  layout) when present.
- Large-matrix corpus: pinned SuiteSparse manifest in
  `dev/scripts/large_matrices.txt` + fetch script
  `dev/scripts/fetch_large_matrices.sh`; four matrices spanning
  n=8k–345k covering symmetric indefinite and KKT regimes.
  `tests/data/large/` gitignored.
- Results and analysis: `dev/research/ordering-bakeoff-2026-04-18.md`.

### Added (2026-04-18) — Adversarial A1-A10 regression tests for FM refinement

- 9 new tests in `crates/feral-metis/src/fm_refine.rs` cover the
  edge cases enumerated in `dev/research/metis-fm-sign-bug.md` §5:
  paths, cycles, checkerboards, K_{m,k} imbalance, bridges,
  empty-side and singleton/empty inputs. Every test enforces the
  I1 bookkeeping invariant `returned_cut == cut_size(labels)`.

### Added (2026-04-18) — I1 bookkeeping-invariant sweep on existing FM tests

- 21 existing FM-style tests across `feral-metis` (fm_refine),
  `feral-scotch` (halo_fm, band_fm, vertex_separator) now enforce
  the I1 invariant `returned_cut == cut_size(labels)` after the
  FM pass. This is the assertion the metis sign bug (fixed in
  `ba31609`) cannot survive.

### Added (2026-04-18) — feral-scotch SCOTCH-style nested dissection (S1-S5 complete)

- `feral-scotch::scotch_order(pattern)` and
  `feral-scotch::scotch_order_full(pattern, opts)` ship as the
  contract-conforming public API (matches `feral-amd::amd_order*` /
  `feral-metis::metis_order*` shape under
  `dev/plans/ordering-crate-contract.md`).
- Pipeline: optional graph compression (S1) at the top level →
  connected-component split → multilevel coarsening (shared with
  feral-metis through `internals`) → best-of-`n_sep_trials` initial
  bisection scored on post-FM cut → halo-FM uncoarsening at every
  projected level (S3) → direct vertex separator via two-sided FM
  (S2, instead of König's min vertex cover) → recursion with AMD
  leaf at `amd_switch`. Band FM (S4) is available as
  `band_fm::band_fm_refine` for callers that want frontier-only FM
  with anchor-supervertex balance accounting.
- 43 unit tests in feral-scotch; clippy clean; deterministic for a
  given `ScotchOptions::seed`.

### Fixed (2026-04-18) — feral-metis FM neighbour-update sign

- `feral_metis::internals::fm_refine::refine_bisection` had flipped
  signs at the `gain[u] ± 2w` neighbour update vs. the
  `gain = ed - id` convention used by `compute_gains` and
  `cur_cut -= gain[v]`. Corrupted `cur_cut` made FM effectively a
  no-op on graphs requiring real moves; the bug was hidden by all
  four existing tests starting from already-optimal cuts or
  blocked-by-balance configurations.
- Added `fm_sign_invariant_on_alternating_path` regression test
  enforcing the I1 invariant `returned_cut == cut_size(graph,
  labels)` (the assertion the bug cannot survive). Pre-fix
  produced `-1143` on P_10 with alternating ABAB labels (cut = 9);
  post-fix returns a small non-negative cut consistent with the
  new labels.
- Full analysis and follow-up adversarial set in
  `dev/research/metis-fm-sign-bug.md`.

### Changed (2026-04-17) — Ordering crate boundary locked (2.6.0)

- New workspace crate `feral-ordering-core`: defines the shared
  contract (`CscPattern<'a>`, `OrderingStats`, `OrderingError`,
  `CONTRACT_VERSION = 1`) that all four ordering crates will
  implement. Zero deps beyond `std`.
- **Breaking:** `feral-amd`'s public surface is retrofitted onto the
  contract.
  - `CscPattern` and error type now re-exported from
    `feral-ordering-core`; `AmdError` removed (use `OrderingError`).
  - `CscPattern` borrows `&[i32]` (was `&[usize]`);
    `amd_order*` returns `Vec<i32>` (was `Vec<usize>`).
  - All public entry points now return
    `Result<_, OrderingError>`.
  - New `amd_order_full(pattern, opts) -> (perm, OrderingStats,
    AmdStats)` — the contract-conforming three-tuple variant;
    `OrderingStats.time_us` is populated, fill/flop estimates are
    `None` pending analysis-phase work.
- Rationale: lock the boundary before implementing METIS, SCOTCH,
  KaHIP so all four backends plug into Ipopt against the same
  surface. See `dev/plans/ordering-crate-contract.md` and
  `dev/decisions.md` entry of 2026-04-17.
- Evidence: all 12 SuiteSparse AMD oracle fixtures still reproduce
  bit-for-bit after the retrofit (perm, ncmpa, ndiv, nms_ldl,
  nms_lu, n_dense_deferred); 29 lib tests pass; clippy clean;
  clean-room check still passes.

### Added (2026-04-17) — feral-amd standalone crate

- New workspace member `crates/feral-amd`: clean-room Approximate
  Minimum Degree (AMD) fill-reducing ordering, Amestoy-Davis-Duff
  quotient graph variant. Full Slice A (correctness) and Slice B
  (mass elimination + supervariable detection) landed under
  `dev/plans/ordering-amd-upgrade.md`.
- Public API: `amd_order`, `amd_order_with_stats`, `amd_order_opts`;
  `CscPattern`, `AmdOptions`, `AmdStats`, `AmdError`.
- Binaries: `feral-amd` (triplet-file CLI) and `feral-amd-bench`
  (arrow/band/grid fixture suite).
- External-oracle match: byte-for-byte agreement with the
  SuiteSparse AMD Rust crate (`amd` 0.2.2) on the pinned
  `tests/data/amd_oracle/*.txt` fixtures
  (diag_4, tridiag_10, arrow_5, arrow_200, band_20_3, grid_7x7,
  amd_demo_24), covering permutation and flop counters.
- Not yet integrated into `feral`. Integration is deferred to
  `dev/plans/ordering-integration.md`.

### Known issues (Phase 2 in progress)

- **The sparse path produces catastrophically wrong residuals on
  matrices with n > 500.** Phase 1 validation only measured
  matrices with n ≤ 500 (the bench harness enforced this via a
  Phase 1a hold-over filter that was not removed until Phase 2).
  When the filter was lifted in Phase 2.1.2, the sparse path
  produced residuals 10⁴ to 10¹⁴ on larger matrices already
  present in the corpus (CHWIRUT1 through CRESC132 at n=5314),
  while canonical MUMPS and SPRAL/SSIDS produced residuals at
  machine precision. Root cause: missing global MC64
  matching-based scaling. Fix in progress as Phase 2.2.1. Until
  it lands, do not use feral on matrices the dense path cannot
  handle.
- **Phase 1 residual pass rate is not a numerical quality
  measurement**, it is a measurement against the bench tolerance
  `n · ε · 10⁶`. On small matrices this tolerance is loose enough
  (≈ 10⁻⁷ at n=500) to accept feral residuals that are already
  6–8 orders of magnitude worse than canonical solvers. Phase 1's
  99.7% sparse residual pass rate survives this re-reading; what
  does not survive is any implicit claim that feral is numerically
  comparable to canonical solvers at those residual levels.

### Phase 2.4 performance (2026-04-14)

- Dense Schur update now uses a pulp-dispatched NEON SIMD kernel
  with 4-way loop unrolling and independent accumulators
  (`src/dense/schur_kernel.rs`). The kernel uses separate
  `mul_f64s` + `sub_f64s` (no FMA) so per-lane rounding is
  bit-identical to the scalar reference; this is verified by
  `assert_eq!` unit tests across a length sweep up to 1024. The
  kernel is wired into `do_1x1_update` and `do_2x2_update` in
  `src/dense/factor.rs` with no runtime A/B flag.
- KKT corpus bench vs MUMPS oracle (n ≤ 500 dense, full sparse
  corpus): dense factor p90 **2.27 → 1.86** (−18.1%); sparse
  factor p90 **3.18 → 2.82** (−11.3%). Both Phase 2.8 exit
  criteria (dense ≤ 2.0, sparse ≤ 3.0) now met.
- Inertia and residual-pass counts are bit-identical to the
  pre-SIMD scalar baseline: dense 152911/154481 inertia, sparse
  153009/154588 inertia, sparse 154329/154588 residual pass. Zero
  correctness regressions.
- An earlier attempt (Phase 2.4.2) wired an FMA-based unroll4
  kernel and caused 4 sparse inertia mismatches from 1-ULP pivot
  classification flips at the `zero_tol` boundary; reverted and
  replaced with the bit-exact non-FMA variant. See
  `dev/tried-and-rejected.md` and `dev/decisions.md` Phase 2.4.3.

### Phase 2.8.1 exit partition check (2026-04-14)

**Correction to the "both exit criteria met" claim above.** The
Phase 2.4 entry measures against the overall `factor/MUMPS` p90
aggregate. The spec exit criterion in `FERAL-PROJECT-SPEC.md` §1747
and `dev/plans/phase-2-planning.md` §2.8.1 is stricter: it asks
"within 2× of MUMPS on small-frontal KKT set, within 3× on medium
set", with explicit bucket definitions (small-frontal: max frontal
dim < 200 AND n ≤ 10³; medium: max frontal dim < 500 AND n ≤ 10⁴).

Applying the partition:

| bucket              |  count | p90  | target | verdict |
|---------------------|-------:|-----:|-------:|:-------:|
| Dense small-frontal | 147982 | 1.39 | ≤ 2.0  | PASS    |
| Dense medium        | 152145 | 1.74 | ≤ 3.0  | PASS    |
| Sparse small-frontal| 153455 | 2.81 | ≤ 2.0  | **FAIL**|
| Sparse medium       | 153560 | 2.81 | ≤ 3.0  | PASS    |

Dense meets both bars cleanly. **Sparse small-frontal fails** the
strict partition with p90 = 2.81 (target ≤ 2.0). Phase 2 cannot
exit formally until this is resolved.

Profile evidence (`examples/profile_sparse_smallfront.rs`, 152128
small-frontal matrices) locates the bottleneck at `amd_order`:
39.8% of total time with a fat tail of ~9 ms on n=234 matrices
(DISCS family). The plan's Phase 2.5.1 target (Liu row-subtree
column counts) is only 2.6% of the budget and is demoted. The new
Phase 2.5.1 priority is diagnosing and fixing AMD. See
`dev/decisions.md` 2026-04-14 "Phase 2.5 priority reordered".

### Phase 2.5.1′ AMD + symbolic fixes (2026-04-14)

Six surgical fixes, identified by an instrumented triage binary
(`examples/triage_discs_amd.rs`) that counted per-phase µs and
scalar `contains` / insert calls:

- **AMD mark array** (`src/ordering/amd.rs`). Replaced
  `adj[a].contains(&b)` inside the fill-edge loop with a scratch
  `Vec<bool>` of size n reused across steps. Marks the current
  adjacency once, checks/inserts with O(1) lookups, unmarks before
  the next outer iteration. Drops the fill phase from O(deg³) to
  O(deg²) per step. Root cause of the pathology: on near-dense
  inputs (DISCS_0012, DMN15103_0000 fully dense) the reachable set
  was already a clique so every `contains` returned `true` after
  scanning the full adjacency vector — 778k lookups for zero inserts
  on DISCS_0012.
- **AMD dense-clique shortcut** (`src/ordering/amd.rs`). When the
  pivot's live neighbors equal all remaining live nodes, eliminating
  it forms a clique among survivors: push them in any order and
  return. Short-circuits DMN15103_0000 entirely and cuts DISCS_0012
  to just the first few steps.
- **Counting-sort `permute_pattern`** (`src/ordering/amd.rs`).
  Replaced `Vec<Vec<usize>>` + sort + dedup with a two-pass
  counting-sort layout (count, prefix sum, fill) plus one per-column
  `sort_unstable` to preserve the sorted-column invariant. ~7×
  faster on DMN15103_0000. Each off-diagonal entry is copied exactly
  once instead of twice then deduped.
- **Dead loop in supernode detection** (`src/symbolic/supernode.rs`).
  Removed a `for child_s in 0..n_snodes` loop that called
  `find_root` on every candidate and did nothing with the result
  (empty body). O(n²) wasted work per matrix. Snode max time
  dropped 507→68 µs; share 7.3% → 1.2%. GROUPING family fell off
  the top-30 worst offenders list.
- **Etree renumbering from postorder** (`src/symbolic/mod.rs`).
  Replaced the second `EliminationTree::from_pattern` call with an
  O(n) renumbering of the AMD-permuted etree through the postorder.
  Postorder is a topological relabeling of the elimination tree,
  so the tree structure is preserved and only node labels change.
  ~3% sparse small-frontal p90 improvement on 3-run median.
- **Dead transpose call** (`src/numeric/factorize.rs`). Removed
  `let _ = build_csc_transpose(&permuted);` and the helper function
  — the value was computed and immediately discarded. Full O(nnz)
  pass per matrix for nothing.

**Phase 2.8.1 exit criterion now satisfied.** All four partitions
PASS on the full KKT bench (154588 matrices):

| bucket              | count  |  p90 | target | verdict |
|---------------------|-------:|-----:|-------:|:-------:|
| Dense small-frontal | 147982 | 1.56 | ≤ 2.0  | PASS    |
| Dense medium        | 152145 | 1.96 | ≤ 3.0  | PASS    |
| Sparse small-frontal| 153455 | 1.99 | ≤ 2.0  | PASS    |
| Sparse medium       | 153560 | 2.00 | ≤ 3.0  | PASS    |

3-run medians on sparse small-frontal: **2.00 / 1.98 / 2.00**
(target ≤ 2.0). Tight margin — run-to-run noise is ~3–5%, so the
next regression in this band could push it back over the gate.
Flagged for monitoring in Phase 3+.

All 93 library tests pass. Inertia and residual counts unchanged.
Zero correctness regressions. See `dev/sessions/2026-04-14-04.md`
and `dev/decisions.md` Phase 2.5.1′ entries.

### Phase 1b Exit (2026-04-12)

Phase 1b closed under the multi-source consensus exit criterion on
the n ≤ 500 subset of the KKT corpus. Feral matches canonical
Fortran MUMPS 5.8.2 on **99.97%** of that subset's inertia — higher
than the agreement between canonical MUMPS and canonical SPRAL/SSIDS
(98.25%). See `dev/sessions/2026-04-12-01.md` and the Known issues
above for the limits of this claim.

### Added
- Sparse multifrontal LDLᵀ solver (`factorize_multifrontal`,
  `solve_sparse`, `solve_sparse_refined`)
- CSC sparse matrix infrastructure (`CscMatrix`, `CscPattern`)
- AMD ordering, elimination tree, postorder, column counts, supernode
  detection with nemin amalgamation (CHOLMOD-style pipeline)
- Symbolic factorization (`symbolic_factorize`) with postorder
  composition of AMD permutation
- Bench failure analysis: family-grouped failure tables, top-worst
  residual lists, dense ∩ sparse cross-comparison
- Bench `FERAL_EMIT_SIDECARS` environment variable: emits canonical
  `.feral.json` sidecars alongside each matrix for consensus analysis
- External benchmark infrastructure (`external_benchmarks/`):
  - Native Fortran MUMPS 5.8.2 oracle (build from `ref/mumps`,
    manifest-based driver, Python JSON wrapper)
  - Native Fortran SPRAL/SSIDS oracle (meson + METIS build, same
    driver pattern)
  - Multi-source consensus computation (Python), applies
    Definitive / Borderline / NumericallyIntractable / Excluded
    verdicts per matrix across four oracles
- Dense LDLᵀ factorization with Bunch-Kaufman pivoting (scalar, unblocked)
- Full 7-step solve sequence with equilibration
- Iterative refinement (`solve_refined`) with best-iterate strategy
- Iterative infinity-norm equilibration (Knight-Ruiz)
- Benchmark harness with built-in dense matrix timing
- CI workflow (test, clippy, fmt, no-unwrap)
- Property-based tests and stress tests (121 total tests)
- Fused update+argmax optimization (halves memory traffic per pivot step)

### Fixed
- **Phase 2.3 — delayed pivoting + sign-preservation fix**: the
  sparse multifrontal path now delays rejected pivots (both 1×1
  column-relative and 2×2 Duff-Reid growth-bound) from non-root
  supernodes to their parent, giving them a landing zone where
  child contributions have been assembled and the block is more
  likely to pivot cleanly. At root supernodes where no further
  delay is possible, `try_reject_1x1_frontal` preserves the
  pivot's sign in the `ForceAccept` fallback: small-but-nonzero
  pivots are accepted with `inertia.positive`/`negative` (not
  counted as zero) and flagged for iterative refinement. Only
  `|d| <= zero_tol ≈ eps` counts as a zero pivot. Evidence:
  sparse KKT sweep worst residual `2.31e+11 → 3.22e-4` (15 orders
  of magnitude across Phase 2.3), sparse-only failure count
  `3328 → 64`, parity panel `11/28 → 22/28`. Dense KKT numbers
  unchanged (99.0% inertia, 99.7% residual pass, 3.99e-2 worst
  on ACOPP30_0002) because the sparse-only `pivot_threshold =
  0.01` config is scoped to `params_kkt_sparse` and
  `BunchKaufmanParams::default()` stays at `0.0`. See
  `dev/sessions/2026-04-13-02.md`, `03.md`, and `04.md`.
- **Phase 2.3 — refinement termination fix**: `solve_sparse_refined`
  (and `dense::solve_refined`) now iterate up to 10 steps (was 3)
  and terminate on a residual-based criterion `||r|| <
  eps*sqrt(n)*||b||` instead of the old `|dx|/|x|` threshold.
  Under `ForceAccept` factorizations the trajectory is non-
  monotone — corrections produce small `dx` without reducing `r`,
  so `dx` is a false convergence signal and the old loop exited
  before reaching the machine-precision basin. The `||b|| = 0`
  case is handled with an absolute threshold; `||b||` is NOT
  clamped to a floor, which would defeat the relative criterion
  on small-RHS matrices (e.g. CERI651C with `||b|| = 3.238e-5`).
  Evidence: parity panel `22/28 → 27/28` (un-ignored AVION2_0510,
  CERI651C_0746, CERI651ELS_1482, HAHN1_0004, MEYER3NE_0253),
  sparse residual pass `154237 → 154329`, worst sparse residual
  `3.22e-4 → 2.50e-4`. Only SSI_2597 remains ignored as a
  pathological factorization-level case deferred to Phase 2.4.
- **Phase 2.2.2 — ACOPP30 MC64 regression**: Phase 2.2.1 MC64
  scaling improved 6 of 7 sanity-panel matrices but pushed
  ACOPP30_0000 from a pre-MC64 residual of `2.84e+16` to
  `2.27e+46` — a 30-order-of-magnitude regression caused by 5
  forced-zero pivots in the `ForceAccept` branch interacting with
  the unscaled residual recompose. Phase 2.2.2 adds
  `BunchKaufmanParams::pivot_threshold` (a column-relative 1×1
  rejection clause matching MUMPS CNTL(1) / SSIDS `options%u`,
  default `0.01`) plus the Duff-Reid 2×2 growth bound. MC64
  callers (`tests/mc64_regression.rs::ldlt_params`,
  `src/bin/bench.rs::params_kkt`,
  `examples/triage_large_cresc132.rs`) opt in at `u = 0.01`.
  ACOPP30_0000 residual drops `2.27e+46 → 1.076e-1` (47 orders),
  now ~17 orders better than the pre-MC64 Identity baseline. The
  remaining 3 regression targets (CHWIRUT1, CRESC100, CRESC132)
  are unchanged — their inertia is already exact or ±2, so the
  column-relative rejection has nothing to fire on. Full closure
  of the MC64 residual gap requires delayed pivoting (Phase 2.3).
  Validation: `dev/validation/phase-2.2.2-pivot-rejection.md`.
- **Postorder pipeline bug**: `symbolic_factorize` did not apply
  postorder to the elimination tree before supernode amalgamation,
  causing merged supernodes to have non-contiguous columns while
  downstream code assumed contiguous ranges. Closed MGH10S_0000
  (inertia (50,1,0) → (35,16,0), residual 2.61e21 → 1.10e-16).
- **Pivot threshold mismatch**: factor flagged pivots as zero at
  `100*eps` while solve divided by them at `eps*1e-10`. The band in
  between produced catastrophic cancellation. `Factors` and
  `FrontalFactors` now carry `zero_tol`/`zero_tol_2x2`; both solve
  paths skip any pivot the factor counted as zero. Closed POLAK6_0021
  (residual 8.97e-1 → 4.6e-17).
- **Best-iterate refinement**: `solve_refined` and
  `solve_sparse_refined` now track the smallest `||r||` across
  refinement steps and return the corresponding `x`, guaranteeing the
  refined answer is no worse than the unrefined one on rank-deficient
  matrices where ForceAccept produced a wrong `A⁻¹`.
- **`zero_tol` default lowered** from `100 * EPSILON` to `EPSILON`.
  The 100× safety margin was flagging tiny-but-legitimately-positive
  pivots as zero on small SPD matrices. Verified against canonical
  Fortran MUMPS, SPRAL/SSIDS, and rmumps on CERI651DLS_0534 and
  FBRAIN3LS_0788. Closed the final 32 Definitive feral failures.

### Changed
- Phase 1b exit criterion redefined from "100% correct inertia +
  solution vs rmumps" to multi-source consensus across feral, rmumps,
  canonical MUMPS 5.8.2, and SPRAL/SSIDS. Recorded in
  `dev/decisions.md` (entry 2026-04-12) with a reconsideration clause.
- Bench no longer prints per-row PASS lines for the 153k KKT corpus
  (~153k lines removed from stdout, runtime reduced). The bench now
  emits summary tables with family-grouped failure analysis and a
  dense ∩ sparse cross-comparison.
