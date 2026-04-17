# Plan: `feral-amd` Standalone Crate

**Status:** Pre-implementation plan — audited twice, gated on oracle precondition
**Date:** 2026-04-16 (revised to standalone-crate scope)
**Research note:** `.crucible/wiki/concepts/approximate-minimum-degree.org`
**Implementation reference:** `.crucible/wiki/summaries/amestoy2004-amd-implementation.org`
**Code reference:** SuiteSparse AMD `amd_2.c` (BSD-3-Clause); faer-rs in-tree
port at `../ripopt/ref/faer-rs/faer/src/sparse/linalg/amd.rs` is a faithful
Rust transliteration and is the most useful reading reference (line numbers
below cite faer's `amd.rs`).
**Related:** `dev/plans/ordering-metis.md`, `dev/plans/ordering-scotch.md`,
`dev/plans/ordering-kahip.md` (sibling crates in the same workspace)

---

## Scope

Build `feral-amd` as a **standalone Cargo workspace member** that
implements true AMD (quotient graph). The crate is self-contained:
its own input type, its own CLI, its own bench harness, its own tests.
Feral's existing `src/ordering/amd.rs` is **not** modified by this plan.
Integration into feral is a separate follow-up plan
(`dev/plans/ordering-integration.md`, to be written after the
family of ordering crates — AMD, METIS, SCOTCH, KaHIP — exists).

The scope in this plan is:
- Bring feral's `Cargo.toml` into workspace form (root + member layout)
- Create `crates/feral-amd/` with lib, bin (CLI), bench
- Implement the quotient-graph algorithm from Amestoy, Davis & Duff
- Ship oracle/property/differential/unit/bench tests
- No change to feral's existing source tree

---

## Audit Findings — Round 1 (faer-expert, 2026-04-16)

Folded into the design below; retained for traceability. Do not re-litigate.

1. **`iwlen` formula** is `nzaat + nzaat/5 + n` (integer division).
   `nzaat = nnz(A+Aᵀ) - diag`.
2. **Dense threshold** is `max(16, min(n, α·sqrt(n)))`; `α<0` disables.
3. **Mark-array overflow** must reset `wflg` (faer `clear_flag`,
   `amd.rs:130-143`) when `wflg ≥ wbig` or `wflg < 2`.
4. **Mass elimination** (`elen[v]==1`, no outside variable neighbors) is
   required for dense-front speedup.
5. **Approximate degree** uses a two-pass `w[e]` scheme (faer
   `amd.rs:368-422`). `degree[i]` is monotone non-increasing — cap each
   update.
6. **Garbage collection** is triggered inline during element construction
   (`pfree ≥ iwlen`, faer `amd.rs:289-338`), not between iterations.
7. **Hash buckets reuse `head/next/last`** via sign-bit encoding
   (faer `amd.rs:452-460`).
8. **Hash includes variable neighbors** as well as elements
   (faer `amd.rs:419,433`).
9. **Absorption happens in two places**: standard absorption during
   element construction at `amd.rs:355-358`; aggressive absorption
   during the approximate-degree loop at `amd.rs:404-407`.
10. **Postorder of the AMD-internal elimination tree** is required.
    Built implicitly via `pe[i] = flip(parent)` during absorption;
    `postorder` (faer `amd.rs:593-599`) uses front sizes.
11. **Supervariable expansion** uses `next[i] = next[e]; next[e] += 1`
    (faer `amd.rs:617-633`). Requires a prior path-compression pass
    over `pe[]`.
12. **`aat` preprocessing** can fail with `IndexOverflow` on
    degenerate inputs.
13. **Tie-breaking in degree-list selection is LIFO**.

---

## Audit Findings — Round 2 (expert panel, 2026-04-16)

### Algorithmic corrections (faer-expert)

A1. **Integer `iwlen` wording.** `nzaat + nzaat/5 + n` — integer
    division, not `1.2·nnz`.
A2. **`wbig` is generic over index type.** This crate commits to `i32`
    indices internally, so `wbig = i32::MAX - n`; document the bound as
    coming from the chosen index type, not the literal.
A3. **Path-compression of `pe[]` happens before postorder**
    (faer `amd.rs:573-590`). The expansion pass does **not** walk `pe`
    chains; it relies on `pe[i]` already pointing to a live
    representative.
A4. **Standard vs aggressive absorption are in different phases.**
    Standard: `amd.rs:355-358`, at the end of each `knt1` iteration of
    element construction. Always runs. Aggressive: `amd.rs:404-407`,
    inside the Pass-2 degree loop. Gated by the `aggressive: bool`
    parameter (default `true`, `amd.rs:975`).
A5. **Missing faer behaviors** to include:
    - Initial zero-degree fast path (`amd.rs:198-219`).
    - Dense-deferred init fast path (same range).
    - `aat` dedup workspace `t_p` of length `n` (`amd.rs:806`).
    - `wflg += lemax` bump before the second `clear_flag`
      (`amd.rs:464-465`).
    - Flop counters `ndiv`, `nms_lu`, `nms_ldl` (`amd.rs:547-566`);
      expose via `AmdStats`.

### Downstream-contract corrections (spral-expert)

A11. **Postorder contract is only "any valid topological ordering."**
     Faer's largest-child-last is one valid ordering; plain DFS is
     another. Whatever we emit must round-trip through an etree
     reconstruction test (§T9a).
A12. **Do not speculate about exporting AMD's internal etree.** If
     later desired, write a separate plan for it.
A13. **Dense-row deferral ≠ delayed pivots.** Structural vs numeric.
     Called out explicitly in §Dense Row Handling.

### Process corrections (Plan)

A14. **Oracle precondition** — `feral-amd/tests/data/amd_oracle/` is
     populated from the SuiteSparse `amd` Rust crate *once*, outside
     this crate. Blocker before any implementation.
A15. **Slice A / Slice B** landing: correctness before perf.
A16. **Rollback rule** spelled out.
A17. **Clean-room CI**: `amd` crate never enters the runtime
     dependency graph of `feral-amd`.
A18. **Property + differential tests** required.
A19. **i32 index limit** documented at the API boundary.
A20. **`AmdStats` telemetry** struct.
A21. **Full-pipeline degenerate-ordering test** — in this standalone
     scope, T10 downgrades to: "large-arrow pattern stays correct and
     does not exceed time/memory envelopes." Full factor+solve
     coverage belongs to the later integration plan.

### Panel round-3 caveats (this revision)

A22. **Oracle tolerance is bidirectional** (§T4). `|nnz_L_new -
     oracle| / oracle ≤ 0.05` in both directions. A dramatic
     improvement usually signals a bug.
A23. **T12 uses median ratio**, not mean, across 200 random matrices
     (raised from 100). Median is robust to single-matrix outliers.
A24. **Programmatic fixture generators** (Arrow, Band, grids) have
     their own provenance block in `tests/data/amd_oracle/README.md`,
     covering generator code location + RNG seed, not just input
     SHA-256.
A25. **§Step 5 wording**: the *call* to `clear_flag` at `amd.rs:466`
     is unconditional; `clear_flag` itself is conditional.
A26. **Memory budget arithmetic** shown: `AmdWorkspace` has 9
     `Vec<i32>` of length `n` (`pe`, `len`, `nv`, `elen`, `degree`,
     `w`, `head`, `next`, `last`) plus `iw` of length `iwlen`, plus
     postorder scratch `3n` and optional `aat` scratch `n`. The
     "`11n`" figure from faer is `9n + 2n` (postorder + aat) —
     rewrite as "9 primary n-arrays, +3n postorder scratch, +n aat
     scratch if enabled."
A27. **No legacy-fallback / env override.** A standalone library has
     no "legacy AMD" to fall back to. Instead, the library returns
     `Err` on `IndexOverflow` and the CLI/bench harness and
     integration callers decide what to do. No `FERAL_AMD=legacy`
     mechanism in this crate.

---

## Oracle Precondition (BLOCKS CODING)

Before the first implementation commit of `feral-amd`'s algorithm code,
land `crates/feral-amd/tests/data/amd_oracle/` as its own commit:

1. In a throwaway directory *outside* this repo, create a Cargo
   project with `amd = "0.2"` (SuiteSparse AMD Rust binding,
   BSD-3-Clause).
2. Run it against each oracle fixture listed in §T4, dumping
   `(perm, nnz_L)` to a text file.
3. For `.mtx` fixtures, record SHA-256 of each input; for programmatic
   fixtures (Arrow, Band, grid), record the generator source path
   inside this plan + the RNG seed.
4. Copy outputs into `crates/feral-amd/tests/data/amd_oracle/<fixture>.txt`.
   Write `tests/data/amd_oracle/README.md` stating provenance, crate
   version, SHA-256s, generator references.
5. `amd` crate must **not** appear in `crates/feral-amd/Cargo.toml`
   under any section.

Rationale: CLAUDE.md hard rule — "the oracle must come from an
external source." Freezing numbers derived from our own
implementation defeats the test.

---

## Workspace Layout

Current state: `feral` is a single package. This plan turns the
repo into a Cargo workspace with `feral` as both the root package and
a workspace member, and adds `feral-amd` as a sibling member.

```
feral/
├── Cargo.toml              # add [workspace] block: members = [".", "crates/*"]
├── src/                    # feral package — UNCHANGED
├── tests/                  # feral tests — UNCHANGED
├── crates/
│   └── feral-amd/          # NEW workspace member
│       ├── Cargo.toml      # name = "feral-amd", license = "MIT"
│       ├── README.md
│       ├── src/
│       │   ├── lib.rs              # public API, re-exports
│       │   ├── pattern.rs          # input types (CSR/CSC slices)
│       │   ├── workspace.rs        # AmdWorkspace, index-type helpers
│       │   ├── algo.rs             # core elimination loop (~600 lines)
│       │   ├── postorder.rs        # AMD-internal postorder + expand
│       │   ├── stats.rs            # AmdStats
│       │   └── error.rs            # AmdError
│       ├── src/bin/
│       │   ├── feral-amd.rs        # CLI: read .mtx, emit perm
│       │   └── feral-amd-bench.rs  # bench harness
│       └── tests/
│           ├── data/
│           │   ├── amd_oracle/     # frozen oracle (Oracle Precondition)
│           │   └── mtx/            # small input fixtures
│           ├── unit_api.rs         # T1, T7
│           ├── fill_quality.rs     # T2
│           ├── oracle_match.rs    # T4
│           ├── dense_handling.rs   # T6
│           ├── property.rs         # T11
│           ├── differential.rs     # T12 (#[cfg(feature = "amd-oracle")])
│           └── etree_roundtrip.rs  # T9a
```

Root `Cargo.toml` adds:
```toml
[workspace]
members = [".", "crates/feral-amd"]
resolver = "2"
```

`feral` package remains on its current dependency list. `feral-amd`
has no dependency on `feral`.

---

## Public API

```rust
// crates/feral-amd/src/pattern.rs
/// Borrowed symmetric sparsity pattern in CSC form (full, both halves).
///
/// `col_ptr.len() == n + 1`, `row_idx.len() == col_ptr[n]`. Row indices
/// within each column must be sorted. The pattern must be structurally
/// symmetric; this is debug-asserted at entry and trusted in release.
#[derive(Debug, Clone, Copy)]
pub struct CscPattern<'a> {
    pub n: usize,
    pub col_ptr: &'a [usize],
    pub row_idx: &'a [usize],
}
```

```rust
// crates/feral-amd/src/lib.rs
/// Compute a fill-reducing AMD ordering using the quotient-graph
/// algorithm (Amestoy, Davis & Duff 1996, 2004).
///
/// Returns a permutation `perm` (new-to-old mapping). Factoring
/// `P·A·Pᵀ` with `P = perm` is expected to produce less fill.
///
/// The input must be the full symmetric pattern (both halves).
pub fn amd_order(pattern: &CscPattern<'_>) -> Result<Vec<usize>, AmdError>;

/// As `amd_order`, but also returns diagnostic counters and flop
/// estimates. The counters are zero-cost in release builds except
/// for `ncmpa`; debug builds populate all fields.
pub fn amd_order_with_stats(
    pattern: &CscPattern<'_>,
) -> Result<(Vec<usize>, AmdStats), AmdError>;

/// Tunable parameters. `Default::default()` matches faer / SuiteSparse
/// defaults (aggressive = true, dense_alpha = 10.0).
#[derive(Debug, Clone)]
pub struct AmdOptions {
    pub aggressive: bool,
    pub dense_alpha: f64,  // negative ⇒ disable dense deferral
}

pub fn amd_order_opts(
    pattern: &CscPattern<'_>,
    opts: &AmdOptions,
) -> Result<(Vec<usize>, AmdStats), AmdError>;
```

```rust
// crates/feral-amd/src/error.rs
#[derive(Debug)]
pub enum AmdError {
    /// Workspace exceeded i32::MAX during AAT or expansion.
    IndexOverflow,
    /// Debug-only: input pattern was not structurally symmetric.
    NonSymmetric,
    /// col_ptr.len() != n+1 or row_idx.len() != col_ptr[n].
    MalformedInput,
}
```

```rust
// crates/feral-amd/src/stats.rs
#[derive(Debug, Default, Clone)]
pub struct AmdStats {
    pub ncmpa: u32,             // garbage-collection count
    pub n_clear_flag: u32,
    pub n_mass_elim: u32,
    pub n_supervar_merge: u32,
    pub n_dense_deferred: u32,
    pub ndiv: u64,
    pub nms_lu: u64,
    pub nms_ldl: u64,
}
```

No dependency on feral's `CscPattern`, `FeralError`, or any other
feral types. The slice-based input lets any caller adapt without
type coupling.

---

## Design

### Data Structures

Arrays use **signed `i32`** indices internally so we can use the
`flip(x) = -2 - x` sentinel trick (faer `amd.rs:126-128`). API
boundaries convert to/from `usize` with
`debug_assert!(n < i32::MAX as usize)` at entry.

```rust
struct AmdWorkspace {
    /// Monolithic adjacency storage, length iwlen = nzaat + nzaat/5 + n.
    /// (faer amd.rs:921-924)
    iw: Vec<i32>,

    /// Pointer into iw for var/element i's list. Negative ⇒ absorbed
    /// (encodes parent as flip(parent)).
    pe: Vec<i32>,

    len: Vec<i32>,     // adjacency list length
    nv: Vec<i32>,      // supervariable size; 0 = absorbed
    elen: Vec<i32>,    // # elements in list (live) / perm index (dead)
    degree: Vec<i32>,  // approx external degree (monotone non-increasing)
    w: Vec<i32>,       // mark array (generation counter `wflg`)
    head: Vec<i32>,    // degree-list heads OR flip-encoded hash-bucket heads
    next: Vec<i32>,    // doubly-linked degree list / hash chains
    last: Vec<i32>,
}
```

Aux: `wflg: i32`, `lemax: i32`, `mindeg: usize`, `pfree: usize`,
`ncmpa: u32`.

**Memory budget:**
- 9 primary `i32` arrays of length `n` (`pe, len, nv, elen, degree,
  w, head, next, last`) = 9n i32s
- `iw` of length `iwlen = nzaat + nzaat/5 + n`
- Postorder scratch: `child`, `sibling`, `stack` = +3n i32s
- AAT scratch `t_p` (only if AAT path enabled): +n i32s
- Total worst case = `iwlen + 13n` i32s

### Elimination Loop

Follows faer `amd.rs` code order — degree update interleaves with
absorption; they are not separable phases.

```
INITIALIZE  (faer amd.rs:198-219)
  - Zero-degree fast path: pre-eliminate vars with deg == 0.
  - Dense-deferred fast path: vars with deg > dense into deferred bucket.
  - Remaining vars inserted into degree lists (LIFO head-insert).

for step in 0..n while nel < n:
  1. SELECT PIVOT  (faer amd.rs:200-241)
     Linear scan from mindeg upward; LIFO unlink.

  2. CREATE ELEMENT  (faer amd.rs:243-361)
     - elenme == 0 path (243-265): build in-place at pe[pivot].
     - elenme  > 0 path (266-361): build at pfree; inline GC if
       pfree >= iwlen (step 6).
     Pass-1 of approximate degree: initialize w[e] = degree[e] + wnvi
     on first touch, w[e] -= nvi on subsequent touches.
     STANDARD ABSORPTION (355-358): at end of each knt1 iter, if
     e != me: pe[e] = flip(me); w[e] = 0.

  3. MASS ELIMINATION  (faer amd.rs:436-444) — Slice B
     If elen[v] == 1 and no variable neighbors outside me:
       pe[v] = flip(me); degme -= nvi; nvpiv += nv[v]; nel += nv[v];
       nv[v] = 0; elen[v] = none.

  4. APPROXIMATE DEGREE  (faer amd.rs:386-422)
     Pass-2: dext = Σ_e (w[e] - wflg).
     AGGRESSIVE ABSORPTION (404-407, if aggressive): absorb when
     dext == 0 but we != 0.
     deg = |A_v| + dext + nvpiv, clamped to n - nel - nvpiv.
     degree[v] = min(degree[v], deg).

  5. SUPERVARIABLE DETECTION (faer amd.rs:391-460, 467-515) — Slice B
     - wflg += lemax; then call clear_flag (unconditional call; the
       function is a no-op when wflg < wbig).
     - hash = wrapping_sum(element indices + variable-neighbor indices)
       mod n.
     - Hash buckets threaded through head/next/last via flip() encoding.
     - Bucket match requires len[j] == ln AND elen[j] == eln.
     - Merge: nv[rep] += nv[dup]; nv[dup] = 0; pe[dup] = flip(rep).

  6. GARBAGE COLLECTION  (faer amd.rs:289-338)
     Triggered INLINE during step 2 when pfree >= iwlen. Preserves
     partial new element via copy_within and flip-head-pointer trick.
     Increments ncmpa.

POST-LOOP  (faer amd.rs:567-633)
  a. Path-compress pe[] (573-590): every dead j gets pe[j] = live e.
  b. Postorder the AMD-internal etree (593-599) using front sizes.
  c. Expand supervariables in-order (617-633):
     next[i] = next[e]; next[e] += 1. No walking — (a) already did it.
```

### Mark-Array Generation Counter

```rust
fn clear_flag(wflg: i32, wbig: i32, w: &mut [i32]) -> i32 {
    // faer amd.rs:130-143
    if wflg >= wbig || wflg < 2 {
        for x in w.iter_mut().filter(|x| **x != 0) { *x = 1; }
        2
    } else {
        wflg
    }
}
```
Called at `amd.rs:366` (pre-construction) and `amd.rs:466`
(pre-supervariable-detection, preceded by `wflg += lemax`).
`wbig = i32::MAX - n`; bound sourced from the chosen `i32` index type.

### Dense Row Handling (Structural, Not Numeric)

Variables with initial degree `> dense = max(16, min(n, α·sqrt(n)))`
are deferred to the end. This is a **structural** device — unrelated
to numeric delayed pivots in LDLᵀ kernels. `α < 0` disables.

### Telemetry

`AmdStats` populated in debug builds. In release, only `ncmpa` has
nonzero overhead (one counter increment inside the GC path). The hot
loop does not branch on stats collection.

---

## Implementation Slices

### Slice A — Correctness

No mass elimination, no supervariable detection. Expected to match or
beat fill quality of a reasonable reference AMD.

1. **Commit 1: Workspace scaffolding.**
   Root `Cargo.toml` gains `[workspace]`. Create `crates/feral-amd/`
   with `Cargo.toml`, empty `src/lib.rs`, `src/pattern.rs`,
   `src/error.rs`, `src/stats.rs`. `cargo build -p feral-amd` green.
   `cargo test` in feral still green.
2. **Commit 2: Oracle precondition.**
   Land `crates/feral-amd/tests/data/amd_oracle/` with provenance.
3. **Commit 3: Workspace init (~100 lines).**
   `AmdWorkspace::new`, zero-degree + dense-deferred fast paths,
   initial degree lists. Unit-test in isolation.
4. **Commit 4: Element construction + standard absorption
   (~300 lines).**
5. **Commit 5: Approximate-degree Pass 2 + aggressive absorption +
   monotone cap (~120 lines).**
6. **Commit 6: Garbage collection (~80 lines).**
7. **Commit 7: Path-compress + postorder + expand (~60 lines).**
8. **Commit 8: Wire `amd_order`, `amd_order_with_stats`, `amd_order_opts`;
   CLI binary; bench binary skeleton.**

**Merge gate (Slice A):** T1, T2, T4, T6, T7, T9a, T11, T12 green.
No benchmark gating.

### Slice B — Performance

9. **Commit 9: Mass elimination (~30 lines).**
10. **Commit 10: Supervariable detection (~140 lines).** Hash includes
    both elements and variables (faer `amd.rs:419,433`); bucket match
    gated on `len[j]==ln && elen[j]==eln`; `wflg += lemax` bump.

**Merge gate (Slice B):** B1, B2, B4 targets met. B1 is "no matrix
slower than Slice A"; perf-target of ≥2× on DISCS/DMN15103 is a
**goal**, not a gate.

---

## Testing Plan

All tests live under `crates/feral-amd/tests/` unless noted.

### Unit tests

**T1. Permutation validity.**
- `perm.len() == n`, bijection, no panics on empty/diagonal/single/
  dense matrices.

**T2. Fill quality on known matrices.**
- Arrow(5): fill = 0.
- Tridiagonal: fill = 0.
- 2D grid (7×7): fill ≤ natural ordering.
- Random sparse (n=100, density=0.05, fixed seed): fill ≤ natural.

**T4. External-oracle match.**
Compare against `tests/data/amd_oracle/*.txt`. Fixtures:
- AMD Demo 24×24 (Davis 2006 Direct Methods §7.2)
- HB/can_24, HB/bcsstk01
- gh_258 regression (52×52) — completes without panic
- Arrow(5) programmatic — `perm == [4,3,2,1,0]`, fill = 0
- Arrow(200) programmatic — hub deferred last, fill = 0
- Band(n, b) programmatic for small (n, b)

Tolerance (bidirectional): `|nnz_L_new - oracle| / oracle ≤ 0.05`
per fixture. A >5% beat is a failure and must be investigated.

**T6. Dense-row handling.**
- One dense row + sparse body: dense row appears last.
- Timing regression guard: Arrow(500) wall time < 10× Arrow(50)
  wall time.

**T7. Edge cases.**
- n = 0, 1, 2; fully disconnected (diagonal); fully connected;
  block diagonal (forest etree).

**T9a. Etree round-trip.**
For each T4 fixture, after AMD:
- Permute the pattern; build an elimination tree from the permuted
  pattern (using a small, crate-local reference etree implementation
  — ~40 lines — so `feral-amd` does not depend on feral).
- Assert the tree is well-formed (acyclic; single root per
  component).
- Assert the permutation is a valid topological ordering of that tree.
Note: this tests the downstream contract spral-expert flagged; it
does not exercise numeric factorization (that belongs to the later
integration plan).

### Property + differential

**T11. Property test (proptest, 256 shrinking cases).**
Random full-symmetric `CscPattern`, `n ∈ [1, 200]`, density
∈ [0.05, 0.5], fixed base seed.
- Bijection.
- `fill(new) ≤ fill(natural) + n` buffer.
- No panics.
- No `Err` other than `IndexOverflow` on pathological sizes.

**T12. Differential vs `amd` crate.** `#[cfg(feature = "amd-oracle")]`,
dev-only, excluded from default `cargo test`. Invoked in a dedicated
CI job that installs the `amd` crate in a *separate* throwaway crate.
- 200 random patterns (raised from 100), fixed seed.
- Per-matrix: `|ratio - 1| ≤ 0.05`.
- **Median** ratio over all 200: within 1% of 1.0. (Median is robust
  to outliers; mean was too fragile.)

### Benchmark

**B1. Ordering-time regression** (bench binary, criterion).
Corpus:
- The T4 fixtures (timing recorded for reference).
- Synthetic matrices: n ∈ {100, 500, 1000, 5000}, fixed seed.
- DISCS-like (n=234, near-dense chain) — synthetic reconstruction,
  not feral's fixture (feral stays untouched).
- DMN15103-like (n=99, near-dense) — synthetic.

**B2. Fill-quality comparison.** `nnz_L` vs external `amd` crate on
T12 corpus. Gate: `ratio ≤ 1.05` on ≥ 95% of matrices.

**B4. Scaling.** Synthetic matrices at n ∈ {1000, 5000, 10000}.
Plot time vs nnz. Expect roughly linear (O(nnz · α(n)) amortized).

### Clean-room CI

- `grep -r '^amd\s*=' crates/feral-amd/Cargo.toml` must return no
  matches outside the `[features]` declaration referencing the test
  feature name. The dev-oracle test runs in a *separate* CI job that
  spawns a scratch crate with the `amd` dependency; nothing flows
  back into `feral-amd`'s dependency tree.
- Automated as a CI job that fails the build on any hit.

### CLI binary

`feral-amd`:
- Input: a path to a Matrix Market (`.mtx`) file of a symmetric
  pattern (numeric values ignored).
- Output: permutation one-per-line on stdout; `AmdStats` as JSON on
  stderr if `--stats` passed.
- Exit 2 on `AmdError`, 1 on I/O error, 0 on success.

### Bench binary

`feral-amd-bench`:
- Iterates a fixed corpus (T4 + synthetic) and prints criterion-
  compatible timing + fill statistics as JSON. Hosts B1/B2/B4.

---

## Migration Strategy (within this crate)

1. **Workspace scaffolding** lands first as a no-op for `feral`.
2. **Oracle precondition** lands second.
3. **Slice A commits 3–8** land one by one; each keeps `cargo test -p
   feral-amd` green (tests are written alongside their commit).
4. **Slice A merge** when T1/T2/T4/T6/T7/T9a/T11/T12 pass.
5. **Slice B commits 9–10** land; B1/B2/B4 recorded.
6. **Slice B merge** when correctness is still green and no
   benchmark matrix regresses below Slice A.
7. Integration into feral is **out of scope** here. When the other
   ordering crates (METIS, SCOTCH, KaHIP) exist, a separate plan
   (`ordering-integration.md`) will decide how feral consumes them.

### Rollback Rule

- **Correctness regressions (T1–T12 fail):** revert the offending
  commit. Do not amend in place.
- **Perf target missed, correctness green:** merge anyway; open
  follow-up ticket. CLAUDE.md: correctness before performance.
- **Unexpected `Err` on any T4 fixture:** treat as a correctness
  regression.

---

## Open-Source Reference Implementations

**Primary reference — SuiteSparse AMD (C, BSD-3-Clause):**
- https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/main/AMD
- `Source/amd_2.c` (core), `Source/amd_order.c` (driver),
  `Source/amd_postorder.c`, `Source/amd_1.c`, `Source/amd_aat.c`.

**faer-rs in-tree port (Rust, the primary reading reference):**
- `../ripopt/ref/faer-rs/faer/src/sparse/linalg/amd.rs` — line
  citations in this plan refer to this file.

**SuiteSparse `amd` crate (Rust FFI, dev-only oracle):**
- `amd = "0.2"` — used in Oracle Precondition and T12 in a *separate*
  scratch crate. Never appears in `crates/feral-amd/Cargo.toml`.

**CHOLMOD symbolic analysis (C, Apache-2.0) — pipeline reference:**
- `Cholesky/cholmod_analyze.c`, `Cholesky/cholmod_rowcolcounts.c`.

---

## Verification Checklist

### Precondition
- [ ] Workspace scaffolding lands; `cargo build -p feral-amd` green;
      `cargo test` (feral root) still green.
- [ ] `tests/data/amd_oracle/` populated from SuiteSparse `amd`
      crate with SHA-256 / generator-provenance in README.
- [ ] `amd` crate not in `crates/feral-amd/Cargo.toml` or any
      transitive dep.

### Slice A
- [ ] `cargo clippy -p feral-amd -- -D warnings` clean
- [ ] No `unwrap`/`expect` in `crates/feral-amd/src/`
- [ ] Memory: `iw` = `nzaat + nzaat/5 + n`; 9 n-sized primary arrays;
      +3n postorder scratch; +n aat scratch (optional)
- [ ] `debug_assert!(n < i32::MAX as usize)` at entry
- [ ] Mark-array `wflg` overflow handled; both call sites covered
- [ ] `wflg += lemax` bump before the second `clear_flag` call
      (the *call* is unconditional; `clear_flag` itself is conditional)
- [ ] Zero-degree + dense-deferred fast paths
- [ ] Element self-absorption during construction (`amd.rs:355-358`)
- [ ] Aggressive absorption gated by `aggressive: bool`
      (`amd.rs:404-407`)
- [ ] Inline GC preserves partial element (`amd.rs:289-338`)
- [ ] Two-pass `w[e]` scheme (`amd.rs:368-422`)
- [ ] `degree[v]` monotone-non-increasing cap (`amd.rs:446`)
- [ ] Path-compress `pe[]` before postorder (`amd.rs:573-590`)
- [ ] AMD-internal etree postordered (`amd.rs:593-599`)
- [ ] Supervariable expansion in-order (`amd.rs:617-633`)
- [ ] T1, T2, T4 (bidirectional ≤5%), T6, T7, T9a, T11, T12 green
- [ ] CLI binary reads `.mtx` and emits perm
- [ ] Bench binary runs without error
- [ ] CI clean-room grep added

### Slice B
- [ ] Mass elimination (`amd.rs:436-444`)
- [ ] Hash includes both elements AND variable neighbors
      (`amd.rs:419, 433`)
- [ ] Bucket match gated on `len==ln && elen==eln` (`amd.rs:490`)
- [ ] B1, B2, B4 recorded; no matrix slower than Slice A
- [ ] `AmdStats.n_mass_elim` and `n_supervar_merge` > 0 on at least
      one fixture

### Post-merge
- [ ] A follow-up plan `ordering-integration.md` decides how feral
      consumes `feral-amd` alongside METIS/SCOTCH/KaHIP.
