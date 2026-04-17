# Ordering Crate Contract (locked API for the four sibling crates)

**Status:** Pre-implementation — drafted 2026-04-17
**Date:** 2026-04-17
**Scope:** The shared input/output surface that `feral-amd`, `feral-metis`,
`feral-scotch`, and `feral-kahip` all implement. Integration into feral's
main solver (via `dev/plans/ordering-integration.md`) is deferred until
after all four crates ship.
**Ipopt reference:** ipopt-expert report, this session.
Primary files: `IpSparseSymLinearSolverInterface.hpp:139-144`,
`IpTSymLinearSolver.cpp:371`, `IpMa27TSolverInterface.cpp:411`,
`IpMumpsSolverInterface.cpp:412-438`.
**Related plans:** `ordering-amd-upgrade.md` (done),
`ordering-metis.md`, `ordering-scotch.md`, `ordering-kahip.md`.

---

## Motivation

Four sibling ordering crates will be built in sequence. Without a locked
boundary contract, each crate will invent its own `Pattern` type, its
own perm integer width, and its own error shape, and feral's eventual
integration step will face an API-drift tax. This document freezes the
shared surface so METIS/Scotch/KaHIP are written against one spec and
`feral-amd` is retrofitted to match.

The contract is deliberately **minimal** because Ipopt's own
ordering-consumer surface is minimal — no elimination tree, no nnz
estimate, no matching permutation ever crosses the Ipopt boundary
(see `IpSparseSymLinearSolverInterface.hpp:139-144`). Anything beyond
`perm` is FERAL-internal convenience.

---

## Non-goals

- **Not a trait.** A free function per crate is enough and matches
  Ipopt's per-solver dispatch. A trait can be added later if feral's
  main solver wants runtime dispatch. No speculative abstraction.
- **Not a unified options enum.** `MetisOptions`, `ScotchOptions`,
  `KahipOptions`, `AmdOptions` stay distinct. Ipopt itself exposes
  per-solver ordering knobs, not a unified enum.
- **Not a scaling/matching carrier.** Ordering crates produce a pure
  fill-reducing permutation. MC64-style matching and MC19 scaling are
  feral-internal concerns composed outside these crates.
- **Not a retry-signalling API.** Ipopt's `CALL_AGAIN` workspace loop
  lives around the numerical factor stage (MUMPS ICNTL(14) doubling
  at `IpMumpsSolverInterface.cpp:494-524`), not analyze. Ordering
  errors are fatal-or-success.

---

## Shared types (live in a new crate `feral-ordering-core`)

A tiny crate with zero dependencies beyond `std`. Exists so all four
ordering crates take the **exact same** input type and emit the same
stats/error types without a type-conversion layer.

```rust
// crates/feral-ordering-core/src/lib.rs
#![forbid(unsafe_code)]
#![deny(missing_docs)]

/// Borrowed symmetric sparsity pattern in CSC form.
///
/// Full-symmetric storage: both halves present. Row indices within
/// each column sorted ascending. 0-based indexing.
///
/// Invariants enforced by `CscPattern::new`:
/// - `col_ptr.len() == n + 1`
/// - `col_ptr[0] == 0`, non-decreasing
/// - `row_idx.len() == col_ptr[n]`
/// - every row index `< n`
///
/// Structural symmetry is debug-asserted, not release-checked.
#[derive(Debug, Clone, Copy)]
pub struct CscPattern<'a> {
    pub n: usize,
    pub col_ptr: &'a [i32],
    pub row_idx: &'a [i32],
}

impl<'a> CscPattern<'a> {
    pub fn new(n: usize, col_ptr: &'a [i32], row_idx: &'a [i32]) -> Option<Self> { /* ... */ }
    pub fn nnz(&self) -> usize { self.row_idx.len() }
}

/// Diagnostic counters shared by every ordering producer.
///
/// Crate-specific counters (e.g. AMD's `ncmpa`, METIS's refinement
/// passes) live in the crate's own `XxxStats` struct returned
/// alongside this one, not inside it.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct OrderingStats {
    /// Wall-clock ordering time, microseconds.
    pub time_us: u64,
    /// Predicted non-zeros in L (upper bound if known).
    /// `None` when the algorithm does not produce an estimate
    /// (METIS/Scotch/KaHIP typically don't without an etree pass).
    pub fill_estimate: Option<u64>,
    /// Predicted factorization flops. `None` when not produced.
    pub flop_estimate: Option<u64>,
}

/// Shared error shape. Crate-specific error variants are carried
/// via the `Internal(&'static str)` variant rather than a wrapped
/// crate-specific enum, to avoid an error-type dependency tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrderingError {
    /// Input failed `CscPattern::new` validation.
    MalformedInput,
    /// Input pattern was not structurally symmetric (debug-only
    /// detection; crates may return `NonSymmetric` or silently
    /// succeed with incorrect output — caller must guarantee
    /// symmetry).
    NonSymmetric,
    /// Index overflow in the crate's internal workspace
    /// (e.g. i32 overflow on large matrices).
    IndexOverflow,
    /// Graph is disconnected and the crate does not handle
    /// disconnected components. AMD handles them natively; METIS
    /// needs explicit component-wise recursion; Scotch and KaHIP
    /// vary.
    DisconnectedGraph,
    /// Crate-specific failure with a short static message. Keep
    /// short — this is not a rich diagnostic channel.
    Internal(&'static str),
}

impl core::fmt::Display for OrderingError { /* ... */ }
impl std::error::Error for OrderingError {}
```

### Index width choice: `i32`

All four crates use **`i32`** for perm and pattern indices.

- MUMPS, METIS, Scotch, KaHIP all use `int` (i32) natively.
- SuiteSparse AMD uses `int` natively.
- Faer uses `usize` internally but converts at the boundary.
- Ipopt's C++ interface uses `Index` which is typically `int`
  (see `IpMa27TSolverInterface.cpp:411`).

Using `i32` means zero conversion at the Ipopt FFI boundary and
matches every reference implementation. The 2G-column ceiling is not
a practical limit for the KKT sizes FERAL targets (Phase 2 caps at
`n ≤ 100K`; Phase 4 distributed is out of scope).

---

## Per-crate producer function

Each ordering crate exposes exactly one public function with this
shape:

```rust
// crates/feral-metis/src/lib.rs (analogous for scotch, kahip, amd)
pub fn metis_order(
    pattern: &feral_ordering_core::CscPattern<'_>,
    opts: &MetisOptions,
) -> Result<
    (Vec<i32>, feral_ordering_core::OrderingStats, MetisStats),
    feral_ordering_core::OrderingError,
>;
```

- First tuple element: `perm` (new-to-old).
  `perm[k] = j` means new index `k` corresponds to old index `j`.
  This matches `feral-amd`'s current convention.
- Second: shared `OrderingStats` — always returned, always populated
  (time_us is mandatory; `fill_estimate` / `flop_estimate` are
  `None` for crates that don't produce them).
- Third: crate-specific stats (`AmdStats`, `MetisStats`, etc.).
  Structure is the crate's choice.

**Convenience overloads are allowed** (e.g. `amd_order(pattern)`
using default options, returning only `perm`) but every crate MUST
provide the three-tuple function above with the exact signature so
the integration layer can dispatch uniformly without a trait.

**No `inv_perm` in the return.** Callers compute it with a trivial
helper if they need it. Both are trivially derivable — requiring
crates to produce both is duplicate work that can desynchronize.

**No `etree_parent` in the return.** AMD produces a parent array
natively; METIS/Scotch/KaHIP do not. If feral's main solver wants an
elimination tree it computes one from the pattern + perm in a
separate phase (standard symbolic analysis). Baking `etree_parent`
into the contract would force three crates to do extra work that
feral's symbolic phase will repeat anyway.

---

## `feral-amd` retrofit checklist

Current `feral-amd` surface to reshape to match the contract:

1. **Move `CscPattern`** from `crates/feral-amd/src/pattern.rs` to
   the new `crates/feral-ordering-core/src/lib.rs`. `feral-amd`
   re-exports it via `pub use feral_ordering_core::CscPattern;` for
   one release, then drops the re-export.
2. **Switch indices from `usize` to `i32`.**
   - `CscPattern.col_ptr: &[i32]`, `row_idx: &[i32]`.
   - Return type: `Vec<i32>` instead of `Vec<usize>`.
   - Internal workspace stays `i32` (already does — see
     `AmdError::IndexOverflow`).
3. **Add `OrderingStats` to the return tuple.** New public function
   `amd_order_full(pattern, opts) -> Result<(Vec<i32>, OrderingStats,
   AmdStats), OrderingError>`. Keep existing `amd_order` and
   `amd_order_opts` as convenience wrappers that discard
   `OrderingStats`.
4. **Translate `AmdError` → `OrderingError`.** `MalformedInput`,
   `NonSymmetric`, `IndexOverflow` map 1:1. `AmdError` stays as an
   internal type for crate-specific unit tests but is not part of
   the public contract function's error.
5. **Populate `OrderingStats.time_us`** by wrapping the core
   algorithm call in an `Instant::now()` boundary.
   `fill_estimate` and `flop_estimate` stay `None` unless AMD's
   `ndiv + nms_lu + nms_ldl` counters are repurposed (deferrable
   — not part of the retrofit).
6. **Update `feral-amd`'s unit tests and bench** to the new
   signatures. Oracle comparisons stay byte-identical
   (permutation values unchanged; only the Vec element type
   changes).

Retrofit is its own PR, sequenced before METIS implementation
starts. It should leave the existing AMD oracle fixtures untouched
and reproduce the same permutations bit-for-bit.

---

## Version the contract

```rust
// feral_ordering_core
pub const CONTRACT_VERSION: u32 = 1;
```

Any backwards-incompatible change to `CscPattern`, `OrderingStats`,
`OrderingError`, or the producer-function signature bumps this and
is logged in `dev/decisions.md`. Crates re-export the constant so
downstream callers can assert at build time that all four crates
link against the same contract version.

---

## Out of scope for v1 of the contract

Deferred until a concrete need surfaces (document here when added):

- A runtime-dispatch trait over the four producers. Add only when
  feral's main solver needs dynamic ordering selection.
- Matching-based scaling (MC64) output as a permutation returned
  alongside the fill-reducing perm. Lives in feral's scaling layer,
  not in ordering crates.
- Elimination-tree production. Lives in feral's symbolic-analysis
  phase.
- Streaming / incremental updates to an existing ordering. Ipopt's
  analyze is one-shot + cached (`IpTSymLinearSolver.cpp:173-180`);
  no incremental path is justified.

---

## Acceptance criteria for the contract lock

Before METIS starts (`dev/plans/ordering-metis.md` K1):

- [ ] `crates/feral-ordering-core` exists with `CscPattern`,
      `OrderingStats`, `OrderingError`, `CONTRACT_VERSION`.
- [ ] `feral-amd` retrofitted; all existing oracle tests pass
      bit-for-bit against SuiteSparse / faer.
- [ ] A short doc-test in `feral-ordering-core` demonstrates the
      producer-function signature using `feral-amd` as the
      reference implementation.
- [ ] `CHANGELOG.md` records the breaking change to `feral-amd`'s
      public surface under Unreleased.
- [ ] `dev/decisions.md` records the index-width choice (`i32`),
      the "no trait" choice, and the "no etree in contract"
      choice.

After those land, METIS, Scotch, and KaHIP implement
`metis_order` / `scotch_order` / `kahip_order` against the frozen
contract without revisiting boundary design.
