# AMF clean-room implementation plan

Status: planning, no code yet.
Date: 2026-04-27.
Research note: `dev/research/amf-clean-room.md`.

## Goal

Land a clean-room Approximate Minimum Fill (AMF / HAMF4) ordering in
pure Rust as a peer of `feral-amd`, achieving feral-amf `nnz_L`
within 10% of MUMPS HAMF4 on the existing kkt corpus, with full
bit-parity preservation of `feral-amd` along the way.

## Phase A — Module factoring (1 session, gate)

Goal: extract `feral-ordering-core` with the AMD quotient-graph
machinery generic over a `Metric` trait. `feral-amd` becomes a thin
wrapper exposing the `MinDegree` metric; **byte-for-byte identical
output** on every test in the existing suite.

### Deliverables

1. New crate `crates/feral-ordering-core` (lib only).
2. `Metric` trait (rough sketch — refine during implementation):

   ```rust
   pub trait Metric {
       /// Score type used for bucketing; the smaller, the more
       /// preferred for the next pivot.
       type Score: Copy + Ord + Default;

       /// State accumulated during Scan 2 for a single variable i.
       type Scan2State: Default;

       fn init_score(len: i32) -> Self::Score;

       /// Called once per element e adjacent to variable i during
       /// Scan 2, after dext(e) is known.
       fn scan2_visit_element(
           state: &mut Self::Scan2State,
           e: ElementRef,
           dext_e: i32,
           deg_e: i32,
           wf_cache: &mut [i32],
       );

       /// Called once per singleton variable j adjacent to i.
       fn scan2_visit_singleton(state: &mut Self::Scan2State, nv_j: i32);

       /// Final score at the end of Scan 2 for variable i.
       fn finalize_score(
           state: Self::Scan2State,
           deg: i32,
           degme: i32,
           nvi: i32,
           nleft: i32,
       ) -> Self::Score;

       /// Bucket index for a score. Length of the head array.
       fn bucket(score: Self::Score, n: i32) -> usize;
       fn n_buckets(n: i32) -> usize;

       /// Whether the bucket at `idx` is in the "coarse" region and
       /// requires an exact linear scan to pick the minimum entry.
       fn coarse_bucket(idx: usize, n: i32) -> bool;

       /// Update parent's score on supervariable absorption.
       fn merge_supervariable(
           parent: &mut Self::Score,
           child: Self::Score,
       );
   }
   ```

3. `pub struct MinDegree;` impl in `feral-ordering-core` reproducing
   AMD exactly. `Score = i32` (degree). `Scan2State = i32`
   (running degree). `bucket = score`. `coarse_bucket = false`.
   `merge_supervariable` is no-op.
4. `feral-amd` retained as a public crate; its `amd_order_full` /
   `amd_order_opts` re-route through `feral_ordering_core::order(…,
   MinDegree)`. Public API preserved bit-for-bit.
5. **Bit-parity test gate**: `cargo test --workspace` passes;
   integration suite (especially `tests/amd_*` and any sidecar perm
   sidecars in `data/matrices/kkt*/*.amd.json`) shows zero diffs.

### Risk and mitigation

- **Risk**: refactor changes inlining and produces a different
  permutation due to tie-break-sensitive linked-list order.
- **Mitigation**: keep the inner-loop control flow byte-identical.
  Trait dispatch is at the *score* level, not the *traversal* level.
  Land the refactor as one commit with explicit "feral-amd output
  unchanged on N matrices" evidence in the message.

## Phase B — AMF metric implementation (1-2 sessions)

Goal: add `MinFill` impl of `Metric` in `feral-ordering-core` and
expose as a new `feral-amf` crate. Produce non-trivially-different
permutations from AMD; functional invariants pass; small-matrix
fixtures match hand-derived expectations.

### Deliverables

1. `WF` workspace allocation (length N i32) added to the core
   driver. Lazy-initialized for elements during Scan 1.
2. `MinFill` `Metric` impl. Six inner-loop sites:
   - **Init**: `WF(i) = LEN(i)`, score is `LEN(i)`.
   - **Scan 1**: when element `e` first encountered this iteration,
     set `WF(e) = 0` (single store).
   - **Scan 2 accumulator**: triple `(DEG, WF4, WF3)` accumulated.
     `WF(e)` lazily computed as
     `dext * (2*deg(e) - dext - 1)` on first use this iteration.
     End: `WF(i) = WF4 + 2*NVI*WF3`. Loose-degree special case
     zeroes `WF4, WF3`.
   - **Supervariable absorption**: `WF(i) = max(WF(i), WF(j))`.
   - **Final score**: compute `RMF` in `f64`, quantize to integer.
   - **Pivot selection**: linear scan when bucket is coarse
     (`MINDEG > NORIG`).
3. Quantized bucket array of length `2N + 2` with `PAS = max(N/8, 1)`
   stride above `NORIG`.
4. New crate `crates/feral-amf` (lib only) exposing
   `amf_order_full` / `amf_order_opts` mirroring `feral-amd`'s API.
5. **Functional invariant tests** (in `feral-ordering-core/tests/`,
   parameterized over `MinDegree` and `MinFill`):
   - Output is a permutation of `0..n`.
   - `pe(root)` chain forms a forest with `n` leaves total.
   - Sum of `nv` over principal supervariables = `n`.
6. **Hand-derived small-matrix fixtures** in `feral-amf/tests/`:
   - 3×3 arrowhead — AMF must place the hub last (where AMD also
     does).
   - 5×5 dual-arrowhead — AMF must defer both hubs (AMD picks one
     hub at iteration 2, AMF picks neither until later).
   - 7×7 banded — AMF and AMD agree on the natural ordering.
   - At least one fixture with score derivation written out by hand
     in a comment, citing Amestoy 1999 metric.

### Risk and mitigation

- **Risk**: subtle quantization bug makes `RMF` overflow integer
  range and bucket placement is wrong.
- **Mitigation**: explicit test case where `RMF > 2^31` (large `N`,
  dense column) verifying the fall-through to `RMF / N`.
- **Risk**: tie-breaking divergence with MUMPS makes nnz_L
  comparison flaky.
- **Mitigation**: accept 10% nnz_L margin; *do not* try to match
  MUMPS perms byte-for-byte.

## Phase C — Test oracle plumbing (1 session)

Goal: produce per-matrix MUMPS HAMF4 sidecars in
`data/matrices/kkt*/<family>/<name>.hamf4.json` containing the
HAMF4 perm and the resulting nnz_L. Gate: feral-amf nnz_L ≤ 1.10 ×
HAMF4 nnz_L on every matrix in a representative subset.

### Deliverables

1. Extend `external_benchmarks/mumps_oracle/` (analogous to the
   existing ssids_oracle) with a Fortran/C harness that calls MUMPS
   analyze with `ICNTL(7) = 2` (force AMF) and dumps `id%SYM_PERM`
   plus the symbolic factor's `nnz_L`. Output JSON next to each
   `.mtx`.
2. CI sidecar-comparison test in `tests/amf_corpus_oracle.rs` (only
   runs when the sidecars exist; skips otherwise).
3. ORBIT2-specific assertion in `tests/`: feral-amf on
   `ORBIT2_0000.mtx` must produce nnz_L ≤ 120,000 (10% margin above
   MUMPS HAMF4's 109,782).

### Risk and mitigation

- **Risk**: MUMPS oracle harness is non-trivial Fortran/C glue.
- **Mitigation**: model on the existing `ssids_oracle/`. Keep
  scope to one MUMPS run per matrix; no per-matrix solve, just
  analyze + dump.

## Phase D — Wire-up and corpus validation (1 session)

Goal: make AMF the default ordering for SYM=2 N≤10000 in the
symbolic pipeline (matching MUMPS), re-run the corpus, audit
regressions.

### Deliverables

1. Extend `OrderingMethod` enum in `src/symbolic/mod.rs` with
   `Amf`. Wire `feral-amf` into the dispatcher.
2. `MetisND`-or-`Amf` heuristic in `symbolic_factorize` matching
   MUMPS's `ana_set_ordering.F:52-78` rule (Amf for N≤10000 SYM=2,
   MetisND otherwise — *but* keep MetisND/AMD available as opt-ins
   for diagnostic comparison).
3. Re-run corpus with `FERAL_KKT_ROOTS=all`. Compare against the
   2026-04-27 baseline log:
   - Geomean nnz_L ratio (feral / MUMPS HAMF4) per cluster.
   - ORBIT2 must be ≤ 120k.
   - COSHFUN, CATENA, CHAINWOO clusters checked individually.
4. CHANGELOG entry under [Unreleased].
5. Session checkpoint per CLAUDE.md.

### Risk and mitigation

- **Risk**: AMF is *worse* than MetisND on some clusters
  (METIS-friendly large meshes). MUMPS's heuristic of switching at
  N>10000 exists for a reason.
- **Mitigation**: keep MetisND as the default for N>10000;
  rerun the corpus against per-cluster heuristics; if a cluster
  regresses by > 20% nnz_L vs the pre-AMF baseline, tighten the
  threshold.

## Out of scope

- HAMF4 halo machinery (V1 boundary preservation). We always pass
  `LEN(i) ≥ 0`; the halo branch is dead code.
- AMF1/AMF2/AMF3 variants (which deduct only some prior cliques).
  AMF4 deducts all and is what MUMPS ships; we ship only AMF4.
- Parallel AMF. The MUMPS HAMF4 is sequential; parallel ordering is
  a separate research direction.
- Tuning `PAS` or `NBBUCK`. Fixed values from the MUMPS source.

## CLAUDE.md compliance

- **Research note first**: ✅ `dev/research/amf-clean-room.md`.
- **Plan note**: ✅ this file.
- **Tests-first**: each Phase B fixture has hand-derived expected
  output before the implementation lands.
- **Pure Rust, MIT, clean-room**: implementation derived from
  Amestoy 1999 thesis + AMD 1996 paper. MUMPS source read for
  algorithmic understanding only; no code copy.
- **No unwrap/expect in src/**: enforced by clippy + grep.
- **Bit-parity preservation**: Phase A gate is feral-amd byte-for-
  byte unchanged; existing kkt sidecars (if any) are the oracle.

## Next concrete step

Phase A. Begin with `cargo new --lib crates/feral-ordering-core`,
copy the AMD quotient-graph machinery from `feral-amd`, define the
`Metric` trait, port `MinDegree` as the first impl, re-route
`feral-amd` to the core. Stop the moment the workspace tests pass
unchanged.
