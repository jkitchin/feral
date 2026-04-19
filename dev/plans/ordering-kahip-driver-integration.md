# Plan: KaHIP driver integration

**Research note:** `dev/research/ordering-kahip-driver-integration.md`
**Recommendation from research:** ship option A (status quo) — do
not modify the dispatcher; document why in the OrderingMethod
docstring; defer option C until a measured workload demands it.

## Steps

### 1. Document current state in OrderingMethod docstring

Edit `src/symbolic/mod.rs` around the `KahipND` enum variant to add
a paragraph explaining that the dispatcher (`pick_default_method`)
does not select KaHIP because:

- K1+K2-K6 ties METIS on fill at 4-6× the per-call cost on the
  41-matrix shape bake-off (geomean fill ratio 1.023 vs METIS 1.024;
  total symbolic time 81s vs METIS 68s vs AMD 14s).
- KaHIP remains reachable via `symbolic_factorize_with_method` for
  callers who want it explicitly.
- `OrderingMethod::Auto` is the multi-method dispatcher and includes
  KaHIP in its decision tree.

Cross-reference the research note. No behavior change.

### 2. Add a single regression test that pins the decision

Add `kahip_default_method_is_not_kahip` to the existing
`pick_default_method_rules` test module: assert that on a synthetic
matrix shaped like CRESC132 (n=5314, nnz/n=4.25), the chosen method
is `MetisND` and NOT `KahipND`. This pins the recommendation: if
some future change wants to route to KaHIP by default, the
maintainer must consciously update the test and read the cross-
referenced research note explaining why.

The test goes one level above the existing `MetisND` assertion in
`pick_default_method_rules` so the two coexist.

### 3. Update CHANGELOG with the planning outcome

Append an Unreleased entry summarizing the decision, pointing to
the research note and plan. No code-level "Changed" — this is a
documentation-only change.

### 4. Append the next-session list

Add the deferred follow-ups to the session-08 checkpoint
"Next session should":

- Lift K1 as a generic preprocessor for AMD/METIS, contingent on
  a measured workload demonstrating arrow-KKT fill wins. Defer
  until such a workload exists.
- Re-evaluate the recommendation if/when a new corpus is added
  (compressed-sensing, SDP, HEMSR MILP) that exhibits leaf-heavy
  patterns AMD handles poorly.

## Out of scope this session

- **K1 as generic preprocessor in front of AMD.** Could be
  ~50 lines: extract `crates/feral-kahip/src/data_reduction.rs`
  helpers, expose at the workspace level, add a
  `symbolic_factorize_with_preprocessing` entry point. Deferred:
  needs a measured workload first.
- **KaHIP performance tuning** (per-call setup cost reduction).
  The 5.6× per-call gap vs AMD is a KaHIP-internal concern
  unrelated to the dispatcher decision.
- **Auto dispatcher revival.** Still rejected per session 04
  evidence (regresses geomean from 0.44 to 0.58 on IPM corpus).
  No new evidence to revisit.
- **HEMSR coarsening** (priority #4 from session 07). Separate
  research thread.

## Tests

Existing tests must continue to pass unchanged:

- `pick_default_method_rules` (extended with new assertion)
- All tests in `crates/feral-kahip/`
- `cargo test --release` full suite

No bench impact expected — no code path changes.

## Estimated effort

15 minutes implementation + commit. This is an explicitly
documentation-and-pinning session.
