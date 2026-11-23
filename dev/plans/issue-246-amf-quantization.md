# Plan — issue #246 step 1: monotone AMF score quantization

Research: `dev/research/issue-246-amf-quantization.md`.

## Change

In `crates/feral-ordering-core/src/quotient_graph/algo.rs`:

1. Extract the RMF → bucket-key quantization into a pure function
   `amf_quantize_score(rmf: f64) -> i64`. It returns `max(1, round(rmf))`,
   saturating at `AMF_SCORE_CAP` (1e18, which stays far from `i64`
   overflow even after the supervariable max-merge). There is no `/N`
   fall-through.
2. Call it from the `run_elimination_amf` re-insertion loop. Remove
   `AMF_DUMMY_I32` and update the doc comments that describe the
   `i32::MAX − 1` bound (the `algo.rs` step list and `Workspace::wf`).

Nothing else changes. `amf_bucket_of` already maps any key above the
coarse range to `NBBUCK`, and `select_pivot_amf` already linear-scans
coarse buckets by exact key.

## Tests (written first, must fail on the current code)

- `feral-ordering-core` unit tests (oracle: hand arithmetic):
  - `amf_quantize_score(2.0e9) == 2_000_000_000` and
    `amf_quantize_score(3.0e9) == 3_000_000_000`. The current code yields
    `round(3e9 / n)`, which for fhnw's n = 1 914 525 is 1 567.
  - Monotone across the `i32::MAX` boundary. The bucket index is also
    monotone: `amf_bucket_of(q(3e9), n) >= amf_bucket_of(q(2e9), n)`.
  - Floor: `q(0.4) == 1`, `q(-5.0) == 1`. Saturation: `q(1e30) == CAP`.
- Fixture-gated integration test `tests/issue246_amf_quantization.rs`:
  AMF nnz(L) ≤ AMD nnz(L) on the fhnw KKT (oracle: the AMD reference
  ordering). It SKIPs when `tests/data/large/fhnw-binschedule1.mtx` is
  absent, which is the case on CI. Regenerate the fixture with
  `dev/scripts/mps_to_kkt.py`.

## Validation

- `cargo test` (workspace), including `feral-amf`'s `expected_perm.rs`.
  Those are small matrices where `RMF < 2^31`, so they must be unchanged.
- Re-run the fhnw and Primal2 prefix probes: expect AMF nnz(L) 1.835e8 and
  1.802e8.
- `cargo run --bin bench --release` at session end.

## Out of scope (later steps of #246)

- Primal2 `avg_deg < 5 → AMD` routing (needs the powerflow22 check).
- A `max_factor_bytes` budget before `symbolic_finish`.
