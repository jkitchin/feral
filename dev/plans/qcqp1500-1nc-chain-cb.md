# Plan: split cascade-break count by trigger

Research: dev/research/qcqp1500-1nc-chain-cb-2026-11-19.md

1. `NodeFactors` gains `cascade_break_chain: bool`. It is true iff CB fired
   and the #215 budget was not exceeded, i.e. only the #221 chain
   backstop tripped. All other constructors set it to false.
2. `SparseFactors::n_cascade_break_chain_nodes()` counts these nodes.
   `n_cascade_break_nodes()` is unchanged (total, both triggers).
3. `FactorStats::n_cascade_break_chain_nodes` is added, and mirrored in
   the Python `FactorStats` (field + repr).
4. The test asserts
   `n_cascade_break_nodes - n_cascade_break_chain_nodes == 0` (no
   budget/legacy-triggered CB), plus the existing exact inertia and
   `inertia_uncertain() == 0`. The tolerance change was approved by the
   user this session; record it in the checkpoint.
5. Verify: the qcqp1500 test passes, the ignored column25 test still
   passes, and the full `cargo test --release` passes. No solver
   behavior changes.
