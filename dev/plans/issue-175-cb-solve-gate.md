# Plan: issue #175 — per-call-overhead term for the tree-parallel solve gate

Research: `dev/research/issue-175-cb-solve-gate-overhead.md`.

1. **Reproduce locally.** Build a NARX-shaped fixture family whose front
   size is a parameter, and time the *pooled* CB core serial vs
   tree-parallel — the exact choice `CbTaskPlan::worthwhile` makes —
   across worker counts. (`solve_sparse_refined_cb` is the wrong probe:
   its residual sweeps dilute a per-front effect.)
2. **Locate the crossover** in work per front, and keep the harness
   in-tree as an `#[ignore]`d test so the constant can be re-derived.
3. **Split the gate** into a shape half (shared with
   `cb_core_profitable`) and a scheduling-only overhead half. The
   overhead half must not reach core selection — #177's contract.
4. **Tests first, from the fixture measurements:** a wide-thin tree that
   passes the shape half must not be scheduled in parallel; the same
   tree with real fronts must still be; the shape half must ignore front
   size. Build them on synthetic `NodeFactors` so they run in a debug
   build.
5. **Verify** the full suite, fmt, clippy, and that the CB parity and
   host-invariance tests (#131, #177) still pass — the change must not
   move a bit.
6. Record in decisions.md, tried-and-rejected.md, CHANGELOG.md, the
   journal, and the session checkpoint.
