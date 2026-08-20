# FERAL Context (auto-generated)

Generated: 2026-08-20T02:05:49Z

## Latest Session
File: dev/sessions/2026-08-19-04.md
```
# Session 2026-08-19-04

## BENCHMARK NUMBERS ARE NOT COMPARABLE TO LAST SESSION

Reported first, per the hard rule in CLAUDE.md — and for the same reason
as 2026-08-19-02: `cargo run --bin bench --release` in this container
finds only the 8 synthetic matrices. The external corpus and the
MUMPS/SPRAL oracle timings are absent, so both Phase 2.8.1 exit
partitions report `N/A` and **no comparison against 2026-08-15-02's
1.61 / 2.00 / 1.67 / 1.67 is possible from this run**. I am not claiming
those numbers held; they were not measured.

What the run does confirm, identical to last session: inertia 2/2 vs
MUMPS, residual 2/2, worst residual 1.26e-16.

It is also the wrong benchmark for this change, which touches solve
*scheduling* and not the factor path. The measurements that matter are
in "Benchmark Results" below.

## Goal

Fix issue #175 — the tree-parallel solve added for #131 Gap A is a net
15% loss on the wide-sparse Mittelmann KKT `NARX_CFy`, and
`CbTaskPlan::worthwhile` has no per-call-overhead term.

## Accomplished

### The report is real, and it is purely a scheduling bug

The reporter serialized the tree-parallel solve with `FERAL_CB_THRESH`
and recovered 7.35 s of 49.41 s (15%) plus ~3.0M involuntary context
switches, on 14 cores over 100 IPM iterations. Two things follow that
the issue does not state:

1. `FERAL_CB_THRESH` does not choose the solve *core* — since #177 that
   is `cb_core_profitable`, which reads neither the worker count nor the
   environment. A huge threshold collapses the plan to one task, so the
   **same** CB core runs serially. The whole 7.35 s is scheduling
   overhead, and fixing it cannot move a bit of any solve.
2. Rows 3 and 4 of the reporter's table are within noise, so the CB core
   running serially already matches switching parallelism off entirely
   on this problem. Nothing in the report argues for changing which core
   runs.

### Mechanism: the overhead is per front, not per task

`cb_run_parallel` takes the shared `contribs` mutex inside its per-front
loop — once per child drained, once to store the front's own block. That
cost scales with the supernode count; `MIN_TOTAL_COST` is a floor on
*total* work. `NARX_CFy` has 45,736 supernodes and a Lagrangian Hessian
```

## Git Status
```
2fbf9b7 Merge pull request #186 from jkitchin/release/0.17.0
5082eff release: 0.17.0
a91082b Merge pull request #185 from jkitchin/fix/pre-release-0.17.0
d758cd8 docs(journal): session 2026-08-19-05 pre-release review entries
8b97b21 docs: correct pre-release claims across CHANGELOG, README and rustdoc
```

## Test Status
```
