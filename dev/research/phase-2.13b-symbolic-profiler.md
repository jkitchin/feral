# Phase 2.13b — Per-Stage Symbolic Profiler

## Motivation

KIRBY2_0007 (n=458) under `Renumber` default:
- numeric phase: 235 µs (1.8× MUMPS — fine)
- symbolic phase: 924 µs (~6× MUMPS's *entire* factor)

The bench's `factor_us` rolls symbolic + numeric, so the headline 9.5×
ratio is dominated by analyze-phase setup, not kernel work. The 924 µs
on n=458 is ~2 µs/row, which is large. We need a per-stage breakdown
to know which symbolic stage(s) carry the constant before we can pick
between caching, stage-skipping, or shrinking a single dominant stage.

## Stages in `symbolic_factorize_with_method`

Reading `src/symbolic/mod.rs:344-543`, the stages worth timing are:

| stage              | code location          | brief                                    |
|--------------------|------------------------|------------------------------------------|
| `symmetric_pattern`| 369                    | `matrix.symmetric_pattern()`             |
| `pick_preprocess`  | 374-377                | resolve `Auto` → `None` / `LdltCompress` |
| `ordering`         | 378-401                | external ordering (AMD/METIS/SCOTCH) or LdltCompress |
| `permute1`         | 408                    | `permute_pattern(&full_pattern, &amd_perm)` |
| `etree_initial`    | 409                    | `EliminationTree::from_pattern`          |
| `postorder`        | 416                    | `postorder(&amd_etree)`                  |
| `perm_compose`     | 421-425                | compose perm + perm_inv                  |
| `permute2`         | 428                    | second `permute_pattern`                 |
| `etree_relabel`    | 438-447                | O(n) etree renumbering through postorder |
| `col_counts`       | 454                    | `column_counts_gnp`                      |
| `renumber`         | 475-507                | Phase 2.12 path: `predict_merges` + `biased_postorder` + rebuild (when bias non-empty) |
| `find_supernodes`  | 511                    | fundamental detection + amalgamation     |
| `small_leaf_groups`| 517-518                | Phase 2.9 leaf grouping                  |
| `peak_contrib`     | 521-523                | contrib sizes + peak memory simulation   |

13 stages. Each is an obvious candidate for the dominant constant on
small-n workloads.

## Design choice — reuse vs new type

Two options:

1. **Reuse `numeric::Profiler`.** Coopt the existing per-supernode
   timings vector to record per-stage timings instead. Rejected: the
   `nrow`/`ncol` fields and the front-size buckets are numeric-specific;
   bending them to mean "stage" pollutes the type.

2. **New `SymbolicProfiler` type, parallel structure.** Mirror the
   numeric `Profiler` API: `Option<Arc<Mutex<SymbolicProfiler>>>` field
   on `SupernodeParams`, `record_stage(name, us)` method, `report()`
   returns a per-stage breakdown. **Chosen.**

The cost: ~80 lines of new code. The benefit: clean separation from
numeric, no shoehorning, and the report is shaped for the question
("which stage dominates 924 µs on n=458?").

## API sketch

```rust
// src/symbolic/profiler.rs
#[derive(Debug, Clone, serde::Serialize)]
pub struct StageTiming {
    pub name: &'static str,
    pub us: u64,
}

#[derive(Debug, Clone, Default)]
pub struct SymbolicProfiler {
    stages: Vec<StageTiming>,
    total_us: u64,
}

impl SymbolicProfiler {
    pub fn new() -> Self { Self::default() }
    pub fn record(&mut self, name: &'static str, us: u64) {
        self.stages.push(StageTiming { name, us });
    }
    pub fn set_total(&mut self, us: u64) { self.total_us = us; }
    pub fn stages(&self) -> &[StageTiming] { &self.stages }
    pub fn total_us(&self) -> u64 { self.total_us }
    pub fn report(&self) -> SymbolicProfileReport { ... }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SymbolicProfileReport {
    pub total_us: u64,
    pub accounted_us: u64,
    pub overhead_pct: f64,
    pub stages: Vec<StagePct>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct StagePct {
    pub name: &'static str,
    pub us: u64,
    pub pct_of_total: f64,
}
```

Wired on `SupernodeParams` as:

```rust
pub struct SupernodeParams {
    // ... existing fields ...
    pub symbolic_profiler: Option<Arc<Mutex<SymbolicProfiler>>>,
}
```

When `None`, every timer is conditional via
`params.symbolic_profiler.as_ref().map(|_| Instant::now())`, identical
to the numeric `Profiler` zero-overhead pattern.

## Validation strategy

Test fixture: a tiny (n=10) symmetric matrix. Run with profiler
attached; assert:
- `report.stages.len() == 13` (or N stages).
- All stage names present.
- `accounted_us <= total_us` and `overhead_pct < 100`.

Then a probe binary (`src/bin/diag_symbolic_stages.rs`) runs on
KIRBY2_0007 + MUONSINE_0000 (5-run median per matrix) and prints the
per-stage breakdown. The diagnostic answer is the dominant-stage row.

## Open question — reset between runs

Each `symbolic_factorize_with_method` call appends fresh stages to the
profiler. If a probe re-uses the same profiler across 5 runs (for
median computation), stages accumulate. Two options:
- (a) profiler caller is responsible for resetting between runs;
- (b) `SymbolicProfiler::new()` per run, then aggregate externally.

(b) is simpler and matches how the numeric profiler is used in
`src/bin/diag_strategy_compare.rs`. Go with (b).

## Out of scope for 2.13b step 1

- `SymbolicProfiler` does not break the call into sub-stages within
  a stage (e.g., AMD vs LdltCompress branch). It records the dispatched
  branch's total only. Sub-stage breakdown is a later iteration if step
  2's dominant-stage finding warrants it.
- The profiler is diagnostic, not a feedback signal. Phase 2.13a's
  `Auto` strategy will use shape predicates, not profiler output.

## References

- `src/numeric/factorize.rs:72-219` — numeric Profiler API template
- `dev/plans/phase-2.10-supernode-profiler.md` — Phase 2.10 numeric profiler
- `dev/plans/phase-2.13-tail-diagnostic.md` — parent plan for 2.13a + 2.13b
- `dev/journal/2026-04-25-03.org` 20:00 — diagnostic finding motivating 2.13b
