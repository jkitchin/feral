# Plan: issue #227 — `ReanalysisPolicy` (re-analyse a stale value-dependent analysis)

Research: `dev/research/issue-227-stale-analysis-reanalysis.md`. Read it
first; the numbers and the design in §7 of that note are what this plan
implements. Scratch root `$J` =
`/private/tmp/claude-501/-Users-jkitchin-projects-feral/a1458847-2377-4832-a0c1-096910fb944e/scratchpad/i227/`
(harness `$J/r227`, init matrices `$J/init-*.jsonl`, sequence lists
`$J/seq-*.txt`, dumps `$J/dump/`, prototype `$J/r227/src/main.rs::proto`).

Hard constraints (do not relitigate): inertia exact on non-singular
matrices; blocked/scalar and parallel/sequential bit-parity
(`tests/blocked_ldlt.rs`, `tests/golden_bits.rs`, `tests/parallel_parity.rs`);
no tolerance loosening; `Never` bit-identical to main; nothing from
`dev/tried-and-rejected.md` 2026-10-10 (no factor-comparison heuristics
*for inertia* — this policy compares cost and requires equal inertia, it
never chooses a count) or 2026-10-20/21.

Measurement rules: fixed sequences before end-to-end; sequential runs
with nothing else on the machine; pounce runs with a 300 s cap. Replay
with `$J/r227` pointed at the working tree (`Cargo.toml` already does),
or with the new `feral` API directly once Step 2 lands (`r227
--policy onstale` should then be re-pointed at `Solver::with_reanalysis`
instead of its own emulation — do that in Step 5).

## Design in one paragraph

A `Solver` under `ReanalysisPolicy::OnStale` keeps, next to the cached
analysis, a store of at most two value-dependent (`LdltCompress`)
analyses it has displaced or probed. Three events open a trial on the
current matrix, all evaluated only at the base pivot threshold and never
while a quality ladder is active: (1) the stored factor's actual flops
exceed `trigger_ratio` × the analysis's predicted flops; (2) the factor
index on the pattern hits the probe schedule (powers of two from 2) and a
fresh analysis predicts less than `predicted/(1+margin)`; (3) the #102
ordering escalation fires. Candidates (fresh analysis, stored analyses)
each factor the same matrix once; the cheapest by actual flops is adopted
iff it beats the incumbent's actual flops by `margin`, reports the same
inertia, and has no more floored/zero pivots. Declined value-dependent
candidates and the displaced incumbent go to the store. Trials per
pattern are capped. There is no rollback: the store is the rollback.

## Step 0 — actual-work counter and its exposure (no behaviour change)

Files: `src/symbolic/supernode.rs` (`front_flops` is private; make it
`pub(crate)` and add `pub fn predicted_front_flops(&SymbolicFactorization)
-> u128` in `src/symbolic/mod.rs`), `src/numeric/factorize.rs` (add
`SparseFactors::actual_front_flops(&self) -> u128 = Σ front_flops(nf.nelim,
nf.nrow)`), `src/numeric/solver.rs` (`FactorStats` gains
`predicted_flops: u128` and `actual_flops: u128`; `WorkEstimate` is
unchanged — its `factor_flops` is the `ncol·nrow²` proxy and pounce reads
it).

Tests first (oracles are hand sums):
- `front_flops` already has `front_flops_matches_hand_summation`; add
  `predicted_front_flops` on `tridiag_csc(5, ..)` (one supernode per
  column: `Σ_{j} (n−j)²` by hand) and on a 2-supernode arrow matrix.
- `actual_front_flops` on `tests/fine_grained_delay.rs`'s fixture where a
  pivot is delayed from a child to its parent: compute by hand from the
  known `nelim`/`nrow` of the two fronts (the test already asserts those).
- `actual == predicted` on any SPD tridiagonal (no delays, by
  construction), and for every matrix in `tests/data/parity` the ratio is
  identical between `with_parallel(true)` and `(false)` and between two
  runs (determinism).

Accept: tests pass, `cargo test` unchanged elsewhere, `FactorStats`
construction updated in `tests/issue52_stats.rs` (exhaustive struct —
this is a breaking change; bump to 0.22.0 in Step 5).

## Step 1 — extract the numeric dispatch (pure move, bit-identical)

File: `src/numeric/solver.rs::factor_impl`.

Move the "pool or sequential" dispatch — the `if let (true, Some(p)) =
(self.use_parallel, pool.as_ref()) { p.install(|| parallel(..)) } else {
sequential(..) }` block that appears twice (initial factor and the #65
retry) — into `fn dispatch_numeric(&mut self, matrix, symbolic:
&SymbolicFactorization, params: &NumericParams, pool: Option<&Arc<ThreadPool>>)
-> Result<(SparseFactors, Inertia), FeralError>`. Nothing else moves.
`begin_recycle` stays where it is (called once, before the initial
factor).

Test first: `tests/issue227_never_golden.rs` — replay a fixed sequence
under `Solver::new()` and assert a hardcoded digest of every factor's
`l`, `d_diag`, `d_subdiag`, `perm` and the inertia. Compute the digests
**on main before Step 1** (`golden_bits.rs::digest` is the hash), commit
them with the test, and keep the test unchanged through Step 5. Sequence:
a synthetic 8-iterate KKT path built in-test from a seeded generator
(`[[D_k, Jᵀ],[J, −δI]]`, `D_k` positive diagonal with entries drifting
over 12 orders of magnitude, `J` full row rank, n ≈ 400) plus
`tests/data/parity/*` matrices factored twice each. The synthetic KKT's
inertia is `(n_x, m, 0)` by Sylvester/Haynsworth (`D` > 0, `J` full rank,
`−δI − J D⁻¹ Jᵀ` negative definite), an external oracle the test also
asserts.

Accept: golden test passes before and after the move; `cargo test`
green.

## Step 2 — policy types, state, store, report; `Never` wired through

Files: new `src/numeric/reanalysis.rs` (re-exported from `lib.rs`),
`src/numeric/solver.rs`.

Types:

```rust
pub enum ReanalysisPolicy { Never, OnStale(OnStaleParams), Always }
pub struct OnStaleParams {
    pub trigger_ratio: f64,        // 2.0
    pub margin: f64,               // 0.25
    pub max_trials_per_pattern: u32, // 8
    pub max_stored: usize,         // 2
    pub probe_first: u32,          // 2   (0 = no probes)
    pub probe_step: u32,           // 4   (probe at first, first·step, first·step², …)
    pub probe_last: u32,           // 2048
    pub min_flops_per_nnz_for_probes: u64, // 100: probe only while predicted_flops / nnz(A) >= this
}
pub enum ReanalysisEvent { Cost { ratio: f64 }, Probe { k: u32 }, Escalation, Always }
pub struct TrialOutcome { origin: AnalysisOrigin, predicted: u128, actual: u128,
                          inertia: Inertia, n_tiny: usize, n_zero: usize, adopted: bool }
pub struct ReanalysisReport { event: ReanalysisEvent, incumbent_actual: u128,
                              trials: Vec<TrialOutcome>, declined: Option<DeclineReason> }
pub enum DeclineReason { Identical, NotCheaper, InertiaDiffers, MorePivotsGivenUp, CapReached, Ladder }
```

Solver fields: `reanalysis: ReanalysisPolicy`, `reanalysis_state:
ReanalysisState { stored: Vec<StoredAnalysis>, trials_used: u32,
factor_index: u32, cost_latched: bool, adoptions: u32, last_report:
Option<ReanalysisReport> }`, `StoredAnalysis { sym: SymbolicFactorization,
origin, predicted: u128 }`. Builders: `Solver::with_reanalysis(policy)`,
accessors `reanalysis_policy()`, `last_reanalysis()`.

`FactorStats` gains `reanalysed: bool` (this call adopted a different
analysis) and `reanalysis_count: u32` (adoptions on this pattern so far),
next to `pattern_reused` (which stays `true` on such a call — the pattern
*was* reused; #215 made that field mean exactly that).

Reset rule on a pattern change (the `!cache_valid` block): the whole
`reanalysis_state` is reset. On adoption: see the table in the research
note §8 — reset `prev_max_n_delayed_in` and `auto_arm_latched`; everything
else untouched.

`Never`: no field is read on the hot path except one `matches!` after the
factor is stored. The golden test from Step 1 must still pass.

Tests: builder round-trip; `last_factor_stats().reanalysed == false` and
`reanalysis_count == 0` under `Never`; report `None` under `Never`.

## Step 3 — cost and probe events with the same-matrix trial

File: `src/numeric/solver.rs` (hook after the `Ok((factors, inertia))`
branch has stored the factor and *after* the escalation block, so an
escalation re-entry has already returned), `src/numeric/reanalysis.rs`
(decision logic as pure functions over integers, so they are unit-testable
without a factorization).

Guard, in order: policy is not `Never`; the stored factor exists;
`self.ladder_start.is_none() && self.rung_before.is_none()`; the call
is not an escalation re-entry; `trials_used < max_trials_per_pattern`.

Cost event: `ratio = actual_front_flops / predicted_front_flops` of the
stored factor against `last_symbolic`; fire if `ratio > trigger_ratio
&& !cost_latched`.
Probe event: `factor_index ∈ {probe_first · probe_step^j} ∩ [probe_first,
probe_last]` (2, 8, 32, 128, …; the prototype used powers of two and its
adoptions were at k = 2, 4 and 8 — the k = 4 one moves to 8, research note
§7) and `predicted_front_flops(last_symbolic) / matrix.nnz() >=
min_flops_per_nnz_for_probes` (research note §6: 42 on clnlbeam and 90
on pooling, where an analysis costs 4–6 factors, against 1 450 on NARX
and 1 600 on qcqp; the guard is exact integer arithmetic, so it is
deterministic).
`Always`: every factor is a probe (same gate).

Fresh candidate: `symbolic_factorize_with_method(matrix, &effective
snode_params (LdltCompress forced if `ordering_escalated`), self.ordering)`.
Decline `Identical` without a trial when both it and the incumbent
resolved `None`. For a probe, trial only if `fresh.predicted · (1+margin)
< incumbent.predicted`; otherwise store it if `LdltCompress` (evicting
the oldest beyond `max_stored`) and stop. For a cost event, trial the
fresh candidate if `fresh.predicted · (1+margin) < incumbent.actual`,
and trial every stored analysis.

Trial = `dispatch_numeric(matrix, &candidate.sym, &effective_params,
pool)` with no recycling (do **not** call `begin_recycle`; the trial's
output buffers are fresh allocations) and no #65 retry; a trial that
errors is discarded. Compare with
`better(cand, inc) = cand.actual·(1+margin) < inc.actual && cand.inertia
== inc.inertia && cand.n_tiny + cand.n_zero <= inc.n_tiny + inc.n_zero`.
Adopt the best; on adoption swap `last_symbolic`/`last_factors`/
`last_inertia`, push the displaced incumbent to the store (if
value-dependent), recompute `FactorStatus` from the adopted inertia and
`check_inertia`, set `reanalysed`, bump `reanalysis_count`,
`symbolic_call_count` (+1 per analysis computed), clear `cost_latched`.
On decline from a cost event set `cost_latched = true` (cleared only by
adoption or pattern change). Every trial increments `trials_used`.

Tests first:
- Decision functions: table tests of `better` and the probe/cost gates on
  hand-written integer inputs (every `DeclineReason` reachable).
- Synthetic sequence from Step 1 with `OnStale { trigger_ratio: 0.5, .. }`
  so the cost event fires on factor 2: the fresh analysis is pattern-only
  → `DeclineReason::Identical`, no trial, `trials_used == 0`,
  `cost_latched` set, and the factor is **bit-identical** to the `Never`
  run (digest). Inertia `(n_x, m, 0)` on every factor (Sylvester oracle).
- A sequence where the fresh `Auto` analysis flips to `LdltCompress`:
  build the synthetic KKT so the predicate fires (≥ 30 % degree-≤2
  columns: add slack rows) and the first matrix has 70 % explicit zeros
  in `W` (as pounce's init does) so the race picks `None`, then the
  second matrix has a dense `W`. With `probe_first = 2`, the probe at
  k = 2 sees `LdltCompress`; assert it is stored or trialled according
  to its predicted flops, that `reanalysed` is `true` exactly on an
  adopting call, and that every factor's inertia is `(n_x, m, 0)`.
- Ladder exclusion: call `increase_quality()` between factors; while
  `ladder_start.is_some()` no event fires (`last_reanalysis()` reports
  `Ladder`), after the ladder is undone events resume.
- Determinism: the same sequence in two `Solver`s and with
  `with_parallel(true/false)` gives identical reports (`Debug` string)
  and identical factor digests.
- Cap: `max_trials_per_pattern = 1` → at most one trial; `CapReached`
  afterwards.
- No `unwrap`/`expect` in `src/`; errors from a candidate analysis are
  `FatalError` only if the *incumbent* path failed, never from a trial.

Accept: all above green; golden `Never` test unchanged.

## Step 4 — escalation (#102) as an event

File: `src/numeric/solver.rs` escalation block.

Under `Never` the block is untouched (bit-identical). Under `OnStale`/
`Always`, after the re-entry returns with the fresh `LdltCompress` factor
stored, run the stored `LdltCompress` candidates as trials against it
and adopt one only under the same `better` gate (fresh is the default).
Record `ReanalysisEvent::Escalation`. The fresh escalated analysis counts
as `origin: Escalation` in the store when displaced.

Tests: `tests/issue102_ordering_escalation.rs` unchanged under `Never`;
under `OnStale` with a stored `LdltCompress` analysis that is cheaper on
the escalation matrix and has the same inertia, the stored one is
adopted (`last_reanalysis()` shows `Escalation` with two trials). Fixture:
the cont5 late KKT used by that test plus an earlier cont5 iterate if
present in `data/matrices/kkt/` (the test already skips when absent);
otherwise the synthetic sequence with a growth spike (scale one `D`
entry to 1e-25).

## Step 5 — replay acceptance, end-to-end report, release notes

1. Re-point `$J/r227 --policy onstale` to `Solver::with_reanalysis(OnStale
   (default))` (keep `proto` for comparison) and run, sequentially:
   - qcqp1000-2c init-first and kkt-first, qcqp1000-2nc both orders,
     clnlbeam init-first, NARX_CFy init-first, pooling_rt2stp init-first,
     each under `Never` and `OnStale` (defaults).
   - Gate (summed over each sequence):
     - `Never`: every factor bit-identical to main (digest of
       `l`/`d_diag`/`d_subdiag`/`perm`, inertia) — produce the main
       digests with the harness before merging.
     - `OnStale` inertia: equal to `Never`'s on every factor where
       `Never` reports `n_tiny == 0`; equal to the dump's
       `num_neg_evals_expected` wherever `Never` is.
     - `OnStale` cost (actual flops summed, and sequential time; the
       reference is `Never` on the *same* call order, e.g. the kkt-first
       reference is what pounce#1046 measured):
       qcqp1000-2c init-first ≤ 0.80 × `Never` in flops (prototype: 0.75,
       `$J/res/qcqp2c.proto2.initfirst.txt`) and ≤ 0.65 × in time;
       qcqp1000-2c kkt-first ≤ 1.00 × `Never` in flops (prototype 0.98)
       and ≤ 0.75 × in time; qcqp1000-2nc init-first = 1.00 × `Never` in
       flops and ≤ 1.15 × in time (prototype with powers-of-two probes:
       1.16–1.22; the overhead is the analyses and two declined trials);
       qcqp1000-2nc kkt-first ≤ 0.60 × `Never` in flops (prototype
       0.55) and in time; clnlbeam, NARX_CFy, pooling = 1.00 × in flops
       and ≤ 1.05 × in time (the `predicted/nnz` guard turns probes off
       on clnlbeam and pooling; NARX keeps them at ≈ 3 % — if it exceeds
       5 %, raise `min_flops_per_nnz_for_probes` to 2 000 before anything
       else).
     - Trials per pattern ≤ cap on every sequence; analyses computed
       ≤ 12 per sequence.
2. pounce end-to-end (report, not gate): clone pounce @ 9958a24e into
   `$J/`, `[patch.crates-io] feral = { path = … }` as `i224/pounce-local`
   does, expose the policy through a `FeralConfig` option in the clone
   only (no pounce source committed), run the models above both ways,
   300 s cap, and report iterations, objective, `total_factor_secs`,
   `n_factors`.
3. `CHANGELOG.md` Unreleased: new policy, new `FactorStats` fields
   (breaking: exhaustive struct) → version 0.22.0; `dev/decisions.md`
   entry "re-analysis is judged by a trial factorization of the same
   matrix; predicted flops cannot rank analyses" with the §4.4 table as
   evidence; session checkpoint.

## Step 6 (optional, off by default) — level (a): scaling re-pick

Only if a dumped sequence shows `pick_scaling_strategy` flapping
(none of the five here does). Add `OnStaleParams::rescale: bool`; at a
cost event, before any analysis, re-run `pick_scaling_strategy(matrix)`;
if it differs from `auto_picked_strategy`, one trial with the new pick
under the incumbent analysis, same `better` gate, adopt by replacing the
pick. Test on a synthetic sequence whose diagonal mass flips the picker
(`tests/auto_strategy.rs` has the shapes).

## Kill criteria

- Step 1's golden test cannot be made to pass with the dispatch moved →
  stop; the hook has to be written without the refactor.
- Step 5 gate: `OnStale` changes the inertia relative to `Never` on any
  factor where `Never` has `n_tiny == 0` → stop, report the matrix, do
  not tune the gate.
- qcqp1000-2c init-first under the real implementation does not adopt
  the stored first-KKT analysis at the escalation (the prototype's
  mechanism) → stop and compare against `$J/res/qcqp2c.proto2.initfirst.txt`
  before changing anything.
- Overhead on clnlbeam / NARX_CFy / pooling > 5 % with default
  parameters → lower `probe_last` or raise `probe_first`; if still > 5 %,
  ship with probes off by default and record it.
- Any non-determinism between thread counts in a report or a factor →
  stop; the policy must only read exact integers and inertia.
