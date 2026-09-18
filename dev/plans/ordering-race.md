# Plan: measured ordering race in `Solver`

**Issue:** #203. **Research:** `dev/research/issue-203-auto-routing-2026-09-17.md`.

## Why a race and not a predicate

Three routing predicates were checked against every matrix measured on
2026-09-17 and all three mispredict: `nnz_L` (arms within 1.5% while
wall-clock differs 3.52x), `flop_proxy` (predicts 1.37x against a measured
3.52x, sign wrong at nfe=24), `max_front` (right on 3 of 6). The dominant
term at nfe=48 is a *scheduling* effect — `Amf`'s 2288-wide fronts serialise
where `MetisND`'s 1293 do not — and no symbolic quantity measured captures
it.

A race needs no predictor. It cannot be wrong the way a heuristic is wrong;
its worst case is spending the race cost to pick a near-tie.

## Why the economics work now

The 2026-05 fill-guarded race was rejected on two grounds: it guarded on
fill (now shown to be anti-correlated with speed) and it was priced as
per-solve overhead. In an IPM the symbolic is computed **once** and reused
across every iterate — pounce does exactly this
(`pounce-feral/src/lib.rs:17`) — so a one-time race over `k` arms amortises
across ~100 factorizations. At k=3 that is ~2% of the run, against a
measured 3.5x on the arm it would have picked.

For a caller that factors once, the race is a straight 2-3x loss. **It is
therefore opt-in and defaults off.**

## Design

```rust
Solver::with_ordering_race(arms: Vec<OrderingMethod>)   // opt-in, default none
Solver::with_race_margin(f: f64)                        // default 0.05
Solver::last_race() -> Option<&RaceResult>              // what happened
```

On the first `factor()` for a pattern, and only then:

1. For each arm build a **probe** `Solver` via `config_clone()` — every
   builder-settable field copied, all cached state empty, `race_arms`
   cleared so a probe can never recurse.
2. Probes share `self`'s `parallel_pool` (an `Arc`), so the race does not
   build `k` thread pools and every arm is timed on the same threads.
3. Factor the matrix on each probe; record wall time and inertia.
4. Arms that fail are excluded. Arms that succeed must **agree on inertia**;
   disagreement is recorded in `RaceResult` and the race declines to choose
   (keeps the configured ordering) rather than picking blind.
5. Winner = fastest, unless it beats the runner-up by less than
   `race_margin`, in which case the arm **earliest in the caller's list**
   wins. Ties go to the caller's stated preference, not to timing noise.
6. `self.ordering` becomes the winner and the winner's *symbolic* is adopted
   (analysis measured at 3.7-12.6 s on the issue #203 matrices, so this is
   worth moving rather than recomputing). The numeric re-runs on the real
   solver, so the race costs `k` extra factorizations, not `k+1` plus `k`
   analyses.
7. Race state is **pattern-bound**: cleared alongside the symbolic cache on
   a fingerprint miss, so a new pattern re-races.

## Tests, written first

Selection is extracted as a pure function so it can be tested without timing:

* `pick_race_winner` returns the fastest arm when the margin is cleared;
* returns the earliest-listed arm when the spread is inside the margin;
* skips failed arms; returns `None` when every arm failed;
* returns `None` (decline) when successful arms disagree on inertia.

End-to-end, on a saddle-point KKT with an analytic inertia oracle:

* racing `[Amd, Amf]` yields the same inertia and the same solution (to
  1e-10) as each arm run alone — the race changes speed, never the answer;
* the race runs **once**: two `factor()` calls on the same pattern perform
  the race on the first only (`symbolic_call_count` pins this);
* a changed pattern re-races;
* fewer than two arms is a no-op and leaves `ordering` untouched;
* a probe inherits configuration — `with_parallel(false)` plus a race still
  produces the oracle inertia;
* `last_race()` reports one entry per arm with the winner marked.

## Acceptance

* every test above green;
* on the issue #203 nfe=48 matrix, a `[Amf, MetisND]` race picks `MetisND`
  (measured 3.52x) and the total cost of `factor()` #1 is under 3x a single
  un-raced `factor()`;
* `cargo test --workspace` green, fmt and clippy clean;
* default behaviour bit-identical: with no `with_ordering_race` call, no new
  work runs on any path.

## Non-goals

* Changing `choose_adaptive`. The race is opt-in and orthogonal; the #67/#73
  question still needs the corpus.
* Racing scaling strategies, or anything other than the ordering.
* Re-racing on numeric drift within one pattern.
