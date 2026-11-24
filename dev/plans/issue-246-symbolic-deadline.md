# Plan — interrupting the symbolic phase; race widening at n ≥ 1M

Research: `dev/research/issue-246-symbolic-deadline.md`.

## Tests first

Oracle: these test control flow, not numerics. The oracles are "an
already-expired deadline / a set flag must stop before any result", "no
interrupt changes nothing", and timing bounds measured against the
unbudgeted run's own wall time.

`feral-ordering-core` (`interrupt` module):
1. `Interrupt::none()` never triggers. `with_flag` triggers once the flag
   is set. `at(past)` triggers.
2. `scope` nests and restores the previous interrupt, and also restores it
   when the closure panics.
3. Sticky: after a trigger, clearing the flag does not un-trigger `poll()`
   inside the same scope.
4. `order::<MinDegree>` and `order::<MinFill>` under an expired interrupt
   return `Err(OrderingError::Interrupted)`.

Ordering crates (one test each, in each crate's tests):
5. `*_order_interruptible(pattern, opts, &expired)` returns `Interrupted`
   for amd, amf, metis, scotch and kahip, on a 2-D grid big enough to
   recurse (60×60).
6. With `Interrupt::none()` the result is bit-identical to `*_order_full`.

feral:
7. `symbolic_factorize_with_method` with an expired interrupt returns
   `FeralError::Interrupted` for Amd, Amf, MetisND, ScotchND, KahipND,
   AutoRace and Auto (with LdltCompress forced), and `FINISH_RUNS == 0`.
8. A flag set from another thread mid-run stops it. A 300×300 grid under
   MetisND, with the flag set after 1 ms, must return `Interrupted`, and
   much faster than the unbudgeted run. The assertion is only on the
   variant, plus elapsed < unbudgeted elapsed (no absolute time, so CI noise
   cannot flake it).
9. `Solver::with_interrupt(Some(expired))`: `factor` gives
   `FatalError(Interrupted)` and `symbolic()` is `None`. Then
   `set_interrupt(None)` and `factor` gives `Success`.
10. `None` leaves every existing test bit-identical. This is covered by the
    full suite.

Race widening:
11. `race_candidates(n)`: `{Amd, MetisND}` for n < 1M, and
    `{Amd, MetisND, ScotchND, Amf}` for n ≥ 1M (unit test on the function).

## Implementation order

1. `feral-ordering-core::interrupt` + `OrderingError::Interrupted` + the
   check in both elimination loops + `order_interruptible`.
2. amd/amf: `*_order_interruptible`. metis/scotch/kahip: `*_order_interruptible`
   plus safe-point polls (recursion entry `check()?`, levels/trials/passes
   `poll()` → break, a `check()?` after each `multilevel_node_bisection`).
3. feral: `SupernodeParams::interrupt`, `FeralError::Interrupted`, the
   OrderingError mapping, the pipeline checks, the Hungarian poll, the
   `Solver` builder/setter, Python error mapping, and the
   feral-diagnostics struct literals.
4. `race_candidates(n)`.
5. Demonstrate on fhnw: `Auto` with a 60 s deadline returns `Interrupted`
   within a few seconds of the deadline. Report the overshoot.
6. CHANGELOG, book, decisions.md (race widening is a behaviour change).
