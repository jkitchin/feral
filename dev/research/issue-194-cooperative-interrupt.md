# Issue #194 — cooperative cancellation for `Solver::factor`

Research note. Written before implementation, per CLAUDE.md §Work.

## The reported problem

A host (pounce) enforcing a wall-clock budget cannot enforce it across a
single `Solver::factor` call. Measured on `emfl050_5_5` (5675 vars, 5625
constraints): a `max_wall_time=5` solve returned `TIME_LIMIT` after
**48.8 s**. The between-operation deadline check passed while under
budget, one factorization then ran ~44 s with no way for the host to
regain control, and only the *next* check tripped.

`factor` takes no interrupt, deadline, cancel flag or progress callback;
`FactorStatus` has no variant that could carry an early return. The
existing `DelayBudgetExceeded` is a *symbolic-time column-count* budget
on the delayed-pivot cascade, not a wall-clock one, and it is armed by
default — it did not save this case.

## Why the host cannot fix this on its own side

Confirmed by reading the issue's account against this tree: `Solver`
holds `workspace: FactorWorkspace`, `last_symbolic`, `permute_cache` and
`parallel_pool` — mutable state reused across `factor` calls, which is
exactly the symbolic-cache reuse an IPM depends on. A thread-based
watchdog that abandons a factorization mid-flight either leaves those
buffers being mutated by an orphaned thread, or forces a fresh `Solver`
per factorization and destroys the reuse. Cancellation has to be
cooperative and inside feral.

## Where the time goes, and therefore where to poll

Two granularities matter, and only the coarser one is what the issue
asks for:

1. **Supernode boundaries.** Both drivers have a natural per-supernode
   loop: `factorize_multifrontal_supernodal_with_workspace`'s
   `while snode_idx < n_snodes`, and `run_parallel_task`'s
   `for &snode_idx in plan.owned[task_idx].iter().chain(once(&task_idx))`.

2. **Dense panel boundaries inside one supernode.** Supernode boundaries
   alone are *not* sufficient for the motivating case. Issue #8 records
   the mechanism the reporter cites: a delayed-pivot cascade concentrates
   118k delayed pivots into three ~14k-column expanded root fronts, so
   the 87 s factor is dominated by a handful of *individual fronts*. A
   check that only fires between supernodes would return "~5 s plus one
   44 s supernode", which is the bug again with extra steps.

   `factor_frontal_blocked_in_place_with_scratch` has one `while k <
   ncol_eff` loop that covers both the panel path (>= `PANEL_MIN_NCOL`
   columns of work per iteration) and the scalar tail (O(nrow) per
   pivot). The unblocked `factor_frontal_in_place_with_scratch_impl` has
   the same loop shape. Polling at the top of those loops is off the
   inner kernels — the loop body is at minimum O(nrow) work — and bounds
   the overshoot at one panel rather than one front.

So: poll at supernode boundaries *and* at dense panel boundaries. The
published contract stays the weaker of the two ("checks at supernode
boundaries, and within a supernode at dense panel boundaries; no
guarantee about when within a panel").

## Mechanism: `AtomicBool`, not a clock

Polling a caller-owned `Arc<AtomicBool>` rather than a deadline keeps
feral clock-agnostic and leaves wall-vs-CPU-vs-budget policy with the
caller. `Ordering::Relaxed` is the right load: the flag carries no data
dependency — we are not reading anything the setter wrote besides the
flag itself — and a missed observation only costs one more panel before
the next poll. Cost when unarmed is an `Option::is_some` branch that
predicts perfectly; when armed, a relaxed load of an uncontended cache
line.

feral only ever *reads* the flag. It never clears it — the caller owns
arming and disarming, so a re-arm after an interrupt is the caller's
`store(false)`.

## Plumbing: a new `FeralError` variant

The drivers already return `Result<(SparseFactors, Inertia), FeralError>`
and both already have an error path that unwinds correctly:

- sequential: `?` out of the supernode loop;
- parallel: the `first_error` mutex plus the fast-exit check at task
  entry, which drains the scope without doing further work.

So `FeralError::Interrupted` gets cancellation for free on both drivers,
including the "several tasks in flight" case, with no new control flow.
`Solver::factor` then maps it to a new `FactorStatus::Interrupted`.

`Solver::factor`'s existing `Err(e) => { last_factors = None; ... }` arm
already clears the stored factor and inertia, which *is* the contract the
issue asks for ("factors are left invalid; a subsequent factor re-runs
cleanly"). The new arm is a copy of it that returns `Interrupted`
instead of `FatalError`.

## API surface

Per the issue, matching the existing builder convention:

```rust
pub fn with_interrupt(self, flag: Arc<AtomicBool>) -> Self;
```

Adding two more, both cheap and both needed by the stated consumer:

- `set_interrupt(&mut self, Option<Arc<AtomicBool>>)` — the pounce
  wiring in the issue is `SparseSymLinearSolverInterface::set_interrupt(
  &mut self, ...)`, and a consuming builder cannot be called through the
  `&mut Solver` a backend handle holds. It is also the only way to
  *disarm*.
- `interrupt(&self) -> Option<&Arc<AtomicBool>>` — lets a test assert the
  arming without factoring.

The flag lives on `NumericParams::interrupt` (`Option<Arc<AtomicBool>>`,
default `None`), which is the single funnel `Solver::factor` already
clones into `effective_params` and hands to both drivers. Direct users of
the free-function `factorize_multifrontal*` API get it too.

C ABI: a new `FERAL_INTERRUPTED = 4` status code. There is no C-side way
to arm the flag in this change — the C ABI owns its `Solver` internally —
so the code is defined and mapped, but unreachable from C until a future
`feral_set_interrupt` is added. Mapping it now keeps the `match` in
`capi.rs` exhaustive and honest rather than folding an interrupt into
`FERAL_FATAL`.

## Oracle for the tests

No external numerical oracle is needed: this is a control-flow feature,
not a numerical one. The properties under test are

1. unarmed ⇒ byte-identical behaviour (same status, same inertia);
2. armed-and-set-before-`factor` ⇒ `Interrupted`, no factor stored,
   `solve` fails with `NoFactor`;
3. after clearing the flag, a subsequent `factor` succeeds and produces
   the same inertia as an un-interrupted run — the "re-runs cleanly"
   half of the contract;
4. armed-but-clear ⇒ `Success` (the flag is polled, not merely present);
5. both drivers (`with_parallel(true)` / `(false)`) honour it;
6. a flag set from another thread *during* a long factorization stops it
   early — the actual reported scenario.

(3) is the one that matters most: it is the property the reporter says
they will rely on, and the one a naive implementation (leaving a
half-written `workspace` or a stale `contrib_blocks`) would break.

## Rejected alternatives

- **A deadline (`Instant`) inside feral.** Explicitly not asked for, and
  it forces a clock choice (wall vs CPU) that belongs to the caller.
- **A progress callback.** Strictly more API surface for the same
  outcome, and a `dyn FnMut` in the supernode loop is harder to make
  zero-overhead than an `Option<Arc<AtomicBool>>`.
- **Returning `Ok` with a partial factor.** The issue explicitly does not
  want partial results, and a partial `SparseFactors` would be a
  correctness trap for `solve`.
- **Polling inside the dense kernels** (`apply_schur_panel_range` and
  friends). That is the hot inner loop; the panel loop above it is one
  level out and already coarse enough.
