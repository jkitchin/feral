# Issue #246 / PR #247 review — interrupting the symbolic phase

Date: 2026-11-24. Journal: `dev/journal/2026-11-23-01.org`.
Source: the pounce-side review on PR #247 (comment 7116780548).

## Problem

`symbolic_factorize_with_method` is one uninterruptible call. A host that
enforces a wall-clock limit (pounce's `max_wall_time`, the cpo3 300 s
protocol) can check its deadline only before and after it. Measured cases:

- L2CTA3D: MetisND did not finish inside pounce's ordering-sweep timeout.
- fhnw-binschedule1: `AutoRace` took 476 s and `Auto` 595 s.

## Where the time goes (code inspection, 0.24.0 + this branch)

| stage | code | long-running loop |
|---|---|---|
| MC64 for `LdltCompress` | `src/scaling/hungarian.rs` `hungarian_match_instrumented` | `for jord in 0..n` (one shortest-path augmentation per column) |
| AMD / AMF ordering | `feral-ordering-core` `run_elimination{,_amf}` | `while ws.nel < ws.n` (one pivot per iteration) |
| MetisND | `feral-metis` `node_nd::recurse` → `multilevel_node_bisection` → `coarsen`, `initial_partition` trials, `fm_refine` passes | per recursion node, per level, per trial, per pass; FM heap loops are O(n) per pass |
| ScotchND | `feral-scotch` `node_nd::recurse`, `band_fm` / `halo_fm` passes, `vertex_separator` passes | same shape as METIS |
| KaHIP | `feral-kahip` `node_nd::recurse`, flow refinement | per recursion node, per BFS |
| column counts, supernodes, static rows | `src/symbolic/mod.rs` | linear; each stage is seconds at n = 2M |
| races | `symbolic_factorize_race`, `symbolic_prefix_preprocess_auto` | one full prefix per arm |

The symbolic phase and every ordering crate are single-threaded. No rayon
is used in `crates/feral-{ordering-core,amd,amf,metis,scotch,kahip}` or
`src/symbolic`. `src/scaling/infnorm.rs` uses rayon, but only from the
numeric phase.

## Design

**`feral_ordering_core::interrupt`**, the lowest common crate:

- `Interrupt { deadline: Option<Instant>, flag: Option<Arc<AtomicBool>> }`
  with `none()`, `at(Instant)`, `after(Duration)` and `with_flag(Arc<AtomicBool>)`.
  `is_triggered()` returns true once the deadline has passed or the flag is
  set.
- **A scoped thread-local** holds the active interrupt: `scope(&Interrupt,
  || f())` installs it for the closure and restores the previous one
  afterwards, including on unwind (a guard). Deep loops call `poll()`
  (bool) or `check()` (`Result<(), OrderingError>`) with no signature
  changes through METIS/SCOTCH/KaHIP's helpers. A thread-local is sound here
  because the whole symbolic call runs on the caller's thread (above). If
  an ordering crate ever goes parallel, it must forward the interrupt
  explicitly.
- **Sticky.** Once `poll()` has seen a trigger, every later poll on that
  scope returns true, even if a flag is cleared, so an early `break` is
  always followed by an error.
- **Cost.** `poll()`/`check()` always read the clock when a deadline is
  set (`Instant::now` costs about 20–30 ns). The per-pivot loops (AMD/AMF)
  and per-column loops (Hungarian) poll every 64 iterations, and coarser
  safe points poll every time. With no interrupt installed, a poll is one
  thread-local read.

**Safe points.** A loop that returns `Result` calls `check()?`. A loop
that cannot (`coarsen`, FM refinement, `multilevel_node_bisection`,
Hungarian) only `break`s at a pass, level or trial boundary on `poll()`.
That leaves a valid but unfinished intermediate (fewer levels, an
unrefined partition, an incomplete matching), and the next `check()?` up the
stack, at the recursion entry or right after the call, turns it into an
error before the intermediate is used. For MC64 the check sits right after
`compute_mc64_cache` returns *either* `Ok` or `Err`, so a partial matching
cannot surface as a "structurally singular" error.

**Error.** `OrderingError::Interrupted` (ordering crates) and
`FeralError::Interrupted { stage: &'static str }` (feral). This is one
variant for both the deadline and the flag: the host holds the `Interrupt`
and can tell which one fired (`Interrupt::deadline_passed()`). The
reviewer suggested `DeadlineExceeded`. `Interrupted` is used because a
cancel flag is not a deadline.

**feral plumbing.**
- `SupernodeParams::interrupt: Option<Interrupt>`. `symbolic_factorize_with_method`
  (and `_with_schur`) install it in a scope for the whole call.
- Pipeline `check()`s: entry, after `symmetric_pattern`, after
  MC64/compress, after ordering, after column counts, between race arms
  (`AutoRace` candidates and the preprocess-`Auto` second arm), and
  between the finish stages (`find_supernodes`, small-leaf, peak-contrib,
  static rows).
- `Solver::with_interrupt(Option<Interrupt>)` and `Solver::set_interrupt`
  (a deadline is absolute, so a host reusing one `Solver` sets a fresh one
  per call). Through `Solver::factor` the error arrives as
  `FactorStatus::FatalError(FeralError::Interrupted { .. })`. The analysis
  is not cached, so the next `factor` re-analyses.
- The numeric phase is out of scope here. The review asked about the
  symbolic phase, and a numeric interrupt needs its own safe-point analysis
  of the parallel driver.

## Race widening at n ≥ 1M (same review, item 2)

`RACE_CANDIDATES = {Amd, MetisND}` is kept for n < 1 000 000, which covers
every pounce fixture and #208's laptime. For n ≥ 1 000 000 the race also
runs `ScotchND` and `Amf`. Reasons:
- AMF is now monotone (0f6eb9c), and on fhnw it wins (1.84e8 vs AMD 2.51e8
  in the issue's AutoRace).
- The reviewer measured SCOTCH as the best on L1_sixm250obs (158 GB
  predicted vs METIS 187 GB and AMD 544 GB).
- The threshold matches pounce's `CONVEX_AUTO_RACE_MIN_DIM` stopgap.
- With the deadline hook, a slow arm (METIS on L2CTA3D) can now be cut off
  by the host.

Ranking stays `factor_nnz_estimate`. This is a behaviour change for
`AutoRace` callers at n ≥ 1M only.
