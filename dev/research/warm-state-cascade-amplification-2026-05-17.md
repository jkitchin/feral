# Wide-supernode cascade on small-δ_w KKT matrices

**Date:** 2026-05-17 (rewritten same day after disproof)
**Status:** Open — investigation refocused
**Origin:** Mittelmann KKT investigation, `dev/journal/2026-05-17-01.org`
**Related issues:** #37 (wide-supernode cascade), #38 (MC64 cache staleness)

## Revision history

The earlier draft of this note (07:30) claimed the cascade was
"warm-state amplification" — that a `Solver` that had factored
matrix N-1 would cascade on matrix N while a fresh `Solver` handed
matrix N alone would factor it cleanly. That claim came from agent
a84f721906859018f's footnote (3) reporting "FRESH=1 kills the cascade
entirely … at iter 5 (1.56 s)".

**Step 1 of the investigation disproved this.** Direct measurement
shows:

- `FRESH=1 probe_kkt_replay pinene_3200` (rebuild `Solver` every
  iteration): iters 0–4 factor in ~1.0 s each, **iter 5 hangs**
  (killed at 180 s wall) — same as the warm `Solver`.
- `FRESH=1 probe_kkt_replay marine_1600`: iters 0–8 factor in
  0.3–1.5 s each fresh, **iter 9 takes 64 s, iter 10 hangs**.
- `probe_warm_cascade pinene_3200 4 5` (a Solver-state-bisection
  probe added today, `src/bin/probe_warm_cascade.rs`):
  COLD mode (fresh `Solver`, factor only the curr matrix) hung at
  240 s on pinene_3200 iter 5.

The agent's "FRESH 1.56 s" appears to have been a misremembering of
the table's `CB+FRESH=1.555 s` row. There is no row for FRESH-alone
in the agent's measurements.

## Corrected claim

The cascade is **structural to the iter-N matrix**, not a property
of accumulated solver state. The trigger appears to be the
numerical content of the matrix, specifically:

- **pinene_3200 iter 5**: the root supernode is wide (~14k cols for
  n=128k, ~11%), and at iter 5 the numerical values produce a
  pivot-search pattern that the default `may_delay = true` policy
  expands without bound.
- **marine_1600 iter 9**: `delta_w` first drops below `delta_c`
  (5e-9 vs 1.5e-9 per sidecar). Below that perturbation level,
  Bunch-Kaufman rejects many pivots; without `cascade_break` they
  accumulate.
- **dtoc2 iter 1**: `delta_w ≈ 6.99e19` saturates the diagonal;
  MC64's row/col scale factors collide with the regularised
  diagonal and produce a pathological pivot-search loop.

In all three cases, `cascade_break(0.5, eps=1e-10)` escapes the
loop in O(0.1 s/iter); a `Solver` rebuild does not.

## Why FRESH doesn't help

The earlier story said "FRESH throws away accumulated delayed-pivot
bookkeeping that bloats subsequent supernodes". That mechanism does
not exist: per `src/numeric/solver.rs:122-129` the pooled
`FactorWorkspace` is "cleared to a well-defined initial state on
every `factorize_multifrontal_with_workspace` entry, so stale data
cannot leak between factor attempts." Step-throughs of `factor()`
confirm:

- `last_factors` is overwritten unconditionally on `Ok` / cleared
  on `Err` (lines 483–510).
- `last_symbolic` is invalidated when the pattern fingerprint
  changes (lines 376–381), and the IPM case has a stable pattern,
  so the cached symbolic is *intentionally* reused — but that
  symbolic is a pure function of the matrix pattern (no numerical
  dependence; see `symbolic_factorize_with_method`).
- `cached_mc64` inside the symbolic is invalidated after every
  numeric factor (lines 466–468, the issue #38 fix).
- `parallel_pool` only owns worker threads; no per-factor state.

Hence FRESH and warm produce the *same* symbolic on a stable
pattern, and the per-call workspace is identically clean. The
cascade therefore cannot be in solver state — it must be in the
matrix's numerical content interacting with the supernodal kernel's
pivot-search loop.

## Why CB=1 rescues all three

`cascade_break(0.5, eps=1e-10)` flips a non-root supernode to
`may_delay = false` once its front carries ≥50% delayed columns
(`src/numeric/solver.rs:255–267`). A flipped node force-accepts the
pivot via `ZeroPivotAction::ForceAccept` rather than continuing to
search and delay further. This caps the pivot-search loop at the
heavy-delay supernode and prevents the explosion.

The cost is a small perturbation on the accepted pivots, which is
absorbed by IPOPT's outer regularisation and iterative refinement
on solve — the same trade-off MA57's `cntl[4]` static-pivoting
fallback makes.

## Open questions (refocused investigation)

1. **What pivot-search trip count is the loop hitting?** Need to
   instrument the supernodal panel (around
   `src/numeric/factorize.rs` pivot loop) with a per-supernode work
   counter; emit when a single front does >K candidate evaluations.
   Today the only visible signal is wall time.

2. **Can the cascade-break threshold be auto-armed at symbolic
   time?** `max_supernode_nrow / n > ALPHA` is a cheap pre-factor
   check (binaries like `diag_cascade_ratio_distribution` already
   compute the distribution). If the dimension ratio that triggers
   the cascade is above some α, arm CB for that solver/factor pair
   automatically — no per-problem env var.

3. **Can a numeric-time trigger arm CB lazily?** e.g. on first
   supernode that hits >M delayed pivots, flip CB mid-factor. That
   would avoid the iters-0-4 overhead pinene pays for CB-default-on.

4. **MC64 + saturated diagonal (dtoc2)** is a separate kernel-level
   bug: when `max|diag| ≫ max|offdiag|`, MC64 should fall back to
   identity scaling on those rows (per agent's suggestion). That
   would fix dtoc2 without needing CB.

## Investigation plan

1. **DONE** — Step 1 (reproduce in isolation). Disproved the
   warm-state hypothesis; cascade is structural to the matrix.
   Evidence: `/tmp/pinene_fresh_verify.log`,
   `/tmp/marine_fresh_verify.log`, `src/bin/probe_warm_cascade.rs`.

2. **NEXT** — Instrument the supernodal pivot-search loop with a
   per-supernode candidate-evaluation counter, gated behind a
   `NumericParams` debug flag so production paths pay nothing.
   Replay the three cascade matrices and confirm what the loop is
   doing (this is the missing direct observation that distinguishes
   "search loop is unbounded" from "search loop is bounded but
   doing useless work").

3. **DISPROVED (2026-05-17 11:05)** — Symbolic-time auto-arm on
   `max_supernode_nrow / n ≥ α` cannot work. `probe_supernode_widths`
   measured the ratio across cascade and benign problems and they
   overlap completely (pinene 0.003–0.005, marine 0.013–0.020,
   dtoc2 0.001, robot 0.001–0.007, clnlbeam 0.000). The cascade
   emerges from delayed pivots accumulating at runtime, and delays
   are a function of NUMERIC values, not pattern. Symbolic
   structure is blind to the cascade by construction.

   Replaced by: **warm auto-arm** (task #67). Persist
   `n_delayed_root` from each factor() call. On the next factor()
   with matching pattern fingerprint, if `prev_n_delayed_root ≥
   β·n`, arm cascade_break for this call. First factor pays
   cascade cost; iter 2+ auto-rescued without user intervention.
   See `dev/journal/2026-05-17-01.org` §11:05 for the disproof
   data and §10:15 for the SQD finding that motivated the
   refocus.

4. **AFTER** — File a separate issue for the MC64 + saturated
   diagonal handling (dtoc2). Not gated on the above.

## Cross-references

- `dev/journal/2026-05-17-01.org` — full agent reports, the original
  (incorrect) warm-state claim, and the disproof.
- `src/bin/probe_warm_cascade.rs` — bisection probe; COLD-mode
  result already invalidates the warm-state thesis.
- `src/numeric/solver.rs:269–328` — `with_cascade_break` and
  `with_cascade_break_eps` (the rescue).
- `src/numeric/solver.rs:371–514` — `factor()` showing all caches
  invalidated correctly between calls.
- `src/bin/diag_cascade_ratio_distribution.rs` — corpus collector
  for the auto-arm threshold.
- `dev/tried-and-rejected.md` — prior CB-default-on attempt with
  >20% overhead (predates the iters-0-4-only measurement that shows
  the overhead is now only 11–15% on the cascade-prone matrices).
- `dev/decisions.md` — 2026-05-15 reclassification of #17.
