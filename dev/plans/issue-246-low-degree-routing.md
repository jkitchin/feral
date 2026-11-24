# Plan — issue #246 step 2: route `n > 100k && avg_deg < 5` to AMF (DRAFT)

Research: `dev/research/issue-246-low-degree-routing.md`.

## Change (drafted, committed separately so it can be dropped)

- `choose_adaptive`: return `OrderingMethod::Amf` instead of `Amd` from the
  very-large-and-sparse branch. Update the doc comments (`choose_adaptive`,
  `OrderingMethod::Auto`, `pick_default_method`) and the inline comments
  that cite the branch.
- Unit test `choose_adaptive_rules`: the n = 200k, avg_deg 3 pattern
  resolves to `Amf`.

## Gate before merge (needs the maintainer's machine)

1. powerflow22: AMD vs AMF, comparing symbolic time, nnz(L), first-factor
   and refactor time, and inertia. Accept if AMF is no worse than 1.1× on
   factor time and the inertia is identical.
2. PDE2_0000: the same.
3. Full corpus bench. Only n > 100k, avg_deg < 5 matrices can move.

If (1) or (2) fails, drop the commit. Steps 1 and 3 of #246 stand on their
own.

## Not in this draft

Collapsing `choose_adaptive` now that every `Auto` route for n > 0 returns
AMF. That changes no output but deletes the arrow detector's influence and
deserves its own decision.
