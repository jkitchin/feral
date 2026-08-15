# Scaling router permutation-invariance (issue #134 item B)

**Date:** 2026-08-15
**Status:** gate (b) fix justified and scoped; gate (a) deferred with evidence
**Probe:** `crates/feral-diagnostics/src/bin/probe_scaling_levers.rs`
(`134b`, `price`, `marg`, `spec`, `table` sections — all read-only)

## The claim under test

Issue #134 item B says `pick_scaling_strategy` counts stored
lower-triangle columns only, so a *trailing* dense border dodges the
`>32` head gate and forfeits the documented 6×–243× MC64 wins. The
IPOPT duals-last convention that pounce and discopt emit produces
exactly that shape.

## The bug is real, and bigger than filed

`CscMatrix` stores only `row >= col`. Under the pure relabeling
`P(i) = n-1-i` — which changes nothing about the matrix as an operator
— the arrow head moves from column 0 to the last column and its mass
redistributes one entry per earlier column:

| family   | stored max deg | reversed | shipped route | reversed route |
|----------|----------------|----------|---------------|----------------|
| VESUVIO  | 1026           | 11       | Mc64Symmetric | InfNorm        |
| VESUVIOU | 1026           | 11       | Mc64Symmetric | InfNorm        |
| MUONSINE | 512            | 4        | Mc64Symmetric | InfNorm        |
| CRESC132 | 2657           | 8        | Mc64Symmetric | InfNorm        |

Over the **full** corpus — `data/matrices/kkt`, `kkt-mittelmann`, and
`kkt-expansion`, 1004 families — the shipped router is
permutation-invariant on only **841**. 163 families (16%) route on an
artifact of index order.

> An earlier sweep in this session covered only `data/matrices/kkt`
> (568 families) and reported 27. That undercounted by 6× and, worse,
> omitted `kkt-mittelmann`, which is where the clnlbeam "MC64 hurts"
> evidence that calibrated the shipped thresholds lives.

## Both gates break, but only one is fixable now

Gate (a) is `#{j : stored nnz(j) == 1 and diag(j) != 0} / n >= 0.30`.
Gate (b) is `max_j stored nnz(j) > 32`. Each can independently be
recomputed on **symmetric** degree (degree of `j` in the full matrix).
Four variants, 1004 families:

| variant                 | route changes | gain | lose | invariant |
|-------------------------|---------------|------|------|-----------|
| shipped (stored, stored)| 0             | 0    | 0    | 841/1004  |
| **(stored a, sym b)**   | **15**        | **15** | **0** | **890/1004** |
| (sym a, stored b)       | 102           | 13   | 89   | 923/1004  |
| (sym a, sym b)          | 143           | 54   | 89   | 1004/1004 |

Reversed-VESUVIO check — does the variant fix the duals-last shape?

| variant           | VESUVIO | MUONSINE | CRESC132 |
|-------------------|---------|----------|----------|
| shipped           | InfNorm | InfNorm  | InfNorm  |
| (stored a, sym b) | MC64    | MC64     | MC64     |
| (sym a, stored b) | InfNorm | InfNorm  | InfNorm  |
| (sym a, sym b)    | MC64    | MC64     | MC64     |

**Gate (b) alone fixes the reported bug.** Gate (a) alone does not, and
costs 89 MC64 losses doing it — strictly worse on both axes.

**Gate (b) alone is monotone.** Symmetric degree ≥ stored degree, so
the head gate can only get easier: 15 changes, all gains, zero losses.
No family loses MC64. The 2026-05-17 calibration panel survives by
construction on the InfNorm side — clnlbeam's symmetric max degree is
5 and ACOPP30's is 29, both still under 32, so neither can cross.

## Why gate (a) cannot be made invariant today

The obvious invariant reading of gate (a) is "fraction of columns with
symmetric degree ≤ 2 and a nonzero diagonal": in an arrow KKT a slack
column carries its diagonal plus one coupling to the head, regardless
of which end the head sits at.

That reading is correct for textbook arrow KKTs and wrong for the
corpus. Symmetric degree of the columns the shipped gate counts:

| family       | n      | min | med | p90 | max |
|--------------|--------|-----|-----|-----|-----|
| MUONSINE     | 1537   | 3   | 4   | 4   | 4   |
| VESUVIO      | 3083   | 5   | 8   | 8   | 11  |
| STEERING     | 3599   | 3   | 5   | 6   | 6   |
| marine_1600  | 76807  | 5   | 10  | 10  | 10  |
| pinene_3200  | 127995 | 5   | 9   | 17  | 17  |
| MSQRTA       | 2048   | 64  | 64  | 64  | 64  |
| LEUVEN3      | 4173   | 1   | 20  | 236 | 379 |

There is no single `slack_max` in that spread. The shipped gate does
not measure "slack mass"; it measures *"j has no coupling to any later
index"*, which is a property of the ordering, not of the matrix.

Sweeping `slack_max ∈ {1,2,3}` confirms it: 124 / 143 / 138 route
changes, and at `slack_max = 2` the 89 losers include **marine_1600,
rocket_12800, ROCKET, STEERING, robot_a/b/c, robot_1600, pinene_3200,
gasoil_3200, qcqp1500-\***, TWIRISM1 — the large IPM families that
dominate feral's real workload, and where the MC64 cache is measurably
paying (marine: 12/18 hits this session, 25–30 ms hit vs 87–127 ms
miss).

Taking MC64 away from those needs IPM-outcome data on 89 families. A
routing-parity argument is not sufficient evidence. Deferred, not done.

## Pricing the 15 gate-(b) gainers

Every iterate, min-of-3 factor, `rhs = A·1`, forward error `‖x−1‖∞`:

- **Inertia identical** under both routes on all 15. No inertia risk.
- **Time** (MC64/InfNorm): median 0.98, range 0.95–1.19. Free or
  faster on 14 of 15.
- **Accuracy** neutral or better on 14. CHAIN 5.02e-6 → 1.42e-8;
  SOSQP1 8.49e-6 → 1.95e-6; C-RELOAD 2.04e-10 → 1.07e-10.

Movers: MSS1, BIGBANK, BLOWEYA, BLOWEYB, BLOWEYC, C-RELOAD, CHAIN,
CLEUVEN2, CMPC1, LEUVEN1, LEUVEN2, SOSQP1, SOSQP2, arki0003, arki0009.

**Unfavorable case.** MSS1 (n=163) is the one family where the change
costs both time (0.872 → 1.037 ms, +19%) and accuracy (1.43e-1 →
6.15e-1). Both numbers are forward error on a system neither route
solves — at 1e-1 the metric cannot separate the routes, so this is
ill-conditioning, not a regression the change introduces. Absolute
cost 0.17 ms. Recorded because it is the only mover MC64 does not
help.

## Decision

Ship `(stored a, sym b)`. Defer gate (a) to an outcome-driven study.

Cost: gate (b) needs an `n`-length degree accumulator, so the router
loses its "no allocations" property. One O(n + nnz) pass becomes two.
Negligible against the factorization it precedes.

## Prior art consulted

- `dev/research/mc64-dense-column-2026-06-06.md`
- `dev/research/lever-c-adaptive-scaling.md`
- `dev/tried-and-rejected.md` (2026-04-28 MUMPS missing-diagonal skip:
  heuristics ported from another solver's literature must be validated
  against the input regime feral actually sees)
- `pick_scaling_strategy` doc comment, threshold panel from
  `dev/journal/2026-05-17-01.org` §14:30
