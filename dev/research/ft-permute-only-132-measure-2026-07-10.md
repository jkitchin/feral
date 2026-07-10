# Issue #132 (Schork–Gondzio permute-only FT update) — measure-first — 2026-07-10

**Verdict: not justified as a performance lever. Do not implement now.** The
Forrest–Tomlin eta arithmetic that #132 can remove is 0.1–5% of update work;
the dominant cost is the *spike solve* over the (dense) factor, which #132 does
not touch. Recorded per the measure-first discipline (cf. #129).

## Context

The current `update_sparse` already implements the logical-`uperm` Forrest–Tomlin
update (research note `ft-row-elimination-design-2026-06-21.md`, "Route 2 ∪
Route 3-logical"): each update computes the spike `ρ = G⁻¹L⁻¹P·aₙₑw`, does a
symmetric cyclic shift of the bump, and eliminates the single pivotal row →
**one** row-eta. #132 is the Schork–Gondzio refinement: when the spiked bump
digraph is *acyclic*, re-triangularize by permutation alone — **zero** eta,
zero arithmetic — falling back to the row elimination only on the cyclic
residual. Its payoff is bounded above by the eta/elimination cost it removes.

## Method

`src/bin/probe_ft_eta.rs` replays real / structured update sequences through the
production `SparseLu` and records, per committed update, `last_eta_ops` (the
solve-replay op count of the eta — the thing #132 removes) and
`last_update_work` (the build-time scatter count, dominated by the
Gilbert–Peierls spike solve over the factor).

## Results

**Real trace — casctanks (m=2169, 3 segments, 144 updates):**
```
eta_ops: total=1722  max=52  mean=12.0   (2.8% of updates have a trivial eta)
eta_ops / (eta_ops + update_work) = 4.9%
histogram: 84 updates 1–9 ops, 56 updates 10–99, none ≥100
```
The single-row elimination already produces trivially small etas. #132's
ceiling here is ≤5% of update work, and less once restricted to the acyclic
subset.

**Synthetic set-covering bases (dense B⁻¹ — discopt's expensive case, where
discopt measures update() at 83% of the sc2000x800 root LP):**
```
m=800  per_col=6  500 upd:  eta mean=372   eta/(eta+work) = 0.1%   (work 1.6e8)
m=1200 per_col=8  500 upd:  eta mean=594   eta/(eta+work) = 0.1%   (work 3.6e8)
m=2000 per_col=6  400 upd:  eta mean=902   eta/(eta+work) = 0.1%   (work 6.1e8)
```
The eta grows with m (the dense-spike phenomenon the design note flagged), but
it is **0.1%** of update work. The other **99.9%** is the spike solve
(`compute_spike`: the Gilbert–Peierls reach + L-solve over the dense factor,
`work += nnz` per reached L-column). That is exactly discopt's observation that
"update cost tracks the factor **fill**, not the entering-column nnz."

## Interpretation

1. #132 removes the **eta/elimination**, which is the small part (0.1% on the
   dense covering bases that dominate discopt's time, ~5% on casctanks). It does
   **not** touch the spike solve, which is the fill-tracking bulk.
2. There is also a structural tension: the expensive-eta case is a *dense* bump,
   and a dense bump's digraph is cyclic, so #132's permute-only path could not
   fire there even in principle — it fires on *sparse* bumps, which are exactly
   the ones whose eta is already cheap.
3. Therefore #132 cannot deliver the "update() is 83% and tracks fill" win. The
   lever for that is **less factor fill** (#133 dynamic Markowitz) or
   more-frequent refactorization / a cheaper `compute_spike`, not permute-only
   re-triangularization.

## Decision

Do not implement #132 as a performance item. Its measured ceiling is 0.1–5% of
update work and it misses the fill-tracking bottleneck entirely. Re-open only if
a workload appears whose updates carry *large* etas on *sparse (acyclic)* bumps
— none of the real trace (casctanks) or the covering-basis model shows that.

## Caveats

- The covering bases are synthetic (random 0/1 columns + identity backbone), not
  from a solved LP; but the real casctanks trace agrees directionally (eta a
  single-digit % of work), and discopt's own "tracks fill" comment corroborates
  the spike-solve dominance.
- The eta also lengthens the FT chain replayed by every later `compute_spike` /
  solve; that cumulative cost is likewise small here (chains are short and the
  L-solve over the base factor dominates each `compute_spike`).
