# Research spike: asymptotic sparse FT bump update (NOT yet scoped for impl)

Date opened: 2026-06-18
Status: **spike / open question.** No implementation until this resolves into a
design with a correctness story. This exists because discopt#229 wants the O(bump²)
worst case gone, and the obvious route is a known landmine.

Related: [[bump-elimination-speedup-2026-06-18]] (the safe constant-factor work that
proceeds independently), [[unsymmetric-lu]] §4.3.

---

## The problem this spike must solve

The in-place partial-pivoting bump elimination is O(bump²) on non-localized spikes
(`casctanks`: avg bump 750). Steps 1+2 cut the constant; they do not change the
asymptotic. A true O(bump · row-width) update needs the bump to stay structurally
narrow (one subdiagonal per column) — which is the textbook FT Hessenberg form.

## Why the obvious route is blocked

The textbook cyclic-permutation Hessenberg approach was **tried and reverted**
(journal 2026-06-08-01.org): in a sparse U the Hessenberg's diagonal pivots are the
old superdiagonal entries `U[k,k+1]`, frequently zero → zero pivot → division by
zero. Partial pivoting (the current scheme) fixes the zero pivot but reintroduces
fill/scan over the whole bump.

The journal names exactly two correct asymptotic routes, both flagged "substantial,
delicate":

1. **Symmetric-permutation FT.** Physically permute U on each update so the bump
   stays Hessenberg, with careful permutation-of-stored-state bookkeeping — "physically
   permuting U each update staleness the relation `L σᵀ` (not lower-triangular)". The
   open question: how to keep the stored unit-lower L consistent with a permuted U
   without re-touching L (which would defeat the bump-locality).

2. **Sparse Bartels–Golub–Reid (Reid 1982).** In-bump partial pivoting with
   row-permutation tracking that *bounds fill* — the form production LP solvers
   (HiGHS, etc.) actually engineer. The open question: the fill-bounding pivot
   strategy that keeps the eta file and the per-bump work near O(fill), not O(bump²).

## Questions to answer before any design

- Does Reid 1982 (`citep:reid1982sparsity`, already in `dev/references.bib`) give a
  pivot rule that keeps the bump's fill near-linear while staying numerically safe on
  sparse U? Consult the spral-expert / faer-expert agents for how SSIDS / HiGHS-style
  updates handle this.
- Can the symmetric-permutation route reuse feral's existing permutation machinery,
  or does it require a new stored-state invariant (and a new differential test for the
  `L σᵀ` relation)?
- Is the column-ordering lever (keep simplex-churned columns late so bumps stay
  narrow — discopt#229's other suggestion) a cheaper way to land in the "localized
  spike" regime the current code already handles well? This may dominate either
  algorithmic rewrite for the McCormick workload specifically.

## Exit criteria

This spike closes by either (a) producing a `dev/plans/` entry with a chosen route
and a correctness story for the zero-pivot/stored-state hazards, or (b) recording in
`dev/tried-and-rejected.md` that no asymptotic route beats steps 1+2 + an ordering
fix at acceptable risk, with the evidence.
