# Research note: KaHIP driver integration

**Session:** 2026-04-18-08
**Question deferred from:** sessions 2026-04-18-04, 2026-04-18-06, 2026-04-18-07
**Scope:** decide what (if anything) to do with `OrderingMethod::KahipND`
in the symbolic-factorize dispatcher. K1-K6 are implemented; KaHIP is
reachable via `symbolic_factorize_with_method(.., KahipND)` and via
`OrderingMethod::Auto`, but the default `pick_default_method`
heuristic never selects it.

## Background

K1 (Ost-Schulz-Strash 2021 data reduction, Rule 1 only with
conservative termination — see `dev/research/ordering-kahip-k1.md`)
is wired *inside* `crates/feral-kahip/src/node_nd.rs:45` as a
preprocessing pass before the K2-K6 multilevel pipeline. That is,
the line "K1 not wired" in the session 07 checkpoint is no longer
literally accurate — K1 runs whenever KaHIP runs. The remaining
integration question is:

  Should `pick_default_method` route any IPM-corpus family to
  `KahipND` (with K1 preprocessing) by default, the way the n>=5000
  && nnz/n<6 rule routes CRESC-class matrices to `MetisND`?

A separate, larger question — should K1 be lifted *out* of KaHIP
and offered as a generic "data reduction" preprocessor that any
ordering can compose with — is treated at the end as a possible
follow-up.

## Evidence

### 41-matrix shape bake-off (this session)

Re-ran `cargo run --release --bin bench_orderings` against
`tests/data/parity` ∪ `tests/data/large`. Key aggregates (full
table in journal 2026-04-18-08, second entry):

```
geomean fill_kahip / fill_amd  = 1.023
geomean fill_metis / fill_amd  = 1.024
geomean fill_scotch / fill_amd = 1.038
geomean fill_auto  / fill_amd  = 0.988

total symbolic time (us, 41 matrices):
  AMD     14_396_571      baseline
  SCOTCH  15_304_726      1.06×  AMD
  METIS   68_157_794      4.74×  AMD
  KaHIP   81_121_482      5.64×  AMD
  Auto    14_747_890      1.02×  AMD  (Auto bypasses KaHIP/METIS for tiny matrices)

per-call examples:
  cresc132 n=5314   AMD 9990us  METIS 14812us  KaHIP 15984us  (KaHIP slowest)
  c-big    n=345k   AMD 14.0s   METIS 67.3s    KaHIP 80.0s    (KaHIP slowest)
  hahn1    n=715    AMD  222us  METIS   542us  KaHIP   693us  (KaHIP 3.1× AMD)
  chwirut1 n=645    AMD  209us  METIS  1030us  KaHIP  1378us  (KaHIP 6.6× AMD)
```

Strict-fill wins of KaHIP over AMD (only 4/41):

```
cresc132    27_096 vs 44_612  (-39%)  also won by METIS, identical fill
chwirut1     2_061 vs  2_065  ( -0.2%) also won by Auto via SCOTCH
cresc100     3_025 vs  3_043  ( -0.6%) also won by Auto
hahn1        3_432 vs  3_457  ( -0.7%) also won by Auto
```

In each case KaHIP **ties** the best other ordering rather than
beats it. K1 preprocessing does not produce a uniquely winning
fill on any matrix in the 41-corpus.

### 154 588-matrix IPM bench (already in checkpoint 07)

The narrow `n >= 5000 && nnz/n < 6 → MetisND` rule already cuts the
factor/MUMPS max ratio from 521× to 85× and improves geomean to
0.42. The remaining tail outlier is the VESUVIO family, which the
session 08 diagnostic showed is a *dense-kernel* problem
(`dev/journal/2026-04-18-08.org` entry 2), not an ordering problem
— METIS gives ≤8% improvement over AMD and KaHIP would only match
METIS.

`OrderingMethod::Auto` was rejected previously precisely because it
routes thousands of n<500 IPM iteration dumps to KaHIP and the
per-call setup cost (3-7× AMD's per-call cost on n<1000) wipes
out any fill win. K1 preprocessing is included in that overhead
budget.

## Decision matrix

```
option                                    fill win?   per-call cost   bench risk
A. status quo: KahipND only via explicit  none        n/a             zero
   `with_method` or `Auto`
B. add narrow rule like the MetisND one,  none on     +3-7× AMD on    high — adds
   e.g. nnz/n very low or arrow detected  IPM corpus  affected matrices  per-call work
C. lift K1 as standalone preprocessor     unknown,    K1 alone is     medium — needs
   in front of AMD                        possibly    O(|V|+|E|) per   real bench
                                          on arrow    pass + 3-5 passes  numbers
                                          KKTs        ≈ 1.5× AMD setup
```

## Recommendation

**Pick option A (status quo).** Concretely:

1. Do not modify `pick_default_method`.
2. Document the bench finding so future sessions don't re-derive
   it: add a short paragraph to `OrderingMethod::KahipND`'s
   docstring noting that the dispatcher does not select KaHIP
   because K1+K2-K6 ties METIS on fill at 4-6× the per-call cost.
3. Keep KaHIP reachable via `symbolic_factorize_with_method` and
   via `OrderingMethod::Auto` (Auto remains opt-in).
4. Defer option C (K1-as-generic-preprocessor) until either
   - a workload appears where leaf-heavy preprocessing matters
     (e.g. very sparse arrow KKTs that AMD currently handles
     poorly), or
   - the dense-kernel work in priority #2 closes enough of the
     factor gap that a different bottleneck surfaces ordering as
     a constraint again.

The reasoning is: K1 ships value to KaHIP users today, KaHIP is
not on the bench's hot path, and there is no measured family on
the IPM corpus where KaHIP-with-K1 is uniquely better than what
the existing AMD-or-MetisND rule already produces. Adding code
without a measured win violates "don't add features beyond what
the task requires" (CLAUDE.md tone-and-style guidance).

## What would change this recommendation

- A new bench corpus (e.g. compressed-sensing, semidefinite-
  programming KKTs, HEMSR-coarsening MILP relaxations) where
  KaHIP+K1 strictly beats METIS on fill *and* the per-call cost
  amortizes against larger n.
- A measured workload where the AMD-only path produces an
  ordering pathology that K1's degree-1 cascade fixes (e.g.
  long degree-1 chains AMD's tie-breaker handles
  pessimistically). The IPM corpus does not exhibit this.
- A rewrite of K1 that runs in <1 µs on n<200 (the per-call
  overhead on the IPM corpus dominates because most matrices are
  tiny and K1's fixed-point setup costs ~5-50 µs even when no
  reduction is possible). Not currently scoped.

## Cross-references

- `dev/research/ordering-kahip-k1.md` — K1 algorithm specification
  and oracle construction.
- `dev/research/ordering-kahip-k5-k6.md` — multilevel pipeline.
- `dev/research/ordering-bakeoff-2026-04-18.md` — earlier 30-matrix
  bake-off (without KaHIP).
- `dev/journal/2026-04-18-08.org` (entries 2-3) — bench raw data
  and VESUVIO finding establishing that ordering is not the
  dominant factor-time bottleneck remaining.
- `src/symbolic/mod.rs:178` — `pick_default_method`.
- `crates/feral-kahip/src/node_nd.rs:45` — current K1 wiring inside
  KaHIP.
