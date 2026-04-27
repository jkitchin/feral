# AMF (Approximate Minimum Fill) — clean-room research note

Status: research, no code yet.
Date: 2026-04-27.
Driver: ORBIT2 cluster regression and MUMPS choice of HAMF4 as the
default ordering for SYM=2, N≤10000 (`ana_set_ordering.F:52-78`).

## 1. Motivation

For symmetric-indefinite KKT systems in our target regime (N up to a
few tens of thousands), MUMPS does not pick AMD. It picks **HAMF4**
— Halo Approximate Minimum Fill, version 4. SCOTCH's `vmumda`
implements the same algorithm. AMD only takes over for very large
problems where HAMF4's per-iteration cost dominates.

This is not an obscure tuning choice — it is the first ordering the
MUMPS analysis dispatcher tries. We have empirical confirmation on
at least one bipartite-KKT family that AMD is dramatically worse than
AMF here:

| Matrix       | feral-amd nnz_L | feral-metis nnz_L | MUMPS HAMF4 nnz_L | gap to MUMPS |
|--------------|-----------------|-------------------|-------------------|--------------|
| ORBIT2_0000  | 5,147,360       | 1,544,349         | 109,782           | 47×          |

feral-amd already implements Davis 1996 §5 dense-row deferral
(`AmdOptions::dense_alpha = 10.0`); empirical probing
(`src/bin/diag_orbit2_quotient.rs`) confirms `n_dense_deferred=1` and
the dense column is correctly placed at the tail. The 47× gap is not
about dense rows — it is about **AMD vs AMF**: AMD greedily minimizes
*degree* whereas AMF minimizes *fill*. On bipartite-KKT graphs with
one or two heavy "hub" rows, those metrics diverge sharply.

We expect the same gap on COSHFUN, CATENA, and broadly any bordered
or arrowhead pattern in the corpus. The case for AMF is not "fix
ORBIT2" — it is "match the reference solver's default for our regime".

## 2. Algorithmic foundation

**Primary reference**: Amestoy, P.R. (1999), *Méthodes directes
parallèles de résolution des systèmes creux de grande taille*,
Habilitation thesis, INPT. AMF4 is the variant in which all
previously formed cliques adjacent to a candidate variable are
deducted from its fill estimate (line 4980 comment in MUMPS).

**Foundation**: Amestoy, Davis, Duff (1996), "An approximate minimum
degree ordering algorithm," SIAM J. Matrix Analysis 17:886–905. AMF
shares all of AMD's quotient-graph machinery — element/supervariable
structure, Scan-1 / Scan-2 update pattern, hash-based supervariable
detection, mass elimination, aggressive absorption. AMF changes only
the **selection metric** and a handful of supporting data structures.

**Not a separate paper**: Rothberg & Eisenstat 1998 ("Node Selection
Strategies for Bottom-Up Sparse Matrix Ordering") is in the same
intellectual lineage but is *not* cited by MUMPS HAMF4. The clean-room
reference of record is Amestoy 1999.

**The "approximate" in AMF** has the same meaning as in AMD: the
element-list bound `|L_me \ L_e|` is computed from the quotient graph
rather than the actual filled graph, and external degrees are used in
place of re-symbolic-factorization. There is no new approximation
introduced by AMF.

## 3. Fill metric — exact definition

For a candidate variable `i` with `nv(i) = NVI`, approximate external
degree `DEG`, and (lazily-computed) accumulated clique area `WF(i)`:

```
WF(i) = WF4 + 2 * NVI * WF3
  WF4 = sum over elements e adjacent to i of  dext(e) * (2*deg(e) - dext(e) - 1)
  WF3 = sum over singleton variables j adjacent to i of nv(j)
  dext(e) = |L_e \ L_me|   (computed by Scan 1)

DEGME = |L_me| - nvpiv          (off-diagonal size of the new element)

if DEG + DEGME ≤ NLEFT:
    RMF = DEG * (DEG - 1 + 2*DEGME) - WF(i)
else:
    [saturated branch — see ana_orderings.F:4970-4983]

RMF /= (NVI + 1)
```

`RMF` is computed in `f64` because `DEG, DEGME ≤ N` so `RMF` can
reach `O(N^3)`. It is then quantized to integer `WF(i)` and used as
the bucket key.

**Interpretation**: `DEG * (DEG - 1 + 2*DEGME)` is approximately twice
the area of the clique that would form around `i` if `i` were
eliminated next. `WF(i)` is the area already covered by previously
formed cliques (avoiding double-count). Division by `NVI + 1`
normalizes per row of the supervariable. This is the exact AMF metric
of Amestoy 1999.

## 4. Bucket discretization

Because `RMF` can be `O(N^3)`, a degree-array of length `N+1` (as in
AMD) is wrong. HAMF4 uses:

```
NBBUCK = 2 * N                       (set in dana_aux.F:663)
PAS    = max(N / 8, 1)               (bucket stride in the coarse region)
HEAD   = vec[i32; NBBUCK + 2]

bucket(s):
    if s ≤ NORIG:  s
    else:          min((s - NORIG) / PAS + NORIG, NBBUCK)
```

Below `NORIG` (`= N`), one bucket per integer score (cheap pivots,
where ordering quality matters most). Above `NORIG`, coarse bins of
width `PAS`. Bucket `NBBUCK + 1` is reserved for halo (V1) variables
— inert in our use case.

When the chosen bucket is in the coarse region (`DEG > NORIG`),
HAMF4 walks the entire bucket linked list and picks the entry with
the smallest *exact* `WF` value (`ana_orderings.F:4392-4418`). Below
`NORIG` it just takes the head, like AMD.

## 5. Data-structure deltas vs AMD

Identical to AMD plus **one new array**: `WF: Vec<i32>` of length `N`,
with two roles:

- For variables `i`: stores the integer-quantized fill score (used for
  bucket placement and tie-breaking).
- For elements `e`: caches `dext(e) * (2*deg(e) - dext(e) - 1)`,
  computed lazily on first use within an iteration.

PE / IW / LEN / NV / ELEN / DEGREE / W / NEXT / LAST / HEAD all
unchanged in semantics. Only `HEAD` changes in length (`2N + 2`
instead of `N + 1`).

## 6. Implementation deltas — six inner-loop sites

For someone with a working clean-room AMD (we have one in
`crates/feral-amd/`), HAMF4 is roughly **400–600 LoC of Rust delta**.
The 1485 lines of MUMPS Fortran are inflated by halo machinery,
verbose comments, the COMPRESS branch (matched by our existing AMD),
and bucket-quantization arithmetic.

The six inner-loop sites that change:

1. **Init**: allocate `WF`, set `WF(i) = LEN(i)` initially. Allocate
   bucket head array of length `2N + 2` instead of `N + 1`.
2. **Scan 1** (`ana_orderings.F:4671-4693`): when an element `e` is
   first encountered this iteration (`W(e) < WFLG && W(e) != 0`), set
   `WF(e) = 0` (line 4688). Single store.
3. **Scan 2** (`ana_orderings.F:4703-4841`): replace AMD's single
   degree accumulator with three:
   - `DEG` (same as AMD).
   - `WF4 = sum WF(e)` over adjacent elements; `WF(e)` computed lazily
     as `dext * (2*deg(e) - dext - 1)` on first use (line 4726).
   - `WF3 = sum nv(j)` over adjacent singleton variables.
   - At end: `WF(i) = WF4 + 2*NVI*WF3` (line 4810). Special case: if
     the AMD-style degree update was loose (`DEGREE(i) < DEG`), zero
     `WF4, WF3` to keep the metric a valid bound (lines 4795-4807).
4. **Supervariable absorption** (`ana_orderings.F:4920`): on merge
   `j → i`, `WF(i) = max(WF(i), WF(j))`. Same hash key and walk as
   AMD.
5. **Final score and re-bucket** (`ana_orderings.F:4954-5017`):
   compute `RMF` (Section 3) in `f64`, quantize to integer, insert
   into bucket `bucket(WF(i), …)`.
6. **Pivot selection** (`ana_orderings.F:4376-4427`): when
   `MINDEG > NORIG`, linear-scan the chosen bucket and pick the entry
   with the smallest exact `WF` value.

Element construction, IW compression, hash detection, mass
elimination, aggressive absorption — all **byte-for-byte the same
algorithm as AMD**.

## 7. Architectural decision

**Recommendation**: extract a `feral-ordering-core` crate containing
the quotient-graph machinery (PE / IW / LEN / NV / ELEN compression,
Scan-1 / Scan-2 skeleton, hash detection). Parameterize it by a
`Metric` trait with two impls:

```rust
trait Metric {
    type Score: Copy + Ord;
    fn init_score(len: i32) -> Self::Score;
    fn scan2_accumulate(state: &mut Self, e: ElementInfo) -> ...;
    fn finalize_score(state: Self, deg: i32, degme: i32, nvi: i32, nleft: i32) -> Self::Score;
    fn bucket(score: Self::Score, n: i32) -> usize;
    fn merge_supervariable(parent: &mut Self::Score, child: Self::Score);
}
```

`feral-amd` and `feral-amf` become thin specialization layers. Pros:

- Zero-cost abstraction via monomorphization (clippy/inlining stay
  intact, no runtime branches).
- Single quotient-graph code path — bug fixes propagate to both
  orderings.
- Each ordering crate stays small and reviewable.

Alternative considered: feature flag on the existing `feral-amd`
inner loop. Rejected because the deltas touch six distinct sites and
the conditional code would clutter the loop, hurt clippy/inlining,
and make A/B testing harder. The trait-based factoring is clean even
if AMF turns out to be the only AMD variant we ever ship.

## 8. Heuristic parameters

- **No equivalent of `dense_alpha`** in HAMF4 itself. Dense-row
  protection is the job of a separate routine (MUMPS_QAMD,
  `ana_orderings.F:5226+`) layered on top of AMD's metric, not AMF's.
  If we want dense-row deferral for AMF, the simplest route is to
  detect dense rows up-front (`degree > alpha * sqrt(N)` per Davis
  1996 §5) and prepend them to the elimination order before entering
  the AMF main loop on the remaining graph. We can copy the existing
  `feral-amd::dense_alpha` mechanism wholesale into the
  ordering-core crate as a pre-processing step.
- **`PAS` (bucket stride)** = `max(N/8, 1)`. Fixed; not a tunable.
- **`NBBUCK`** = `2*N`. Affects only memory.
- **No `CNTL` knob** controls AMF behavior. HAMF4 is deterministic
  given a fixed input pattern (modulo the hash modulus).

## 9. Test oracle strategy

HAMF4 is **not byte-deterministic across implementations**. Reasons:

1. Tie-breaking inside coarse buckets depends on linked-list
   insertion order, which depends on Scan-2 emission order, which
   depends on `IW` layout and compression history.
2. Supervariable hash collisions depend on `HMOD = max(1, NBBUCK-1)`.
3. `RMF` quantization is lossy.

Test strategy in order of strictness:

1. **Functional invariants** (always exact): output is a permutation;
   every `pe(root) == 0`; the elimination tree implied by `pe` is a
   forest spanning all variables; supervariable counts sum to `N`.
2. **Inertia after symbolic factorization** with the produced
   ordering — `nnz(L)` is the meaningful quality metric. Compare our
   AMF's `nnz_L` to MUMPS HAMF4's `nnz_L` on the existing
   `data/matrices/kkt*` corpora. Pass criterion: feral-amf nnz_L
   ≤ 1.10 × MUMPS HAMF4 nnz_L on every matrix where MUMPS HAMF4 is
   itself reasonable.
3. **Exact permutation match** on tiny matrices (N ≤ 20) where
   structure is regular enough to avoid ties. Hand-derived from the
   AMF metric on the AMD test fixtures.
4. **Pivot sequence match on a deterministic order** (debugging only,
   not CI): fix `IW` layout and tie-break rules to match MUMPS's
   convention; reproduce HAMF4 byte-for-byte on a small corpus.

For ORBIT2 specifically: feral-amf must produce nnz_L within 10% of
MUMPS HAMF4's 109,782, i.e. ≤ ~120k. Currently feral-amd produces
5,147,360 (47× worse than MUMPS).

## 10. Scope estimate

Rough budget:

- **Module factoring** (extract `feral-ordering-core`, port AMD as
  the first `Metric` impl): 1 session. Bit-parity test
  (feral-amd-pre vs feral-amd-on-core) is the gate.
- **AMF metric implementation** (Sites 1-6 above, with `WF` array
  and quantized buckets): 1-2 sessions including the matching
  bit-parity tests against AMD on all-zero-WF inputs.
- **Test oracle plumbing** (run MUMPS HAMF4 on the corpus, dump
  `SYM_PERM`, store as sidecar `.hamf4.json` for CI comparison): 1
  session.
- **Bench-corpus validation and tuning** (re-run W-1+W-2 corpus with
  AMF as default, address regressions): 1 session.

Total: 3-5 sessions, dominated by the module-factoring and oracle
plumbing rather than the AMF math itself.

## 11. Constraints and caveats

- Pure Rust, MIT, clean-room from Amestoy 1999 paper plus the
  reference reading of MUMPS source (algorithmic understanding only;
  no code copy). Same ground rules as the existing feral-amd.
- The MUMPS source path is `../ripopt/ref/mumps/src/ana_orderings.F`
  for reference reading. SCOTCH `vmumda` is the same algorithm under
  CECILL but we will not consult it given the paper-derived
  constraint.
- Halo machinery (V1 boundary-preservation) is out of scope; we
  always pass `LEN(i) ≥ 0` so `NBFLAG = 0` and the halo branch is
  dead.

## 12. Decision

Proceed with the trait-based `feral-ordering-core` factoring, then
AMF as a second `Metric` impl. First gate: feral-amd-on-core
bit-parity with the existing crate on the kkt corpus.

Plan note: `dev/plans/amf-clean-room.md` (next).

## 13. References

- Amestoy, P.R. (1999), *Méthodes directes parallèles de résolution
  des systèmes creux de grande taille*, Habilitation thesis, INPT.
- Amestoy, Davis, Duff (1996), "An approximate minimum degree
  ordering algorithm," SIAM J. Matrix Analysis 17:886-905.
- Davis (1996), "A column pre-ordering strategy for the
  unsymmetric-pattern multifrontal method" (dense-row deferral §5).
- MUMPS 5.8.2 source, `src/ana_orderings.F:3722-5207` (HAMF4 routine);
  `src/dana_aux.F:663-710` (caller, `NBBUCK` setup);
  `src/ana_set_ordering.F:52-78` (HAMF4 selection logic).
