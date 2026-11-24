# Issue #246 — AMF score quantization inverts priority at large n

Date: 2026-11-23. Journal: `dev/journal/2026-11-23-01.org`.

## Symptom

`OrderingMethod::Auto` → AMF on `fhnw-binschedule1` (pounce HSDE LP KKT,
n ≈ 1.9M) exhausts an 11–13 GB address-space cap inside the symbolic phase,
before any prediction is produced (issue #246, pounce#1249).

## Reproduction (this container)

The Mittelmann MPS file was converted to the IPM augmented system
`[[I, Aᵀ], [A, −1e-8 I]]` (pattern within 0.2% of pounce's KKT:
n = 1 914 525 vs 1 911 607, stored nnz 10 525 851 vs 10 517 895).
A probe ran the symbolic *prefix* only (ordering → postorder → etree →
`column_counts_gnp`), preprocess `None`:

| ordering | nnz(L)   | max colcount | Σc² (flop proxy) | order time |
|----------|----------|--------------|------------------|------------|
| AMD      | 4.369e8  |  24 445      | 5.56e12          | 86 s       |
| MetisND  | 4.031e8  |  21 754      | 3.65e12          | 81 s       |
| AMF      | 1.388e10 | 155 589      | 1.49e15          | 39 s       |

Prefix peak RSS was 1.3–1.9 GB. The memory blow-up is `symbolic_finish`
building row structure for a 1.4e10-entry factor. The ordering is the
defect, not the symbolic phase's memory use.

Degree profile of fhnw: median 5, p99 54, 8 692 columns of degree > 100,
216 > 1 000, four at 56 842 and two at 454 823. Only those six exceed the
dense-row threshold `10·√n = 13 836` and are deferred.

## Root cause

HAMF4 computes the approximate-fill score `RMF` in `f64` and stores it as
an integer bucket key. The quantization (feral
`crates/feral-ordering-core/src/quotient_graph/algo.rs`, re-insertion loop
of `run_elimination_amf`) is:

```text
if RMF < IDUMMY            → WF = round(RMF)
elif RMF / N < IDUMMY      → WF = round(RMF / N)
else                       → WF = IDUMMY            (IDUMMY = i32::MAX − 1)
```

The middle branch is **not monotone**. At n = 1.9M a score of 3e9 is
stored as ≈ 1 579, far below a score of 2e9 (stored as 2e9). Near-dense
variables therefore land in cheap buckets and are eliminated early. Each
such pivot creates a clique over its whole (tens of thousands wide)
neighbourhood.

The branch fires only when `RMF ≥ 2^31`. Since `RMF ≈ deg·(deg − 1 +
2·degme)/(nv + 1)`, that needs degrees of roughly 27k or more. Primal2_1000
(low degree) is unaffected, and so is every matrix the AMF corpus oracle
covers.

**This is verbatim MUMPS behaviour.** MUMPS 5.8.2 `ana_orderings.F:4985-4994`
(HAMF4) and `:8074-8079` (`MUMPS_CST_AMF`) contain the identical code.
`WF` is a 32-bit `INTEGER` (`:3758`), and there is no 64-bit variant and no
guard. (Checked by the mumps-expert agent against the 5.8.2 release
tarball.) Feral's port is faithful. The defect is inherited.

## Experiment

Store `round(RMF)` as `i64` (feral's `ws.wf` is already `Vec<i64>`, for
O1) with no `/N` fall-through, so the key is monotone in `RMF`. The bucket
mapping `amf_bucket_of` already clamps any key above the coarse range into
the last bucket `NBBUCK`, where `select_pivot_amf` linear-scans for the
smallest exact key. Ordering among saturated scores is therefore still
exact.

| matrix  | AMF nnz(L) before | after   | max colcount | Σc²     | order time |
|---------|-------------------|---------|--------------|---------|------------|
| fhnw    | 1.388e10          | 1.835e8 | 10 623       | 5.11e11 | 39 → 128 s |
| Primal2 | 1.802e8           | 1.802e8 | 5 066        | 4.09e11 | 2.0 → 2.3 s |

After the fix, AMF on fhnw has 2.4× less fill and 11× fewer flops than
AMD, and 2.2× / 7× less than MetisND.

**Cost.** Ordering time on fhnw rises from 39 s to 128 s (AMD: 86 s). Two
things contribute. The inverted ordering was cheap because it collapsed
the graph early. Correct scores also put many large-score variables into
the last bucket, which is linearly scanned. This is accepted: the
alternative does not finish.

## Consequences

- AMF orderings change only where `RMF ≥ 2^31` at some pivot. Below that
  the code paths are bit-identical. Small and medium corpus matrices
  (n ≲ 27k, or low degree) are unaffected by construction.
- This is a deliberate deviation from the MUMPS reference. It is recorded
  in `dev/decisions.md`. The AMF corpus oracle compares feral nnz_L to
  MUMPS HAMF4 sidecars with a 10% margin. On a matrix where MUMPS itself
  inverts, feral would now be *better* than the oracle, which the gate
  permits (it bounds feral from above).
- It does not address the Primal2 half of #246 (`avg_deg < 5 → AMD`
  routing), which needs the powerflow22 regression check before changing.
