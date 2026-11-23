# Issue #246 step 2 — `n > 100k && avg_deg < 5`: AMD → AMF (DRAFT)

Date: 2026-11-23. Journal: `dev/journal/2026-11-23-01.org`.
**Status: draft, pending the powerflow22 A/B.** powerflow22 is the
calibration case for the current rule (#50). It is GAMS-licensed and not
available in the container this was written in.

## Question

`choose_adaptive` sends very-large-and-sparse matrices (`n > 100_000`, full
average degree < 5) to AMD (#50, 2026-05-23). On Primal2_1000 that gives
3.69e8 nnz(L) and an 8.30 GB predicted peak, against 2.08e8 and 3.53 GB for
AMF (full production path, LdltCompress). Should the branch go to AMF?

#50 chose AMD over ScotchND/MetisND because nested dissection's *symbolic*
cost at n ≈ 2.8M was 2× AMD's, with no fill benefit on powerflow22. AMF was
not measured there, and at that time it had the large-n quantization
inversion (fixed in 0f6eb9c), so a pre-fix measurement would not have been
representative anyway.

## Evidence (public LP KKTs in the regime)

The KKTs are the IPM augmented system `[[I, Aᵀ], [A, −1e-8 I]]` from
Mittelmann LPs (`dev/scripts/mps_to_kkt.py`). Prefix probe, preprocess
`None`, AMF with the monotone quantization:

| matrix    | n     | avg deg | AMD nnz(L) / Σc²  | AMF nnz(L) / Σc²  | MetisND nnz(L) | order time AMD / AMF |
|-----------|-------|---------|-------------------|-------------------|----------------|----------------------|
| Primal2   | 3.86M | 3.85    | 3.187e8 / 2.41e12 | 1.802e8 / 4.09e11 | 1.998e8        | 1.7 / 2.0 s          |
| dlr1      | 10.9M | 4.38    | 1.903e8 / 7.09e11 | 1.176e8 / 2.38e11 | —              | 9.3 / 13.1 s         |
| lo10      | 453k  | 4.59    | 3.071e8 / 5.01e12 | 3.051e8 / 4.95e12 | 5.100e8        | 0.8 / 1.3 s          |
| long15    | 786k  | 4.83    | 3.697e8 / 6.61e12 | 3.692e8 / 6.59e12 | 4.594e8        | 1.9 / 2.7 s          |
| netlarge2 | 1.2M  | 4.87    | 4.952e8 / 1.02e13 | 4.945e8 / 1.02e13 | 6.361e8        | 2.9 / 4.4 s          |

AMF's fill is never above AMD's. On the two LP KKTs (Primal2, dlr1) it is
1.6–1.8× lower in nnz(L) and 3–6× lower in the flop proxy. On the network
LPs it is within 0.6%. The ordering costs 1.3–1.6× AMD's, which is
seconds. MetisND is worse than both on every network LP, which confirms
#50's reason for keeping nested dissection out of this regime.

## Numeric (Primal2, sequential driver)

`Solver::new().with_ordering(..).with_parallel(false)`, one process per
ordering, RLIMIT_AS 14 GB, 4-core container:

| ordering | first factor (analysis + numeric) | refactor | inertia | VmHWM / predicted peak |
|----------|-----------------------------------|----------|---------|------------------------|
| AMD (current `Auto`) | 1010.2 s | 999.2 s | (+2 559 380, −1 299 380, 0) | 11.49 GB / 8.30 GB |
| AMF (this draft)     |   95.9 s |  83.8 s | (+2 559 380, −1 299 380, 0) |  6.37 GB / 3.53 GB |

AMF refactors **11.9× faster** with an identical inertia, which is exactly
(nvar, ncon). Here fill does predict speed. The flop-proxy ratio is 5.9×,
and the wall-clock ratio (11.9×) is larger still. Why it is larger was not
investigated. AMD's largest front is also 3× taller (max column count
15 815 vs 5 066).

lo10's numeric A/B was not completed. At this container's rate each
factor would take over an hour, and its AMD/AMF fill differs by 0.6%.

Separate finding: the real high-water mark exceeds `peak_bytes` by 1.4–1.8×
(sequential) and by ~2.6× under the parallel driver (Primal2 AMF: 9.36 GB
vs 3.53 GB). That gap also limits the step 3 budget. It is recorded in the
journal and is not addressed here.

## Risks

- **Fill is not speed (#67/#73, #209).** Where AMD and AMF tie on fill
  (the network LPs), the change is roughly neutral in expectation, but
  that is not measured here.
- **powerflow22 / PDE2.** These are the corpus matrices the #50 branch
  was built on (`dev/research/issue-50-*`). Both are needed before
  merging.
- **`Auto` collapses to AMF.** After this change, every `Auto` route for
  `n > 0` returns AMF (n ≤ 10k → AMF, arrow → AMF, would-be-MetisND →
  AMF, this branch → AMF). The arrow detector and `pick_default_method`'s
  size split stop affecting the result. They are left in place in this
  draft, and simplifying them is a separate decision.

## Required before merge

1. powerflow22 A/B (AMD vs AMF): symbolic time, nnz(L), factor time,
   inertia.
2. PDE2_0000 (corpus, n = 451k, avg 4.59): the same.
3. Corpus bench (`cargo run --bin bench --release` with `data/matrices/`).
   Only matrices with n > 100k and avg_deg < 5 can change.
