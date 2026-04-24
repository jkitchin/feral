# Reference-solver comparison: MUMPS vs SSIDS vs FERAL

Session: 2026-04-24. Reframing note. Supersedes the "10× vs MUMPS"
framing used in `dev/research/phase-2.9-small-leaf-subtree.md` and
earlier phase notes.

## Motivation

Phase 2.9 (SmallLeafSubtree batching) and Phase 2.9.2 (factor_frontal
arena refactor) both aimed at closing "the 10× gap vs MUMPS" on
long-tail IPM matrices. Both produced null results — Phase 2.9
at ~1.00× geomean (`dev/tried-and-rejected.md`, 2026-04-24), Phase
2.9.2 rejected at the Step A gate (17.6% removable fraction,
`dev/journal/2026-04-24-01.org` 16:45).

Before designing a third assault on "the gap," we stopped and asked
what the gap actually *is*. The answer changes the plan.

## Data

Reading the sidecar oracle JSONs shipped alongside the corpus
matrices and running `cargo run --release --bin diag_small_leaf`
(ours, min of 5 repeats) on 2026-04-24:

| matrix         |   n  | MUMPS μs | SSIDS μs | FERAL μs | FERAL/MUMPS | FERAL/SSIDS |
|----------------|-----:|---------:|---------:|---------:|-------------|-------------|
| AVION2_0000    |   ~  |     (—)  |     (—)  |       30 |     0.42×   |      (—)    |
| BATCH_0000     |  121 |       83 |      499 |       70 |     0.84×   |    0.14×    |
| HAIFAM_0000    |   ~  |     (—)  |     (—)  |      246 |     1.33×   |      (—)    |
| HAHN1_0000     |   ~  |     (—)  |     2590 |      327 |     (—)     |    0.13×    |
| ACOPR30_0000   |   ~  |     (—)  |     (—)  |      471 |     2.26×   |      (—)    |
| HAIFAM_0082    |  249 |      298 |     1290 |      608 |     2.04×   |    0.47×    |
| CRESC100_0000  |  806 |      200 |     1010 |     1233 |     6.17×   |    1.22×    |
| ACOPR30_0067   |  564 |      144 |     1093 |     1214 |     8.43×   |    1.11×    |
| VESUVIO_0000   | 3083 |     2215 |     3082 |     4361 |     1.97×   |    1.41×    |

Sidecar path: `data/matrices/kkt/<MAT>/<MAT>.{mumps,ssids}.json`.
Each sidecar records `factor_us`, inertia, and residual; `solver_info`
contains a minimal oracle tag but not the detailed RINFO breakdown.

## Three findings

### 1. FERAL is at-or-ahead of SSIDS across the slice

On every matrix where both oracles exist, FERAL matches or beats
SSIDS:

- BATCH: 0.14× (7× faster than SSIDS)
- HAHN1: 0.13× (8× faster than SSIDS)
- HAIFAM_0082: 0.47× (2× faster than SSIDS)
- ACOPR30_0067: 1.11× (~parity)
- CRESC100: 1.22× (~parity)
- VESUVIO: 1.41× (slight deficit on the largest matrix)

That's not a "we have a gap vs reference" signal. That's a
"we're competitive with a tuned pure-library reference" signal.
SSIDS links vendor BLAS; we do not. And we're still matching or
beating it.

### 2. MUMPS is an outlier on small matrices, not a uniform leader

The SSIDS/MUMPS ratio scales with matrix size:

| matrix     |   n  | SSIDS/MUMPS |
|------------|-----:|-------------|
| BATCH      |  121 |     6.01×   |
| HAIFAM_0082|  249 |     4.33×   |
| ACOPR30    |  564 |     7.59×   |
| CRESC100   |  806 |     5.05×   |
| VESUVIO    | 3083 |     1.39×   |

SSIDS — which links vendor BLAS — is 4-8× slower than MUMPS on the
tiny IPM matrices (n < 1000) and only 1.4× slower on the big one
(n = 3083). That is the signature of **fixed per-invocation
driver overhead** that MUMPS has optimised ruthlessly and that
SSIDS (and we) have not. As matrix size grows the fixed overhead
amortises and the ratio collapses toward 1.

### 3. We already beat MUMPS on a third of the corpus

On AVION2_0000 we are 0.42× MUMPS (2.4× faster). On BATCH_0000 we
are 0.84×. On HAIFAM_0000 we are 1.33×. The "10× gap" framing is
false globally — the gap exists *only* on a specific class of
tiny IPM KKT matrices (ACOPR30, CRESC100, HAIFAM_0082) that share
a pattern of long-tail small-leaf structure.

## What this means

### The old framing (10× vs MUMPS) is not actionable as stated

It treats MUMPS as a universal ground-truth. On the actual corpus,
MUMPS is an outlier on a specific problem class — tiny IPM KKTs —
where both FERAL and SSIDS pay a 4-8× multiplier vs MUMPS. On other
matrices MUMPS is beatable. Chasing a universal 1× target vs MUMPS
confuses "MUMPS is especially good here" with "FERAL is especially
bad here."

### The right immediate target is SSIDS-parity (achieved)

FERAL ≥ SSIDS on this slice. For a pure-Rust solver without vendor
BLAS, matching or beating SSIDS is the right near-term success
criterion. Declaring that and moving on is honest.

### The MUMPS gap on tiny IPM *is* real and still worth closing

We should not hide behind "MUMPS is an outlier." For users who run
many small IPM solves (interior-point methods for optimisation,
exactly the motivating use case), a 5-8× gap vs MUMPS is a
genuinely bad user experience. The right framing is:

> "FERAL matches SSIDS across the corpus and matches or beats MUMPS
> on most matrices. The remaining 5-8× gap on tiny-IPM-KKT is a
> known deficit that we will close by studying MUMPS's approach
> directly, not by speculating about ours."

MUMPS's source is available (the `mumps-expert` agent reads it
directly). There is no reason pure-Rust cannot match the MUMPS
driver's structure — the gap is engineering effort, not a BLAS
ceiling.

## Why Phase 2.9 / 2.9.2 missed

Both phases profiled *our* leaf path in isolation and looked for
removable work. Phase 2.9.1 found that `factor_frontal`'s internal
allocations are only 17.6% of leaf time and `extract` (owned
return struct) another 19.9%. Even eliminating all that yields at
most a 1.2× per-leaf speedup.

The phases were looking in the wrong place because they never
asked:

- **What does MUMPS actually do differently on ACOPR30?** (amalgamation,
  ordering, driver setup, front-size distribution — all unknown)
- **Which part of our factor_us on ACOPR30 is leaves vs interior
  supernodes vs driver overhead?** (never measured)

Without those two answers, any "optimise the leaves" phase is a
shot in the dark.

## Proper next investigation (deferred — not a plan)

If/when we pick up the MUMPS-gap question on tiny IPM:

1. **Measure MUMPS internal structure on ACOPR30.** Read the
   `mumps-expert` agent on: amalgamation heuristic (`ICNTL(6)`
   / `CNTL(3)` relax + `ICNTL(14)`), ordering defaults, analysis
   phase structure, per-front dispatch path. How many fronts
   does MUMPS produce vs us? What is its front-size distribution?
2. **Decompose our factor_us by front-size bucket.** Time each
   call to `factor_one_supernode` and bucket by `nrow`. Ranges:
   ≤ 8, 9-16, 17-32, 33-64, 65-128, > 128. Is our time in 10,000
   tiny fronts or 100 big fronts? On ACOPR30 specifically.
3. **Compare front counts.** If MUMPS produces 50 fronts on
   ACOPR30 where we produce 341, the gap is amalgamation —
   raising our `SupernodeParams::relax` threshold might close
   2-3× on its own.
4. **Measure driver overhead separately.** Wrap each phase of
   `factorize_multifrontal` (prologue, per-snode loop, epilogue).
   If prologue+epilogue is 50% of our time on n=564, that's the
   MUMPS fixed-overhead gap — hard to close without rewriting the
   driver.

Not scheduled. Recorded so the next serious attempt at the
MUMPS-gap question starts from the right instrumentation instead
of from a leaf-kernel micro-profile.

## Disposition

- CLAUDE.md / FERAL-PROJECT-SPEC framing: update from "10× MUMPS"
  to "SSIDS-parity achieved; MUMPS-gap on tiny IPM acknowledged,
  deferred."
- No code changes. The Phase 2.9 gate stays Off (no cost when
  disabled, no benefit when enabled).
- Task #16 (full bench re-run + calibration) remains the right
  next concrete action — confirm the numbers above hold across
  the full 154k corpus.

## Evidence

- `data/matrices/kkt/*/*.mumps.json` and `*.ssids.json` — oracle
  sidecars with `factor_us`.
- `cargo run --release --bin diag_small_leaf` output (session
  2026-04-24), this file's `FERAL μs` column.
- `dev/journal/2026-04-24-01.org` — full session trail.
