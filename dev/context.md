# FERAL Context (auto-generated)

Generated: 2026-08-21T15:45:10Z

## Latest Session
File: dev/sessions/2026-08-21-01.md
```
# Session 2026-08-21-01

## Unfavorable result, reported first (per CLAUDE.md)

**The solve tail regressed 33–56%, this session's change caused it, and a
controlled A/B proves that rather than leaving it as a hypothesis.**

`src/bin/bench.rs:1985` times the sparse solve with `RefineOptions::default()`,
so the new stopping criterion sits directly in the measured path. I ran the
bench with the shipped default, then reverted `Default for RefineOptions` to
`EpsSqrtN`, rebuilt, and re-ran on the same quiet machine:

| ratio | control (`EpsSqrtN`) | shipped default | delta |
|---|---:|---:|---:|
| factor/MUMPS p90 | 1.54 | 1.56 | +1.3% (noise) |
| solve/MUMPS geomean | 0.08 | 0.08 | none |
| solve/MUMPS p50 | 0.08 | 0.08 | none |
| solve/MUMPS p90 | 0.15 | **0.20** | **+33%** |
| solve/MUMPS p99 | 0.71 | **1.08** | **+52%** |
| solve/SSIDS geomean | 0.94 | **1.06** | **+13%** |
| solve/SSIDS p90 | 2.50 | **3.60** | **+44%** |
| solve/SSIDS p99 | 8.33 | **13.00** | **+56%** |

**The shape is what the design predicts.** Geomean and p50 are identical to
three decimals — the well-scaled majority never needed the extra conjunct
and does not pay for it. The tail pays, because that is where the matrices
live whose componentwise error was above `√ε` and which now actually refine.
On the tracked parity corpus that is 13 of 63 (21%), the right order to move
a p90.

Factor is unchanged within noise, as it must be — the change is confined to
the refinement loop's stopping test. That also validates the control: machine
state would have moved factor too, and did not. **Unlike 2026-08-20-02,
nothing here is attributed to machine state.**

**Against the standing bar** ("rigorous, thoroughly correct, and result in
performance gains, or we will not include it in a release"): this is a
measured performance *cost*, not a gain. It buys MA57/MUMPS componentwise
parity on systems where feral returned ω up to 9.5e-5 and reported
`Converged`. Whether that trade ships is the human's call. The alternative is
to keep `EpsSqrtN` as the default and make the conjunction opt-in — which
costs pounce the fix it asked for.

## Goal

Human instruction: *"we need to fix the deficiency gap with ma57/mumps if
there is one because this is causing an issue on a class of problems in
pounce. surprisingly, it works well for the vast majority of problems."*

## Accomplished
```

## Git Status
```
f5bc004 docs(dense): drop unverifiable pounce provenance from the growth-flag docs
396bfa3 fix(solve): default refinement now certifies componentwise accuracy
001a7db diag: four probes that locate the MA57/MUMPS deficiency gap
963884c docs: session checkpoint 2026-08-20-03 (#190 measured; premise refuted)
65f488c docs(refine): correct #190's premise against the corpus measurement
```

## Test Status
```
