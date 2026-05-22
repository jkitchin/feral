# FERAL Context (auto-generated)

Generated: 2026-05-22T00:40:31Z

## Latest Session
File: dev/sessions/2026-05-21-04.md
```
# Session 2026-05-21-04

## Goal

Investigation session (no `src/` changes). Two questions from the human:

1. **#47** — do explicit-zero KKT entries still cost a ~2× slowdown on
   POUNCE CHO `parmest` after Fix 1 (`42434a5` fine-grained delayed
   pivoting) + Fix 2 (`80c05f5` cancellation-free 2×2 inertia)?
2. **#44** — is the NARX_CFy per-factor cost issue (ipopt-feral times
   out at 600 s) still relevant after the same two fixes?

## Accomplished

### #47 — reproduces, root cause pinned

A standalone iter-0 factor shows **no** penalty (stripped 439 ms vs
kept 456 ms), consistent with the issue's own caveat that iter-0 is not
the slow case. So the cost is in the warm-refactor path.

End-to-end POUNCE CHO `parmest` on current feral HEAD (measured by
*temporarily* repointing `pounce-feral` at the local checkout and
env-gating its zero-strip — **both POUNCE edits reverted afterwards,
binary rebuilt against `b3e4d3e`; POUNCE is untouched**):

| variant             | wall   | IPM iters | feral factor calls |
|---------------------|--------|-----------|--------------------|
| explicit zeros stripped | 10.6 s | 41    | 50                 |
| explicit zeros kept     | 22.6 s | 35    | 44                 |

#47 **still reproduces** — ~2.1× wall — with the #46 cascade fix in
place. It is not the #46 cascade.

**Root cause (pinned):** explicit zeros defeat the MC64 value-bounded
scaling cache (Track B2). New probe `probe_explicit_zeros` factors the
iter-0 KKT 4× on one warm `Solver`:

```
stripped:            cold 434ms -> warm 15/14/15ms    symbolic_calls=1  mc64_cache_hits=1,2,3
explicit zeros kept: cold 468ms -> warm 359/359/360ms  symbolic_calls=1  mc64_cache_hits=0,0,0
```

- Cold factor fine either way (~450 ms) — not a cascade, not fill.
- Symbolic analysis **is** reused either way (`symbolic_calls` stays 1)
  — the pattern fingerprint / symbolic cache is not the problem.
- The MC64 cache **never hits** with explicit zeros
  (`mc64_cache_hits` stays 0) — the Hungarian match reruns every
  factor, ~345 ms of the ~360 ms warm refactor. Stripped, the cache
  hits and the warm refactor collapses to ~15 ms (24× gap).

```

## Git Status
```
129f268 docs(plan): record issue-47 value-aware routing plan
e49694b fix(scaling): make pick_scaling_strategy value-aware (#47)
ed147dd test(issue-47): add scaling-strategy routing diagnostic to probe_explicit_zeros
49444d4 docs(journal): record MUMPS/SSIDS explicit-zero handling for #47
15c3a74 docs(journal): correct #47 root cause — MC64 degeneracy, not value-bound
```

## Test Status
```
