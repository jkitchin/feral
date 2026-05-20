# FERAL Context (auto-generated)

Generated: 2026-05-20T01:38:05Z

## Latest Session
File: dev/sessions/2026-05-19-01.md
```
# Session 2026-05-19-01

## Goal

Close the gap that prevents an IPM backed by FERAL (`pounce`) from bumping
its Hessian perturbation `δ_w` on KKT systems that are ill-conditioned but
land on the correct inertia. Ipopt+MA57 solves three Mittelmann-class
problems in 100–291 iters; pounce+FERAL stalls because the perturbation
handler never fires. The agreed fix: have FERAL report a near-singularity
signal — `min|λ(D)|`, the smallest accepted pivot magnitude — that the
perturbation handler can threshold.

## Accomplished

- **Research note + plan** — `dev/research/near-singularity-signal.md`,
  `dev/plans/near-singularity-signal.md`. Root cause traced: FERAL's default
  `ZeroPivotAction::ForceAccept` force-accepts a near-singular pivot and
  returns `FactorStatus::Success`; the only near-singularity-adjacent fact
  reaching the IPM is `needs_refinement`, which is internal, a coarse
  boolean, and already true on healthy cascade-break factorizations. MA57
  reports the analogous case via `CNTL(2)` → `INFO(1)==4` → Ipopt
  `SYMSOLVER_SINGULAR` → `PerturbForSingularity`.

- **Rust API** — `SparseFactors::{min,max}_pivot_magnitude` over a shared
  `pivot_magnitude_extent()` pass, mirroring the existing `min_diagonal()`.
  `Solver::{min,max}_pivot_magnitude` delegate. 2×2 smaller magnitude
  computed `|det|/larger` to stay cancellation-free on near-singular blocks.
  Kept deliberately distinct from `min_diagonal()` (signed-min vs.
  magnitude-min).

- **C ABI** — `feral_min_pivot` / `feral_max_pivot` (`-1.0` sentinel on
  no-factor / null handle). Declared in `feral-ipopt-shim/include/feral_capi.h`.

- **Evidence** — 5 new tests, all with hand-computed oracles external to the
  implementation:
  - `diag(5,-2,3,-7)`: `min|λ|=2`, `max|λ|=7`, `min_diagonal=-7`
  - `[[0,1],[1,0]]`: 2×2 block, `min=max=1` (`|smaller eig|`, not
    `d_diag[0]=0`)
  - `diag(1,1e-14,-3)`: inertia `(2,1,0)` still correct, `min|λ|≈1e-14`,
    ratio `min/max≈3e-15` — thresholdable where inertia alone is silent
  - `None` / `-1.0` sentinel before any factor
  - C ABI `capi_min_max_pivot`: `[[1,2],[2,1]]` under identity scaling →
    `min|λ(D)|=1`, `max|λ(D)|=3`
  Full `cargo test` exit 0; `cargo clippy --all-targets -- -D warnings`
  clean.

- Three atomic commits (research+plan; Rust API; C ABI). CHANGELOG
  Unreleased and `dev/decisions.md` updated.

## Benchmark Results
```

## Git Status
```
40b5612 fix(stress): allowlist 3 #28 cross-arch BK-pivot divergences
8298d7b chore(session): 2026-05-19-01 -- near-singularity signal min|λ(D)|
f6640eb feat(capi): feral_min_pivot / feral_max_pivot near-singularity ABI
cb03009 feat(numeric): min/max pivot magnitude near-singularity signal
5b81db0 docs(research): plan near-singularity signal (min|λ(D)|)
```

## Test Status
```
