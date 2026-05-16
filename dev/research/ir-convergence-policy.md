# Iterative-refinement convergence policy (issue #30)

**Status.** Research complete; no implementation change recommended.
The current loop's exit criteria are correct as measured.

**Bottom line.** On the full 28-matrix stress manifest, the
existing `solve_sparse_refined` loop already short-circuits when
the unrefined relative residual is below `ε·√n` (17/28 matrices).
The remaining 11 matrices either need IR (gain 0.8–10.2 decades,
7 matrices) or stagnate above the threshold (4 matrices, bounded
to ≤3 wasted solves by the existing 2-strike rule). There is no
cheap a-priori predictor — the Hager–Higham κ̂(A) overlaps
massively between buckets and would mis-route `bratu3d`
(κ̂=1.16e3, IR gain 10.24 decades) into the skip path.

## Questions from issue #30

> 1. When does IR strictly improve residual? (cond(A) > what?)

There is no κ̂ threshold that separates "IR helps" from "IR is a
no-op". Per-matrix evidence:

| matrix     | κ̂(A)    | rel_res₀ | rel_res_best | gain (dec) | bucket |
| ---------- | ------- | -------- | ------------ | ---------- | ------ |
| bratu3d    | 1.16e3  | 1.72e-5  | 9.93e-16     | 10.24      | B      |
| tuma1      | 1.82e3  | 2.01e-14 | 2.01e-14     | 0.00       | A      |
| mario001   | 3.26e4  | 6.02e-16 | 6.02e-16     | 0.00       | A      |
| helm3d01   | 3.03e6  | 1.57e-12 | 1.79e-16     | 3.94       | B      |
| c-55       | 6.92e8  | 3.68e-16 | 3.68e-16     | 0.00       | A      |
| ill_cond_e10 | 6.59e10 | 6.99e-16 | 6.99e-16   | 0.00       | A      |
| cvxqp3     | 2.17e16 | 8.04e-12 | 1.58e-16     | 4.71       | B      |
| stokes64   | 2.32e16 | 7.01e-14 | 3.37e-14     | 0.32       | C      |
| ncvxqp3    | 1.33e19 | 4.54e-9  | 1.57e-16     | 7.46       | B      |
| turon_m    | 9.47e19 | 1.61e-14 | 1.61e-14     | 0.00       | A      |
| ncvxqp1    | 2.29e29 | 1.77e-15 | 1.77e-15     | 0.00       | A      |

Bucket A (no-op): κ̂ spans `[9.94e1, 2.29e29]`, median 6.6e10.
Bucket B (helps): κ̂ spans `[1.16e3, 8.00e22]`, median 3.7e13.

The κ̂ distributions overlap completely. Routing decisions cannot
be made from κ̂ alone.

The actual driver is not κ̂ — it is the unrefined relative
residual `||A·x₀−b||/||b||`. That can only be measured by
performing one `symv` + `norm2` after the unrefined solve. The
existing loop already does this and exits at line 834 when
`||r||/||b|| < ε·√n`. No cheaper predictor exists.

> 2. When does IR converge to the wrong fixed point?

Two matrices in the corpus stagnate on a non-zero residual:

- `stokes64` (κ̂=2.32e16): rel_res₀=7.01e-14, rel_res_best=3.37e-14
  after 7 useful steps. IR reduces residual by 0.32 decades but
  never reaches `ε·√n ≈ 2.5e-14`. The corpus oracle (MUMPS 5.8.2)
  also reports rel_res ≈ 3.4e-14 on this matrix — the residual
  floor is intrinsic to the matrix's near-singular Stokes
  geometry, not a feral bug.
- `stokes128` (κ̂=2.60e16): rel_res₀=2.18e-13, rel_res_best=9.57e-14
  after 4 useful steps. Same pattern.
- `rankdef_50_5` and `rankdef_200_20`: synthetic rank-deficient
  matrices where the unrefined residual already sits at the
  attainable floor and IR cannot improve it. Loop exits at 2
  stagnant steps (max_stagnant_steps=2 rule).

In none of these cases does IR diverge — it asymptotes
gracefully. The 2-strike plateau rule plus the divergence guard
(`r_norm > best_r_norm * 100`) already prevent wasted work.

> 3. Is the right exit criterion "stop when correction norm < eps"
>    or "stop when residual stops decreasing"?

Residual-based termination is correct. A prior version of this
routine used `||δx||/||x|| < ε·√n` and was changed (see the
code comment at lines 647–652 of `src/numeric/solve.rs`) because
on `ForceAccept`-perturbed matrices the iterate stopped updating
(tiny δx) without the residual having actually dropped into the
target basin. The current rule — exit when
`||r||/||b|| < ε·√n` (machine precision reached) OR when 2
consecutive steps fail to improve (plateau) — is honest about
"are we done yet" and matches the buckets seen in the data:

- Bucket A exits at the residual check (step 1, no IR applied)
- Bucket B exits at the residual check after 1–2 IR steps
- Bucket C exits via the 2-strike plateau or max_steps=10

## Methodology

1. `src/bin/probe_ir_trajectory.rs` — calls
   `solve_sparse_refined_with_diagnostics` on every manifest entry
   and emits a sidecar containing `kappa_1_est`, per-step
   `||r||_2`, `||r||/||b||`, Skeel forward-error bound, and a
   `improved` flag.
2. `external_benchmarks/stress/run.py` builds the matrices via
   `fetch.py` (SuiteSparse) and `synth.py` (synthetic).
3. `external_benchmarks/stress/analyze_ir.py` joins the per-step
   trajectory against the manifest, classifies each matrix into
   one of three buckets, and prints the κ̂-versus-bucket summary
   reproduced above.

All 28 manifest entries factored cleanly. Probe wall time
≈ 60 s on the full corpus.

## Buckets

**A. Starts below `ε·√n` (IR is a no-op): 17 matrices.**
The unrefined relative residual is already at floor noise.
The loop exits at the residual check on step 1 with zero
extra solves. Includes well-conditioned PDE (bratu3d's
counterexample notwithstanding — bratu3d is *not* in this
bucket), saddle-point and many ill-conditioned matrices.

**B. IR strictly helps (moves residual below `ε·√n`): 7 matrices.**
| matrix              | rel_res₀ | rel_res_best | gain (dec) |
| ------------------- | -------- | ------------ | ---------- |
| bratu3d             | 1.72e-5  | 9.93e-16     | 10.24      |
| helm3d01            | 1.57e-12 | 1.79e-16     | 3.94       |
| near_singular_eps9  | 2.59e-15 | 4.41e-16     | 0.77       |
| near_singular_eps12 | 3.10e-15 | 4.06e-16     | 0.88       |
| cvxqp3              | 8.04e-12 | 1.58e-16     | 4.71       |
| ncvxqp3             | 4.54e-9  | 1.57e-16     | 7.46       |
| bloweybl            | 1.89e-13 | 4.85e-16     | 2.59       |

All seven converge in exactly 1 useful IR step. The loop exits
at step 2 (1 IR step + 1 stagnant step, since the 2-strike rule
allows one).

**C. IR runs but residual stagnates above `ε·√n`: 4 matrices.**
| matrix         | rel_res₀ | rel_res_best | useful_steps |
| -------------- | -------- | ------------ | ------------ |
| stokes64       | 7.01e-14 | 3.37e-14     | 5            |
| stokes128      | 2.18e-13 | 9.57e-14     | 4            |
| rankdef_50_5   | 2.96e-15 | 2.96e-15     | 0            |
| rankdef_200_20 | 1.31e-14 | 1.31e-14     | 0            |

All four exit via the 2-strike plateau guard. The two stokes
matrices do extract a fraction of a decade of improvement before
asymptoting. The two synthetic rankdef matrices spend exactly
3 solves (1 attempted IR step + 2 stagnant strikes) and cannot
improve — this is the bounded-cost worst case.

## Solve-call accounting

| bucket            | matrices | extra IR solves | comment                         |
| ----------------- | -------- | --------------- | ------------------------------- |
| A (no-op)         | 17       | 0               | exits at the residual check     |
| B (strictly helps)| 7        | 7               | 1 useful + 0 wasted per matrix  |
| C (stagnates)     | 4        | 17              | bounded by max_stagnant_steps=2 |

The total "wasted" IR work across the corpus is 4 matrices ×
≤3 extra solves each = at most 12 wasted solve-calls beyond
what a perfectly-prescient policy would do. The current loop is
already near-optimal.

## Decision

**Do not change `solve_sparse_refined`.** The exit criteria —
residual-based termination at `ε·√n`, 2-strike plateau guard,
100× divergence guard, max 10 steps — already produce the right
behavior on every bucket of the stress corpus. There is no κ̂
threshold that would route matrices correctly, and a
correction-norm exit was tried in a prior iteration and rejected
(see code comment at lines 647–652).

**For callers who want bare back-substitution**: they already
have `Solver::solve`, `solve_sparse`, and `solve_sparse_many`
which skip IR entirely. The "skip IR" knob exists at the
Solver-method level, not inside `solve_sparse_refined`.

The decision is recorded in `dev/decisions.md` as
"2026-05-16 — Issue #30 IR convergence policy: keep
residual-based exit, do not add a κ̂ skip heuristic".

## Reproducibility

```
cd external_benchmarks/stress
python3 fetch.py                # SuiteSparse downloads
python3 synth.py                # synthetic matrices
python3 run.py                  # produces out/feral/ (IR on)
cargo build --release --bin probe_ir_trajectory
# build per-row manifest, then:
./target/release/probe_ir_trajectory /tmp/ir_probe.manifest
python3 analyze_ir.py           # bucket A / B / C table + κ̂ stats
```

Raw sidecars: `external_benchmarks/stress/out/ir_probe/*.out`.

## Open questions

- **Block-IPM workloads.** The corpus measures one solve per
  matrix. IPM solvers do tens of solves per factor (changing RHS
  but fixed `A`). If the per-solve symv+norm2 cost matters at
  that scale, the right escape hatch is the existing `solve`
  method, not a knob inside the refined path. No measurement
  here; flagged for the IPM bring-up note when it lands.
- **bratu3d / cvxqp3 / ncvxqp3 dynamics.** Three matrices with
  rel_res₀ ∈ [4.5e-9, 1.7e-5] converge to ε in one step. That is
  consistent with theory (one IR step on a non-pathological
  matrix multiplies the residual by κ·ε ≈ 1.16e3·1e-16 ≪ 1).
  No follow-up needed.
- **Stokes residual floor.** stokes64/stokes128 floor at
  ~3e-14 / ~1e-13, slightly above `ε·√n`. The corpus oracles
  agree, so this is a property of the matrix, not the solver.
  A future "report residual floor as an honest output" feature
  could expose this to callers but is out of scope for #30.
