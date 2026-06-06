# Dense-column regression fixture for issue #80 (2026-06-06)

## Problem

Issue #80 (closed) covered two MC64 cost classes on KKTs with a near-dense
coupling column / near-tree structure:

1. **pf22 heap-realloc** (n=2.8M) — fixed (`6699f09`) and guarded by the
   deterministic unit test `mc64_hungarian_no_quadratic_heap_realloc_regression`
   (`src/scaling/hungarian.rs`), which exercises *degree-3 random-sparse*
   ladders only.
2. **rocket dense-column** (n=89601, one degree-38401 column) — inherent cost
   (`dev/research/mc64-dense-column-2026-06-06.md`).

**Coverage gap (this note).** The entire `data/matrices/` tree is gitignored
(`.gitignore:20`, "regenerated from ripopt CUTEst runs"). `git ls-files` shows
**0 tracked files** under `rocket_12800`, `ROSEPETAL`, `ORTHREGF`, and `kkt/`.
So the corpus matrices that exercise the #80 dense-column path exist only as
local harvests — CI and fresh clones have none of them. The only #80 guard that
survives a clean checkout is the synthetic Hungarian unit test, and that test
never sees a near-dense column (it uses degree-3 random sparse). pf22 itself is
too large to commit (n=2.8M).

**Goal:** something *on disk* that reproduces the dense-coupling-column
archetype and is checked for correctness on every `cargo run --bin bench`,
without committing a matrix file or a stored result.

## Decision (user, 2026-06-06)

> "We don't need a committed result, but we need something on disk to check
> for regression." → chose the **bench synthetic tier**.

So: a deterministic *generator* (committed source) that writes the matrix to
disk on every bench run, and a runtime correctness check (inertia + residual)
against an **external** oracle. No committed `.mtx`, no committed solution.

## Archetype: dense-coupling-column symmetric quasidefinite KKT

`K = [[D1, A^T], [A, -D2]]`, `D1` (n_var) and `D2` (n_con) diagonal SPD.

By the **Vanderbei (1995) symmetric-quasidefinite theorem** (`vanderbei1995sqd`),
a matrix of this block form is nonsingular and has inertia exactly
`(n_var positive, n_con negative, 0 zero)` for *any* off-diagonal block `A`.
Inertia is invariant under symmetric permutation, so the layout is free. This
is the external oracle — independent of feral, satisfying the FERAL rule
against same-session implementation+oracle.

`A` has **one dense row** coupling every variable (the near-dense coupling that
drives the #80 MC64 cost) plus a sparse band on the remaining constraints.

### Layout choice — dense constraint at index 0

`to_csc` (`src/io/mtx.rs:21`) builds the CSC straight from the stored
lower-triangle triplets; symmetrization happens later in the symbolic phase.
To make the dense structure a genuine dense **column** in the *stored* CSC
(so it is visible to the symmetric MC64 matching and to the diagnostics'
`max_col_degree`, not just after symmetrization), the dense constraint is
placed at **index 0**:

- index 0: dense constraint. `(0,0) = -d2_0`; couplings `(v, 0)` for every
  variable `v ∈ 1..=n_var` (lower triangle, `v > 0`). → column 0 degree
  `n_var + 1` (a real dense column).
- indices `1..=n_var`: variables, diagonal `+d1_v` (SPD).
- indices `n_var+1 ..`: remaining `n_con-1` constraints, diagonal `-d2_k`
  (negative), each banded to two variables.

This is a symmetric permutation of the SQD block form, so the inertia oracle
`(n_var, n_con, 0)` is unchanged. `max_col_degree ≈ n_var+1`.

## On-disk wiring

`gen_densecol_kkt(n_var, n_con)` emits lower-triangle triplets + RHS + the
analytic `Inertia`. `ensure_densecol_regression_fixtures()` writes, on every
bench run, `data/matrices/synthetic-regression/densecol_kkt_<n_var>/
densecol_kkt_<n_var>_0000.{mtx,json}` (deterministic; overwrites). The dir is
**always** loaded into `kkt_entries` (prepended, independent of
`FERAL_KKT_ROOTS`), so the fixtures run even with no corpus on disk. They then
flow through the existing sparse loop unmodified: symbolic → scaling →
multifrontal → solve → inertia check (`== sidecar oracle`) + relative-residual
check (`<= n·eps·1e6`).

Sizes: `(n_var, n_con) ∈ {(300,3), (1000,4)}`. Small/fast for the default run;
`max_col_degree ∈ {301, 1001}` exercises the dense-column path. Under
`FERAL_SCALING=mc64 cargo run --bin bench` the symmetric MC64 matching runs on
the degree-1001 column — the exact #80 path — with the correctness check live.

## Regression signal

- **Default run** (InfNorm): factors the dense-column SQD KKT, asserts inertia
  `(n_var, n_con, 0)` and tiny residual. Catches dense-column breakage on the
  default path (ordering/preprocess/numeric).
- **`FERAL_SCALING=mc64` run**: drives symmetric MC64 on the dense column;
  same correctness assertions. Catches a re-introduced O(n²) MC64 blow-up
  *with a correctness check* (the existing unit guard checks the structural
  iteration invariant; this checks the numeric result).
- **`cargo test`**: a structural unit test on `gen_densecol_kkt` asserts the
  dense column (col-0 degree `n_var+1`), lower-triangle storage, and the SQD
  inertia oracle — so the generator itself can't silently drift.

## Not in scope

- ROSEPETAL/ORTHREGF compress cost/benefit — a numeric-fill question with no
  cheap structural predictor yet (`dev/research/mc64-symbolic-skip-2026-06-06.md`
  §6); not an invariant to assert, deferred to the separate compress-gate
  workstream.
- Committing the matrices or a result oracle — explicitly out by user decision.
