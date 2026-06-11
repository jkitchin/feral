# Verification of repo-review fixes — 2026-06-11

Scope: independent verification of the 85 commits (b80af2d..33bcb25)
that addressed the findings in `repo-review-2026-06-09.md`. Method: six
parallel audits (one per finding series), each re-reading the current
code (not just diffs), red-checking new tests by reverse-applying fix
hunks in a throwaway worktree where feasible, and adversarially
re-testing the rejected findings. Local gates at HEAD: root suite
green, all six ordering-crate suites green, `cargo fmt --check` clean,
`cargo clippy --all-targets -- -D warnings` clean after 1785ba2 (see
REG-0 below).

## Headline

Of the 71 finding dispositions: **~60 verified clean** — genuinely
fixed with real red→green tests, or honestly deferred with accurate
`tried-and-rejected.md` entries. One rejection (D3) was confirmed
justified with strong evidence. The remainder needs action:

| ID | What | Severity |
|----|------|----------|
| REG-1 | N7 fix introduced a silent-wrong-factor regression (stale permute cache) | **high** |
| REG-2 | D5 rejection unsound for `static_pivot_floor > 0` — NaN/inf + wrong inertia reachable, demonstrated | **high** (knob-gated) |
| REG-3 | D4 fix incomplete: sparse solve still has the naive-det gate the dense fix removed | medium |
| REG-4 | X2 silently skipped — only finding with no disposition anywhere; X10's new test enshrines the unvalidated behavior | medium (process + harness integrity) |
| REG-0 | clippy `erasing_op` in D8's test broke `--all-targets -D warnings` (CI) | fixed in 1785ba2 |
| — | residual nits (validation hole, doc residue, untracked facets) | low |

---

## Action required

### REG-1 (high): N7 (`131a6de`) converts a perf fix into a silent wrong answer

`permute_csc_values_cached` now skips the cache *build* when
`pattern_reused_hint == false` but does not *invalidate* an existing
cache, and the warm-path validation (`factorize.rs:3543`) checks only
`(input_n, input_nnz, value_map.len())` — not the pattern. Confirmed
reproduction at HEAD on one `Solver`: factor pattern A twice (call 2
builds A's cache), then pattern B with the same `(n, nnz)` twice —
B-call 2 warm-hits A's stale `value_map`, scatters B's values through
A's structure, and returns Success with solve residual **2.1e+2**
(vs 5.7e-14 on B-call 1). The same probe passes at the parent commit
`0ddaa8b`; the regression is definitively introduced by `131a6de`.
Second route to the same root cause: `invalidate_symbolic_cache()` +
re-factor where AutoRace picks a different permutation for the same
pattern. Exposure: sequential and Schur drivers.

Fix: minimal — `*cache = None` on the `hint == false` cold path (O(1),
preserves the one-shot savings). Durable — store a pattern fingerprint
(or perm hash) in `PermuteCache` and validate it on the warm path.
Regression test: alternate two same-`(n, nnz)` patterns on one Solver,
assert second-pattern second-call residual.

### REG-2 (high, knob-gated): reopen D5 — `perturb_2x2_to_floor` can create an exactly singular 2×2

The rejection's core proof (BK-selected 2×2 has det ≤ −0.59·γ₀² at
default `static_pivot_floor = 0`) was verified correct. But the
auxiliary claim that the perturbation "only lifts eigenvalues away
from 0" is false: τ is added to *both* diagonals, and for a
BK-selected block (opposite-sign eigenvalues) it moves the negative
eigenvalue **toward** zero. Demonstrated through the public `factor()`
API: 3×3 with leading block `[[-0.5, 1.0],[1.0, -0.5]]` (eigenvalues
0.5/−1.5, dyadic, pre-equilibrated), `A[2,0]=0.25`, `A[2,2]=1.0`,
`ForceAccept`, `static_pivot_floor = 2.0` → perturbed block
`[[1,1],[1,1]]`, det exactly 0, unguarded `t = 1/(d00·d11−1)` at
`factor.rs:4777` → `d_diag = [1.0, 1.0, NaN]`, ±inf in L, inertia
(1, 2, 0). The frontal path re-gates after perturbing; legacy
`do_2x2_pivot` does not.

Fix: post-perturbation det gate (or det == 0 guard) in
`do_2x2_pivot`; revisit the D13 deferral (block32 det==0 no-op) at the
same time, since its unreachability argument leans on D5's disposition.

### REG-3 (medium): D4 fixed the dense solve gate but not the sparse one

`dense/solve.rs` now gates 2×2 D-block solves on the shared
`ssids_det_floor_fail` — verified correct and red-checked. But the
production sparse multifrontal solve still uses the old naive
`det.abs() > ff.zero_tol_2x2` gate at `src/numeric/solve.rs:281` and
`:799`, against frontal factors accepted under the SSIDS floor. The
D4(b) scenario (well-conditioned small-scale 2×2 accepted at factor
time, silently skipped at solve time) remains live on the sparse path,
and the dense and sparse solve gates are now mutually inconsistent.
No commit or tried-and-rejected entry covers this.

Fix: port the shared gate to both sparse D-solve sites (forward and
transposed), with a small-scale 2×2 regression test mirroring
`tests/d4_solve_2x2_gate.rs`.

### REG-4 (medium): X2 has no disposition

X2 (MTX declared-nnz never validated against actual entry count;
duplicate entries summed by `to_csc` but overwritten by `to_dense`)
is the only finding in the review with no fix, no
`tried-and-rejected.md` entry, and no journal mention. Both halves are
still live. Compounding it, the X10 fix added
`huge_nnz_header_does_not_abort` (`mtx.rs:260-274`), which asserts
successful parsing of a file declaring 10^17 entries while containing
2 — a future X2 fix must revise that test. X2 was triage-priority 3
(harness integrity) in the review.

Fix: error on entry-count ≠ declared-nnz (and decide one duplicate
semantics for both paths), or record an explicit deferral.

---

## Smaller residuals (queue as opportunistic fixes)

1. **X6 residual hole**: `CscMatrix::validate` still accepts
   `col_ptr[0] != 0` — a monotone `ia` starting at k>0 with
   `ia[n] == nnz` passes all checks while positions `0..k` of
   row_idx/values are silently never factored. Same class X6 was
   about; one-line check.
2. **L2 edge**: sparse diagonal-preference test
   (`sparse_factor.rs:316`) lacks the dense path's `> ztol` conjunct
   (`dense_factor.rs:268`), making the formerly-dead `±ztol` clamp
   (`:322-324`) reachable — a silent sub-tolerance pivot perturbation
   under `Fail` and a dense/sparse drift. Also `utol` is never
   range-validated on either path.
3. **N4 tradeoff visibility**: the pattern-keyed retry latch can
   suppress a values-dependent MC64 rescue (singular at iterate k,
   rescuable pivot-collapse at k+1 on the same pattern → wrong inertia
   where pre-fix it recovered). Acknowledged in the commit message
   only; given the inertia hard rule it belongs in `decisions.md` and
   the field doc (`solver.rs:223-233`).
4. **Stale/incorrect docs**: `factorize.rs:258` still says drivers
   (plural) consult the permute cache — only sequential+Schur do;
   `solver.rs:~650` and `factorize.rs:~216` still describe the static
   floor as `t·‖A‖∞` (unscaled) post-N2; bench.rs:1427-1431 dense-KKT
   rationale comment — the X3 rejection itself called fixing it "the
   real defect, left for a documentation pass" that never happened;
   `feral_capi.h:4-5` still claims codes "mirror Ipopt's
   ESymSolverStatus" (the exact false claim X8 corrected on the Rust
   side) and the header never documents `feral_num_neg`'s −1 sentinel;
   `error.rs:49-53` unqualified "original basis column" claim is
   violated by the position-reporting solve paths, and `NeedsRefactor`
   doc omits the singular-replacement-update case it now covers (L8);
   D7 dispatch comment names a function that no longer exists; D6 test
   header implies Miri flags the pre-fix code, contradicting the
   commit's own honesty note; kahip `graph.rs:31-33` overstates K4
   separator balance wiring.
5. **capi `FERAL_SCALING` still silent on unknown values**
   (`capi.rs:182`) — bench warns, the shim (the tool an IPM host
   actually runs) does not. Half of X5 remains.
6. **Untracked open facets**: N3's `pattern_reused_hint`/`small_leaf`
   parallel-driver facets and N5's parallel-workspace/warm-permute-
   clone facets were honestly scoped out in commit messages but have
   no tracking entry; N1 leaves direct callers of the pub
   `factorize_multifrontal*` drivers (incl. the Schur driver) with a
   silent `NumericParams::fma` no-op (fix landed at the Solver funnel).
7. **Process consistency**: S3/S4 (one-line hardening) were deferred
   under a "no reproducing test possible" doctrine that the same
   sessions did not apply to the equivalent O3/O19 hardening commits;
   the doc sub-items of S5/S10 could ship under the docs-commit
   exemption.

## Verified-clean summary (no action)

- **D-series**: D1 (red-checked, reconstruction err 1.534 → exact),
  D2 (red-checked, panel/scalar inertia parity restored), D6, D7
  (red-checked), D8, D9(a/d/e), D10 (all four sites) — fixed
  correctly. D3 rejection justified (fix attempts provably violated
  the inertia gate on ACOPP30; honest failure record). D9(b/c),
  D11–D13 deferrals accurate (D13 to be revisited with D5).
- **N-series**: N1, N2, N3 (profiler facet), N5 (facet), N6 fixed
  correctly; N4 correct keying/reset; N8–N11 deferrals honest.
- **L-series**: L1, L3, L4, L5, L6, L8, L9, L10, L13 fixed correctly
  with real oracles; L2 correct except the `ztol` edge above; L7,
  L11, L12 deferrals honest.
- **S-series**: S1 provably output-identical with a deterministic
  sort-work regression test; S7 clean; all eight deferrals honest
  (S2's entry correctly narrows an overstatement in the original
  review).
- **X-series**: X1, X4 (semantics verified against `build_matching`,
  fully-matched path bit-identical), X5 (vocabulary), X6 (as
  specified), X7 (sound safe-Rust split borrow + zero-clone test),
  X9, X10, X11, X15 fixed correctly; X3 rejection empirically
  defensible (oracle-pinned inertia on both named matrices + 50-matrix
  sweep, DIVERGE=0); X8/X12/X13 honest.
- **O-series**: all 21 verified — code fixes correct (i64 fill score
  at all three sites, true SHEM via shuffle+stable-sort, GGGP gain,
  FM termination proof, weight-based flow balance), docs accurate,
  and the 3ef8967 KKT test change confirmed NOT a tolerance loosening
  (bug-pinning assertion replaced by strictly stronger ones, justified
  in commit body + journal).

## Suggested order

1. REG-1 (one-line invalidation + regression test) — ships first;
   it is a live silent-wrong-answer hazard on the default path.
2. REG-2 (post-perturbation det gate) + revisit D13.
3. REG-3 (port the SSIDS gate to the sparse solves).
4. REG-4 (X2 disposition) + the X6 `col_ptr[0]` one-liner.
5. Item 3 (decisions.md entry for N4) and the doc-residue batch —
   all eligible for the docs-commit exemption.
