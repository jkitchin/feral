# Tried and Rejected Log

Append-only. Do not modify existing entries.

---

## 2026-04-12 — Trace-based 2×2 inertia counting fix (deferred)

**What.** Replace the buggy `count_2x2_inertia` in `src/dense/factor.rs:929`
which uses `a00` to decide the sign of the non-zero eigenvalue in the
near-singular branch with `trace = a00 + a11`. The function comment said
"the other has sign of trace" but the code used `a00` alone.

**Why it's a real bug.** For 2×2 blocks where `a00 = 0` (KKT variable rows
have zero Hessian diagonal — common in ACOPP30, FBRAIN3LS, and similar
problem families), the `if a00 > 0.0` check is false and the inertia
falls into the negative branch regardless of what `a11` says. The
mathematically correct sign comes from the trace.

**Why it was deferred.** The fix was attempted during the ACOPP30
triage in this session. Two findings:

1. **It does not fix ACOPP30.** The blocking matrix
   (ACOPP30_0000 D[58]) has both diagonals zero, so trace is also
   zero. The trace-based fix would output `(0, 0, 2)` for the block
   instead of the buggy `(0, 1, 1)`, but neither matches the rmumps
   sidecar's `(72, 137, 0)`. ACOPP30 is fundamentally a different
   pivot strategy issue (delayed pivoting / Phase 2).

2. **It causes a 16-matrix dense regression on the 153k corpus.**
   With the trace-based fix, dense residual pass count drops from
   152717 to 152701. Sparse drops by 1 (152788 → 152787). The
   buggy code happens to be calibrated to rmumps's behavior on the
   regressed matrices, even though it's mathematically wrong. The
   trace fix is more correct in absolute terms but moves feral
   away from the current oracle.

**Decision.** Revert and re-attempt after canonical Fortran MUMPS becomes
available as a second oracle (per `dev/plans/phase-1b-consensus-exit.md`).
At that point we can verify whether canonical MUMPS uses trace-based or
a00-based inertia counting on the 16 regressed matrices and reapply the
fix in the direction that the canonical solver agrees with.

**Code state.** A `KNOWN BUG` comment is left in
`src/dense/factor.rs::count_2x2_inertia` documenting the issue and
linking back here. The function signature remains unchanged so we don't
need `#[allow(clippy::too_many_arguments)]` for code that we know will
need to change again.

**Symptoms.** Inertia error pattern `(p+1, n+1, 0) → (p, n, +1)` on
matrices with zero-diagonal Hessian rows. The "lost positive" appears
as a "gained zero" in feral's output. Most visible on the ACOPP30
family (68 matrices, all with the same `(72,137,0) → (71,137,1)`
mismatch).
