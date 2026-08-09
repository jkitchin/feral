# Small-front eager path: fused rank-1 update + argmax kernel (Stage 3)

Session 2026-08-09-01. Prereq reading: `kernel-simd-x86-baseline-2026-08-09.md`
(Stage 0 attribution + the Stage-1 dispatch-overhead lesson),
`panel-fragmentation-2026-07-10.md` (corpus steer to the small-front path).

## Why

- The feral/MA57 comparison loses exactly one bucket: small (n=102-888),
  4.23x geomean (`external_benchmarks/comparison/REPORT.md`). The
  pounce#552 report shows the same signature end-to-end: 3.5-4.8x vs
  MA27/MA57 on chain-structured KKTs ("very long chain of tiny blocks"),
  localized by the pounce authors to the numeric kernel.
- Stage-0 attribution on the small parity fixtures: the scalar pivot
  path (`scal%`) is the dominant dense-front bucket (sawpath 89.8%,
  hydcar20 45.6%, twirism1 44.4%); the Schur trailing update is 13.6%
  aggregate. The blocked-path stages (1-2) cannot reach this time.
- The eager path's hot loop is `do_1x1_pivot` (factor.rs): a fused
  scale + rank-1 trailing update + argmax-of-next-column, pure scalar,
  duplicated in the static-floor branch. `do_2x2_pivot` is the rank-2
  analog (dominant on cascade KKTs per the 2026-05-13 pinene profile;
  out of scope for the first cut, extend if 1x1 wins).

## Design

One new kernel in `schur_kernel.rs` (pulp boundary policy):

    rank1_trailing_argmax_nofma(colk, tail, n, k, d) -> (f64, usize)

- `colk` = rows `k+1..n` of the (already 1/d-scaled) pivot column
  (read-only borrow from the `split_at_mut((k+1)*n)` head).
- `tail` = columns `k+1..n` of the front, col-major stride `n`.
- Column `j = k+1+c` update is an aligned axpy: `dst = tail[c*n+j..c*n+n]`,
  `src = colk[c..]`, `alpha_c = colk[c]*d`, per element `dst -= src*alpha`
  (`mul -> sub`, the exact nofma reference shape).
- **One pulp dispatch per pivot step** — the column loop lives inside
  the `with_simd` body. This is the load-bearing design constraint from
  Stage 1: the dispatch boundary costs ~100-200 ns
  (`examples/bench_packed_tiny`), so per-column dispatch on short
  columns would lose. (This is also why the existing per-column
  axpy kernels were never wired into `do_1x1_pivot`: the 2026-05-16
  journal measured pulp == scalar at lengths 3..128 — call overhead
  cancels lane gains. Interior column loop amortizes it.)
- Column `k+1` (c = 0): diagonal handled separately (matches scalar),
  off-diagonal rows tracked for argmax.

## Argmax bit-semantics (no FP rounding involved)

Scalar reference: ascending `i`, `val > best` strictly ⇒ result =
(max |v|, smallest index attaining it); NaN never replaces; all-zero /
empty column returns `(0.0, k+2)`.

SIMD: per-lane running (max, earliest-index) via `greater_than` +
`select`; horizontal reduce merges lane candidates with
`val > best || (val == best && idx < best_idx)`; scalar remainder
elements (indices above every vector element) merge with plain strict
`>`. This reproduces "smallest index among attaining positions"
exactly. Property test pins ties, ±0.0, denormals, NaN against the
scalar loop.

## Bit-exactness of the update

Identical per-element chain (`round(dst - round(src*alpha))`) at every
lane width; columns are independent; `colk` is read-only during the
update. Same argument as Stage 1, enforced by: a bit-for-bit unit test
of the refactored `do_1x1_pivot` against the preserved scalar shape,
`tests/blocked_ldlt.rs` (blocked-vs-eager byte identity), the parity
fixture suite, and the golden digests (which cover the eager path via
`factor_frontal_blocked`'s scalar-tail? — no: golden fixtures route
through the blocked panel path; ADD an eager-path golden case).

## Gate

`m = n-k-1`; SIMD when `m*(m+1)/2 >= 1024` mul-subs (m >= ~45),
mirroring the Stage-1 packed work gate; below, the existing scalar
loops run (bit-neutral gate — both sides byte-exact). Env override
`FERAL_RANK1_SIMD_MIN_WORK` for M-series retune.

## Acceptance / rejection

Accept: >=3% geomean on the small fixture set (3-run medians), no
fixture >2% worse. Reject and record in tried-and-rejected: if the
small fixtures' pivot columns are mostly below the gate (chain blocks
are 6-32 wide) and nothing moves.

## Measured result — REJECTED (reverted same session)

Implemented exactly as designed (one dispatch per pivot step, work gate
1024, scalar argmax re-scan; byte-exact, all 83 suites + golden digests
green). Performance was flat everywhere; the kernel and gate were
reverted, keeping only the de-duplication of the two scalar copies.
Full numbers and the failure analysis: dev/tried-and-rejected.md
2026-08-09 entry. Key datum: direct eager-driver A/B at n=512 measured
11.21 (SIMD) vs 11.23 ms (scalar) — the plain eager loops already
autovectorize, and the eager path is pivot-search/bandwidth-bound.
The small-front/MA57 gap must be attacked elsewhere (per-front
overhead, pivot scans, scalar_pivot_step, delayed-pivot cascade).
