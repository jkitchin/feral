# Research: BLAS-3 trailing-update kernel for tall-skinny fronts

Date: 2026-05-12
Status: ACTIVE — informs `dev/plans/phase-2.4.3-blas3-trailing-update.md`
Supersedes (in scope): `dev/research/block32-register-resident-kernel.md`
  for the dominant-cost target. The block32 register-resident work
  (`src/dense/block_ldlt32.rs`, Steps 1+2a landed) remains a separate
  follow-up — it addresses the *within-block* factor, not the
  trailing-update.

## 1. Why this note exists

Issue #9 framed the work as "32×32 in-register SIMD kernel". A profile
of CHAINWOO_0000 (see `dev/research/feral-kernel-profile-chainwoo.md`)
shows the dominant cost is not the 32×32 factor — it is the rank-1
trailing-update that walks the 1984×32 root front. Snode 1933 is
**62 % of total CHAINWOO factor time** at ~13 ms; the inner kernel
(`do_1x1_update` → `axpy_minus_unroll4_nofma`, 32 times) does ~62 MFLOP
at 5–8 GFLOPS.

MUMPS and SSIDS formulate the trailing update as a single rank-`bs`
DSYRK rather than `bs` separate rank-1 axpys; that is where their
~10× arithmetic-intensity advantage comes from. Feral already has a
*dual-column* (NN=2) src-sharing variant of the trailing-update
kernel (`schur_panel_minus_nofma_strided_dual` at
`src/dense/schur_kernel.rs:946`); going wider (NN=4) reduces source
bandwidth and pulp-dispatch overhead per source-vector load by an
additional 2×.

## 2. Current state of the kernel chain

```
factor_frontal_blocked_in_place              -- src/dense/factor.rs:903
  └─ lblt_panel_frontal                       -- panel pivot search
       └─ peek_ahead_column                    -- column rebuild from prior pivots
  └─ apply_blocked_schur_panel                -- src/dense/factor.rs:1867
       ├─ schur_panel_minus_nofma_strided_dual   (NN=2, j step 2)
       └─ schur_panel_minus_nofma_strided        (NN=1, odd tail)
```

`apply_blocked_schur_panel` is the rank-`n_elim` DSYRK flush after a
panel of pivots completes. It walks trailing columns in pairs and
dispatches the dual kernel. The dual kernel holds 4 dst0 SIMD
accumulators + 4 dst1 SIMD accumulators (8 vector registers) and per
`q` loads each src vector once, reusing it for both destination
columns. Bit-exact per column with two sequential rank-1 strided
dispatches.

Per-pivot pulp dispatch counts on the 1984×32 root:

| kernel             | dispatches per panel of 32 pivots | src loads per pivot |
|--------------------|-----------------------------------:|--------------------:|
| single (`_strided`)|              ⌈1952 / 1⌉ = 1952    |             1952    |
| dual (current)     |              ⌈1952 / 2⌉ =  976    |              976    |
| quad (proposed)    |              ⌈1952 / 4⌉ =  488    |              488    |
| oct                |              ⌈1952 / 8⌉ =  244    |              244    |

Each dispatch carries ~20–30 cycles of inline-asm setup overhead, plus
one full pass over the live src column per pivot pair. The dispatch
count drops 2× per doubling; src-bandwidth drops 2× because the same
src column now feeds 4 destinations instead of 2.

## 3. Why quad first (NN=4) — register pressure

Vector register budgets:

| arch     | 64-bit FP regs | f64 lanes per reg |
|----------|---------------:|------------------:|
| NEON     |             32 |                 2 |
| SSE2     |             16 |                 2 |
| AVX2     |             16 |                 4 |
| AVX-512  |             32 |                 8 |

The dual kernel uses an unroll-4 inner body: 4 chunks × 2 dst columns
= 8 acc regs, + 4 src regs concurrently live = 12 regs. Fits AVX2
comfortably.

Naive quad with unroll-4 needs 4 chunks × 4 dst = 16 acc regs + 4 src
+ 4 alpha splats = 24 regs. Spills on AVX2 (16 ymm).

The viable quad shape is **unroll-2 with 4 dst columns**:
- 2 chunks × 4 dst = 8 acc regs (live across q)
- 1 src reg per chunk = 2 src regs (load once per q × chunk)
- 1 alpha splat reg reused
- Total live: ~11 regs. Fits AVX2 with headroom; under-uses AVX-512.

Per q, per chunk: load src once, then apply (alpha0, alpha1, alpha2,
alpha3) sequentially to (dst0, dst1, dst2, dst3) accumulators in that
chunk. That is 4 mul + 4 sub per chunk = 8 SIMD ops on 1 src load.
Compared to dual (2 mul + 2 sub on 1 src load), quad has 2× the
arithmetic intensity per src load.

NEON can fit a wider unroll because of its 32-reg budget; AVX-512 also
can. We pick a single body that works on the worst-budget arch
(AVX2 / SSE2) and accept that AVX-512 leaves regs on the table — the
src-bandwidth halving is the dominant win, not register utilization.

## 4. Bit-parity contract

The rounding contract (`dev/decisions.md:464`, 2026-04-14) requires
that every multiply-then-subtract step is a separate `mul` rounded to
nearest, then `sub` rounded to nearest — no FMA. The dual kernel
already follows this:

```rust
acc = simd.sub_f64s(acc, simd.mul_f64s(alpha_splat, src_vec));
```

The quad kernel must apply the same per-element rounding chain. For
each `(i, q)` element in destination column j, the value computed is

    d[j][i] = d[j][i] - round(alpha_q[j] * src_q[i])    (per q in order)

where the loop order is `for q in 0..n_elim { for j in 0..4 { ... } }`.
Inside one q, each j produces an independent `mul` + `sub` on its
own accumulator — column j sees exactly the same ordered chain that
a single rank-1 strided dispatch on column j would produce. Therefore
the quad kernel is bit-exact per column with four sequential
single-column dispatches.

This is the same parity argument the dual kernel relies on, lifted
to 4 columns. The test oracle is `axpy_minus_unroll4_nofma` (the
single-column SIMD reference) called 4 times.

## 5. Expected speedup

Crude bandwidth model for the 1984×32 root: src column is 1952×8B =
15.6 KiB; L1d on M-series is 64 KiB / core; L2 is 4 MiB. The src
column fits in L1d, so the dual→quad win is primarily *pulp dispatch
overhead* + *register-file pressure reduction*, not memory bandwidth.

Dispatch overhead model: 30 cycles × 976 dispatches = 29 K cycles per
panel = 7.3 µs at 4 GHz. Quad cuts that to 14.5 K cycles = 3.6 µs.
Net savings per panel ≈ 3.7 µs. With 32 pivots in the snode 1933
panel and a 13 ms total, the per-panel cost is the whole frontal —
so 3.7 µs is ~0.03 % of frontal time. **Dispatch overhead is not the
bottleneck.**

The real win is the inner-loop arithmetic intensity: each src-vector
load now drives 8 SIMD FLOPs instead of 4 — meaning the inner loop
moves from src-bandwidth-bound to compute-bound. On Apple M-series
the f64 SIMD throughput is ~2 fmla/cycle = 4 GFLOPS per core on
1-lane unbroken pipelines. The dual kernel measures 5–8 GFLOPS; quad
should measure ~8–12 GFLOPS, a 1.5–2× win on the inner loop.

Carried to the frontal level: snode 1933 at 13 ms → ~7–9 ms,
shaving ~5 ms off the 24 ms CHAINWOO total. That brings ns/nnz from
89 → ~58, still 2× MUMPS but a real step toward closing.

If quad is not enough, the natural follow-up is a true MR×NR tiled
DSYRK where we hold a NR-column-wide × MR-row-tall accumulator and
stream the panel down it once per tile. That is a substantially
larger rewrite and is gated on quad measurement.

## 6. Risks

| risk | mitigation |
|------|------------|
| Quad bit pattern diverges from per-column rank-1 sequence | Inner loop order `for q { for j in 0..4 { mul; sub; } }` keeps each column's chain identical to its single-column dispatch. Bit-parity tests below. |
| Register spill on AVX2 | unroll=2 (8 acc regs, not 16) keeps live registers ≤ 12; compiler will not spill. |
| LLVM fails to keep accumulators in registers | const-generic NN value (not runtime); inner loop bounds known statically; same shape as dual which compiles cleanly. |
| Wiring into `apply_blocked_schur_panel` misroutes pivot scheme | Reuse existing dispatch path: walk `j` in 4s, fall through to dual (2s) for tail of size 2 or 3, then single for tail of size 1. Each fall-through is already tested. |
| Quad does not move the needle | Measure first via the kernel-level micro-bench; do not wire if it does not. |

## 7. References

- `dev/research/feral-kernel-profile-chainwoo.md` — CHAINWOO 89 ns/nnz, Snode 1933 = 62 % of factor time.
- `dev/research/ssids-small-frontal-speed.md` — SSIDS 29 ns/nnz target.
- `dev/research/mumps-small-frontal-speed.md` — MUMPS 14 ns/nnz target.
- `dev/decisions.md:464` (2026-04-14) — bit-exact mul+sub, no FMA.
- `dev/plans/dense-kernel-blas3.md` Phase B-1 — original BLAS-3 plan; this note narrows Phase B-1 to the incremental quad step before considering full MR×NR tiling.
- `src/dense/schur_kernel.rs:946` — `schur_panel_minus_nofma_strided_dual` (NN=2 reference).
- `src/dense/factor.rs:1867` — `apply_blocked_schur_panel` dispatch site.
