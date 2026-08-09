# Release 0.15.0 — checklist

Why this release matters more than further optimization: pounce pins
`feral = "0.14.0"` (2026-07-11). Every kernel result from the
2026-08-09 sessions is unreleased, so the 3.5–4.8× factorization gap in
[pounce#552] still describes the **pre-SIMD** kernel — before a ~10×
x86 dispatch fix and a 2.7–7× packed trailing update. Until 0.15.0
ships we do not know the current gap, and further optimization is
guesswork. Cutting the release *is* the measurement.

## 0. Prerequisite

- [ ] Merge PR #150 (task coarsening, env caching, profiler ns, python
      bindings). CI green as of 7ef9e52.

## 1. aarch64 (M-series) revalidation — **COMPLETE, ALL GATES PASS** (2026-08-09)

All 2026-08-09 performance numbers were taken on an x86_64 4-core AVX2
container. The correctness argument is architecture-independent, but it
must be *demonstrated* on aarch64 before shipping, because pounce
publishes macOS arm64 wheels.

> **Result (reported on PR #150):** 1a PASS — 84/84 binaries, 802 tests,
> `golden_bits` 3/3 on aarch64 (x86-recorded digests reproduce → cross-arch
> bit-identity demonstrated); corpus 156,929 matrices, dense + sparse inertia
> both 100.0%, Phase 2.8.1 4/4 PASS, worst residuals identical to the x86
> baseline to the digit. 1b PASS and then some — the packed SIMD default beats
> `FERAL_PACKED_SIMD=0` by 3.19× (n=2955) / 2.25× (n=512) on NEON, so no
> per-arch gate or `MIN_WORK` retune is needed. §2 answered: **keep
> `PAR_MIN_SEEDS = 2`** — coarsening leaves chain KKTs at `seeds=1` so the
> knob is a no-op there, and `marine_1600` loses 28% if it is raised.

### 1a. Correctness (MUST pass — blocks release)

```sh
cargo test --release          # expect 84/84 test binaries green
```

The load-bearing one is `tests/golden_bits.rs`: it asserts hardcoded
`to_bits` digests of L/D recorded on x86_64. **If those pass on
aarch64, cross-platform bit-identity of the new SIMD kernels is
proven.** A failure there is a correctness event, not a tuning issue —
stop and investigate; do not ship.

```sh
cargo run --bin bench --release       # full corpus
```

- [ ] Dense + sparse inertia match 100% (excluding consensus-excluded)
- [ ] Phase 2.8.1 exit partitions all PASS
      (dense small-frontal p90 ≤ 2.0, medium ≤ 3.0; sparse likewise)
- [ ] Worst residuals not materially worse than the 2026-07-11 baseline
      (dense 2.46e-1 POLAK6_0021, sparse 2.94e-4 ERRINBAR_0824)

### 1b. Performance (SHOULD — informs the release note, not a blocker)

The open question is whether the **explicit-SIMD packed kernel** beats
the old autovectorized tile walk on NEON. On x86 the old walk compiled
fully scalar; on aarch64 NEON is a baseline feature, so LLVM may
already have vectorized it and the win could be small or negative.

```sh
cargo run --release --example bench_dense_front -- 2955 5
cargo run --release --example bench_dense_front -- 512 20
FERAL_PACKED_SIMD=0 cargo run --release --example bench_dense_front -- 2955 5
cargo run --release --example bench_schur_micro -- 2048 2048 64 10
```

- [ ] Compare default vs `FERAL_PACKED_SIMD=0` (the A/B that isolates
      this change). Default should win or tie.
- [ ] If the default **loses** on NEON: ship `FERAL_PACKED_SIMD=0` as
      the documented aarch64 mitigation, or gate the SIMD path by
      `cfg(target_arch)`. Do not block the release on it — the
      correctness gates above are what matter.
- [ ] Retune `FERAL_PACKED_SIMD_MIN_WORK` (x86 default 1024) if small
      fronts regress; the gate is bit-neutral.

**Measurement methodology (decisions.md 2026-08-09):** paired
alternating A/B, ≥10 pairs, `min_us` per sample, sign test. Do NOT
compare medians collected at different times — on the container that
produced a 1.9× spread on *identical* code and led to two wrong
conclusions.

## 2. ~~Optional~~ DONE: calibrate `FERAL_PAR_MIN_SEEDS` → keep the default (2)

**Answered: keep `PAR_MIN_SEEDS = 2`.** An earlier reading of the
2026-08-09 review suggested the default was too conservative (the
*pre-coarsening* driver lost to sequential on 4–5 of 6 real matrices).
Calibration superseded that: coarsening collapses five of the six to a
single task (`seeds = 1`), so they already take the sequential fallback
and the threshold is a no-op on them at every setting. It only decides
for trees with genuine parallelism, and there a low value wins —
`marine_1600` loses **28%** at `min_seeds >= 4` (0.78×, 0/15 pairs,
p = 1e-4). Forcing the task graph on (`=1`) costs 3–9% on the five.

It is now runtime-tunable, and byte-identical at every setting
(`tests/task_plan_parity.rs` sweeps 0/1/2/4/64/u64::MAX), so this is a
pure scheduling experiment with zero numerical risk:

```sh
for ms in 1 2 4 8 16 64; do
  FERAL_PAR_MIN_SEEDS=$ms <your harness> <matrix>
done
FERAL_DEBUG_TASK_PLAN=1 <harness>   # prints n_snodes/n_tasks/seeds/cutoff/min_seeds
```

Calibrate on real matrices (clnlbeam, dtoc1nd, steering_12800,
rocket_12800, marine_1600, dtoc2). Expect chain-shaped trees to want a
*high* threshold and wide trees (marine_1600, grid Laplacians) a low
one — on grid250 here, forcing `min_seeds=64` costs 1.85×, so the knob
demonstrably routes both ways. If a single default serves the corpus,
change `PAR_MIN_SEEDS` and record it in decisions.md.

## 3. Release — **SHIPPED 2026-08-09**

- [x] Bump **all six** version strings (a `0.14.0` grep must come back
      empty except for the unrelated `feral-*` ordering crates at 0.2.1):
      `Cargo.toml:13`, `Cargo.lock:198`, `python/Cargo.toml:5`,
      `python/Cargo.lock:62` (`feral`), `python/Cargo.lock:113`
      (`feral-python`), `python/pyproject.toml:7`
- [x] Verify both lockfiles still resolve: `cargo metadata --locked` in
      the root and in `python/`
- [x] `CHANGELOG.md`: move the Unreleased section under `[0.15.0]`
- [x] PR #151 merged as `808babb` (squash). Main CI green on the merge
      commit: CI, Python wheels, Pages all success.
- [x] Tag `v0.15.0` pushed at `808babb`; **GitHub Release published**
      (`release.yml` fires on `release: published`, not on tag push, and
      it verifies the tag matches `Cargo.toml`). Tag/version check
      passed, `cargo test` green in the release job.
- [x] Tag + publish: crates.io `feral 0.15.0` live (all seven crates in
      dependency order); PyPI `feral-solver 0.15.0` live with four
      wheels (macOS universal2, manylinux x86_64, manylinux aarch64,
      win_amd64) plus the sdist. `uv pip` smoke test passed.
- [x] Notify pounce (issue #148 / pounce#552) so the factorization
      comparison can be re-run against a released feral. Posted as
      [pounce#552 comment 5232312068](https://github.com/jkitchin/pounce/issues/552#issuecomment-5232312068):
      what changed, that defaults are bit-identical, and the one-line
      pin bump at `Cargo.toml:127` plus dropping any uncommitted
      `[patch.crates-io]` redirect.

**§3 complete. 0.15.0 is fully shipped and announced.**

## 4. After the release — the real next round

Ordered by the 2026-08-09 review, now unblocked by the nanosecond
profiler:

1. **Scaling warm-start** — the largest single line item: the warm
   prologue is 15–39% of factorization and Knight–Ruiz is 63–81% of it,
   re-derived from scratch every call. `SCALING=none` measures the
   ceiling at −11% (clnlbeam) to −20% (dtoc2). **Changes numerical
   output** → needs a full corpus run.
2. **Amalgamation `nemin` sweep** — 90% of clnlbeam's supernodes are ≤8
   columns. Now measurable (the profiler used to report them as free).
   **Changes fill and numerics** → corpus run.
3. **Permute-cache off-by-one** — call 1 reports `pattern_reused=true`
   yet still pays a cold `from_triplets` rebuild (3 ms clnlbeam, 19 ms
   dtoc2). First-solve latency only; amortizes to ~0 across an IPM.
4. **`nrow` underestimate (#128)** — now load-bearing for *where* task
   boundaries fall, not just a yes/no gate. Sensitivity check on the
   sibling rule.

[pounce#552]: https://github.com/jkitchin/pounce/issues/552
