# `FERAL_*` numeric env knobs: silent fallback to the default (issue #176)

Session 2026-08-19-03. Prereq reading: `issue-148-parallel-task-granularity.md`
(what `FERAL_PAR_TASK_MIN_FLOPS` gates), `dev/sessions/2026-08-19-02.md`
(the `FERAL_CB_THRESH` solve-core work that the reporter was measuring
when they hit this).

## The report

Every numeric `FERAL_*` knob is read with the same shape:

    std::env::var(NAME).ok().and_then(|v| v.parse::<T>().ok()).unwrap_or(DEFAULT)

`parse::<u64>()` rejects `1e18`, `.ok()` throws the error away, and
`unwrap_or` substitutes the default. The knob is *set*, the process
behaves as if it were unset, and nothing is printed. The reporter's
evidence (issue #176):

    $ FERAL_PAR_TASK_MIN_FLOPS=1e18 pounce NARX_CFy.nl --no-sol max_iter=1
    task_plan: n_snodes=45736 n_tasks=21 seeds=11 cutoff=1000000 min_seeds=2

`cutoff=1000000` is `PAR_TASK_MIN_FLOPS`, the built-in default — the
`1e18` did nothing. The same happened with `FERAL_CB_THRESH=1e18`. Two
perf measurements were attributed to "this path costs nothing" when the
path was in fact fully enabled and cost ~15%.

`1e18` is the natural thing to type because the prose and the pounce
option help write the defaults as `1e6` / `1e8`.

## Inventory (the sweep the issue asks for)

Every numeric-valued `FERAL_*` read in the tree, with what it accepts today:

| site | knob | type | today |
|---|---|---|---|
| `src/numeric/solve.rs:788` | `FERAL_CB_THRESH` | u64 | silent default |
| `src/numeric/factorize.rs:3431` | `FERAL_PAR_TASK_MIN_FLOPS` | u64 | silent default |
| `src/numeric/factorize.rs:3423` | `FERAL_PAR_MIN_SEEDS` | usize | silent default |
| `src/dense/factor.rs:728` | `FERAL_INTRAFRONT_MIN_AREA` | usize (>0) | silent default |
| `src/dense/factor.rs:3870` | `FERAL_PACKED_SIMD_MIN_WORK` | usize | silent default |
| `src/capi.rs:118` | `FERAL_PIVTOL` | f64 (finite, >=0) | silent default |
| `src/capi.rs:125` | `FERAL_STATIC_PIVOT` | f64 (finite, >0) | silent default |
| `src/capi.rs:155` | `FERAL_AUTO_CB_BETA` | f64 (finite, >=0) | silent default |
| `src/bin/bench.rs:1159` | `FERAL_DENSE_MAX` | usize | silent default |
| `src/bin/bench.rs:1174` | `FERAL_SPARSE_MAX` | usize | silent default |
| `src/bin/probe_ft_eta.rs:51` | `FERAL_PIVTOL` | f64 | silent default |
| `crates/feral-diagnostics/.../diag_amf_vs_amd.rs:58` | `FERAL_AUDIT_*` | usize | local `env_usize`, silent |
| `crates/feral-diagnostics/.../bench_solver_corpus.rs:52` | `FERAL_BENCH_*` | usize | local `env_usize`, silent |
| `crates/feral-diagnostics/.../probe_wide_supernode.rs:407` | probe knobs | usize | local `env_usize`, silent |
| `crates/feral-diagnostics/.../bench_issue8.rs:138` | `FERAL_ISSUE8_*_GUARD_S` | f64 | local `env_f64`, silent |
| `crates/feral-diagnostics/.../diag_*.rs` | `FERAL_DIAG_*` | usize/f64 | inline, silent |

Boolean / enum knobs (`FERAL_PARALLEL`, `FERAL_PACKED_SIMD`,
`FERAL_INTRAFRONT`, `FERAL_TRACE_SUPERNODE`, ...) are matched against a
literal vocabulary rather than parsed and are out of scope here, with one
precedent worth copying: `FERAL_SCALING` already warns on an
unrecognized value (`src/capi.rs:110`, "X5 follow-up") instead of falling
through silently. That is exactly the behaviour the numeric knobs lack.

## Design

One shared module, `src/env.rs`, public as `feral::env`, with the parse
policy in exactly one place:

- **Accept scientific / float notation.** `1e18`, `1e6`, `2.5e3` and
  plain `1000000` all parse for the integer knobs. The docs write the
  defaults in `1e6` notation, so the input the docs teach must work.
- **Warn, never silently substitute.** A set-but-unusable value prints
  one line to stderr in the established `FERAL_SCALING` shape:

      warning: FERAL_PAR_TASK_MIN_FLOPS="1e" not a number; falling back to default

  and only then falls back. The knob's *intent* is still lost, but the
  measurement is no longer silently invalid — which is the actual
  complaint in #176 ("a knob that silently no-ops is worse than one that
  doesn't exist, because the measurement it produces looks valid").
- **Warn once per (name, value)** — `par_task_min_flops()` is read once
  per factorization and `intrafront_min_area()` once per front, so an
  unconditional `eprintln!` would flood stderr on a corpus run.
- **Clamp, don't reject, an above-range magnitude.** `1e30` on a u64
  knob means "switch this path off"; clamping to `u64::MAX` honours that
  intent, and the clamp is announced on stderr. Falling back to the
  default here would reproduce the exact bug being fixed.
- **Reject negatives and non-finite values** for the counting knobs
  (they have no meaning as a flop count) with the same warning.

### Why not "parse strictly and error out"

The issue offers erroring as an option. Rejected: these are debugging
knobs read from deep inside a factorization that returns
`Result<_, FeralError>` for *numerical* failure. Turning a typo'd
env var into a factorization error would give pounce a numeric-looking
failure for an environment problem. A warning plus a documented,
deterministic fallback keeps the failure legible without inventing a new
error class.

### Rounding

`2.5` on an integer knob rounds half-away-from-zero (`f64::round`) to 3
rather than truncating. Truncation would make `FERAL_PAR_MIN_SEEDS=0.9`
mean 0 ("always parallel"), the opposite of what the value asks for.

## Test oracle

The parse policy is decided here, not derived from a reference solver,
so the oracle is this note plus the values in the issue text:

- `1e18` -> 1_000_000_000_000_000_000 (the reporter's case, verbatim)
- `1e6` -> 1_000_000, `1000000` -> 1_000_000 (the two spellings agree)
- `""`, `"abc"`, `"1e"`, `"-1"`, `"nan"`, `"-inf"` -> warn + `None`
- `"1e30"` on a u64 knob -> clamp to `u64::MAX`, warn
- `"1e400"`, `"inf"` on a u64 knob -> clamp to `u64::MAX`, warn. These
  parse to `+inf`, which is the same request as `1e30` spelled past the
  range of `f64`; routing them to the fallback instead would leave the
  clamp rule with a hole an operator hits precisely by trying to be more
  emphatic. On a *float* knob (`FERAL_PIVTOL`) there is no such reading
  and `inf` stays refused — the asymmetry is deliberate: the unsigned
  knobs count work and saturate, the float knobs are thresholds.
- `"18446744073709551615"` -> exact `u64::MAX` (the integer path must be
  tried before the f64 path: `u64::MAX as f64` is not representable and
  would round to 2^64)

The last row is why parsing goes integer-first and only then float.


## Addendum, 2026-08-19 (review of PR #182)

Two gaps the review found, both now closed, recorded here because each
was a case of the policy being *stated* more broadly than it was
*enforced*.

1. **The clamp had a hole above `f64` range.** `1e30` clamped to
   `u64::MAX`; `1e309` fell back to the default. Both are the operator
   saying "never", and the second silently meant the opposite. Fixed by
   giving `+inf` its own arm ahead of the non-finite refusal.

2. **The source-scan guard was narrower than its claim.** It matched
   only the literal `env::var("FERAL_`, so the local
   `fn env_usize(key: &str, ...)` helpers — the sites easiest to
   reintroduce, because the name is not visible at the read — could not
   trip it; and it exempted anything containing `.split(`, which was
   meant to spare two comma-list knobs but would have spared every
   future one. The scan now matches `env::var(` with any argument and
   carries no exemption but `src/env.rs`. Making it enforceable meant
   converting 21 further sites, all in `feral-diagnostics` and all
   unprefixed (`MAX_N`, `LIMIT`, `PROBE_REPS`, `START`, `STOP`,
   `SAMPLE_STRIDE`, `ONLY`, `PIVTOL`, `AUTO_CB`, ...). They carried the
   identical defect; only the `FERAL_` prefix had been hiding them.

Also split: the behavioural test and the source scan now live in
separate integration binaries (`tests/env_knob_parsing.rs`,
`tests/env_knob_scan.rs`). The first mutates process-global environment
under `unsafe`, which is sound only with no other thread reading the
environment; libtest threads the tests in a binary, so "one test per
binary" is what makes that claim true rather than merely asserted.
