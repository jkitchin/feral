# Plan — one parse policy for the numeric `FERAL_*` knobs (issue #176)

Research note: `dev/research/env-knob-parsing-2026-08-19.md`.

## Steps

1. **`src/env.rs`** (new, `pub mod env`): `u64_var`, `usize_var`,
   `f64_var` plus `_where` variants for the knobs with validity
   constraints. Each returns `Option<T>`: `None` for unset *and* for
   set-but-unusable (after warning), so every call site keeps its own
   default expression, including the closure-shaped one in
   `solve.rs`.
   Pure `*_value(name, raw)` parsers underneath so the policy is
   unit-testable without touching process env.
2. **Tests first**: unit tests in `src/env.rs` against the oracle table
   in the research note; `tests/env_knob_parsing.rs` for the real-env
   path (own binary — it mutates process env) plus a source scan that
   fails if any `env::var("FERAL_...")` in `src/` or `crates/*/src` is
   followed by a bare `.parse()`, so a new knob cannot re-introduce the
   silent shape.
3. **Convert the call sites** in the inventory table, library first
   (`solve.rs`, `factorize.rs`, `dense/factor.rs`, `capi.rs`), then the
   bins, then the four local `env_usize`/`env_f64` copies in
   `feral-diagnostics`.
4. **Docs**: README env-knob table gains the numeric knobs and a note
   that `1e6` notation is accepted and a bad value warns; CHANGELOG
   Unreleased entry.

## Non-goals

- Boolean/enum knobs (`FERAL_PARALLEL`, `FERAL_PACKED_SIMD`, ...) —
  they match a vocabulary, and `FERAL_SCALING` already warns.
- Erroring out on a bad value (see the research note).
- Any change to a knob's *default* or to what it gates. Scheduling and
  numerics are untouched; `tests/task_plan_parity.rs`,
  `tests/cb_core_choice_ignores_env.rs` and `tests/golden_bits.rs` must
  stay green unchanged.
