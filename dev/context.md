# FERAL Context (auto-generated)

Generated: 2026-05-17T22:01:57Z

## Latest Session
File: dev/sessions/2026-05-17-02.md
```
# Session 2026-05-17-02

Continuation of 2026-05-17-01. After auto-CB + scaling fix landed
feral at MA57 parity on the Mittelmann panel, this session shipped
v0.4.0 — cleanly — to PyPI and Crates.io, fixed the wheel pipeline,
and tightened the parity-test gate to match the project's actual
correctness contract.

## Goal

1. Re-audit the README "Known limitations" section against current
   parity-test status.
2. Run the 13 `#[ignore]`'d parity tests on current `main` and
   un-ignore any that close cold.
3. Switch the parity gate from MUMPS-only to oracle-consensus
   (CLAUDE.md correctness contract).
4. Document the Python interface in the README.
5. Publish `feral-solver==0.4.0` to PyPI.
6. Fix whatever the v0.4.0 publish run breaks.

## Accomplished

### Parity — un-ignored 7 panel matrices, filed one real regression

Reran the 13 `#[ignore]`'d parity tests cold against current
`main`. Two passed under the MUMPS-only gate (CERI651DLS_0618 and
ROSZMAN1_0241) and were un-ignored directly.

Switched `tests/parity.rs` from MUMPS-only to oracle-consensus:
feral inertia must match **either** MUMPS 5.8.2 **or** SPRAL SSIDS.
This is verbatim the CLAUDE.md contract:

> Inertia must be exactly correct on non-singular matrices. On
> matrices where the canonical Fortran direct solvers (MUMPS 5.8.2
> and SPRAL SSIDS) disagree on inertia, feral must agree with at
> least one of them.

Updated the generator (`examples/select_parity_panel.rs`) to emit
the new gate and a `try_read_oracle()` helper. Hand-edited
`tests/parity.rs` to match (the example was non-runnable —
`autoexamples = false` in `Cargo.toml`).

Result: **20 passed / 0 failed / 6 ignored** (was 13/0/13 cold).
Newly-passing under oracle-consensus: ACOPP14_{0001,0003},
ACOPP30_{0000,0001}, CERI651CLS_0486.

Genuine outliers still ignored:

| matrix             | reason                                                                |
|--------------------|-----------------------------------------------------------------------|
```

## Git Status
```
2442d1f ci(python-wheels): switch wheel matrix to maturin-action
07d385e ci(python-wheels): bootstrap rustup in manylinux + drop uv frontend
b0c7521 chore(python): bump feral-solver to 0.4.0 to match Rust crate
48c5d13 docs(readme): add Python bindings section
c966c61 test(parity): oracle-consensus gate matches CLAUDE.md correctness contract
```

## Test Status
```
