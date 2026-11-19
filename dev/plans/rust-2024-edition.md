# Plan — Rust 2024 edition, resolver 3, MSRV 1.85 (issue #147)

Research: `dev/research/rust-2024-edition-2026-11-19.md`.

1. Baselines (done): exported `feral_*` symbols of `libferal.a` (9);
   harness digests 2b7436a25c646d27 / e321bec7612e0d4f.
2. `cargo fix --edition --workspace --all-targets`, and the same in
   `python/`.
3. Set `edition = "2024"` and `rust-version = "1.85"` in all nine
   crates; `resolver = "3"` in both workspaces.
4. Hand-review `src/capi.rs`; add `unsafe {}` blocks with safety
   comments wherever `unsafe_op_in_unsafe_fn` fires.
5. `cargo fmt` (style edition 2024 may reformat), clippy
   `--all-targets -D warnings`, full `cargo test --release`.
6. Verify: symbols identical, digests identical, `cargo +1.85 check`
   clean, python crate builds.
7. CI: MSRV job (`dtolnay/rust-toolchain@1.85`, `cargo check`).
8. decisions.md entry, CHANGELOG (Unreleased: edition/MSRV, 0.24.0 note),
   journal, session checkpoint.
