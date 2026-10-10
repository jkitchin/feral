# Plan: issue #215, auto re-analyse on cascade-break

Status: REJECTED, code removed. See `tried-and-rejected.md` 2026-10-10.

Research: `dev/research/issue-215-reused-analysis-cb.md`.

1. Tests first, in the `solver.rs` unit tests (they need private access to
   the cached symbolic):
   - Comparator: `reanalysis_wins(old, new)` follows the documented order
     (CB, then tfa + n_tiny, and the incumbent keeps exact ties). The oracle
     is the rule written in the research note.
   - Trigger: a KKT `[H Aᵀ; A -δI]`-style matrix (H SPD, A full row rank)
     analysed with `preprocess = LdltCompress`. Factor once, then zero the
     cached symbolic's `delayed_capacity` to simulate a stale analysis that
     forces CB, and refactor. Expect: the re-analysis runs, the counter
     increments, the new factor is adopted with CB = 0, and the inertia
     equals the theoretical (n_x, m, 0). The oracle is Sylvester's law of
     inertia applied to the KKT structure.
   - Gate: the same setup with `with_auto_reanalyse_on_cb(false)` keeps CB
     and does not re-analyse. A pattern-only analysis (preprocess None) does
     not re-analyse.
   - Stats: on a fresh Solver whose first factor escalates,
     `pattern_reused == false`.
2. Implement in `Solver::factor`:
   - Add an inner `factor_impl(matrix, check, reentry_pattern_reused)` that
     escalation calls with `Some(pattern_reused)`.
   - Add fields `auto_reanalyse_on_cb: bool` (default true),
     `reanalyse_miss_streak: usize`, `auto_reanalyse_count`,
     `auto_reanalyse_adopted_count`, plus a builder and getters.
   - The hook goes after the escalation check in the success branch.
3. Validate with `cargo test`, clippy and fmt, then the seq215 probe
   (warm it004 should re-analyse and keep the better factor).
4. Write CHANGELOG, decisions.md, journal, bench and session checkpoint.
