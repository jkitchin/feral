# Plan — issue #127 symbolic pipeline split

Refactor `src/symbolic/mod.rs` only. No public API change.

## Steps

1. **`struct SymbolicPrefix`** (private): `n`, `perm`, `perm_inv`,
   `permuted_pattern`, `etree`, `col_counts`, `factor_nnz`,
   `factor_nnz_estimate`, `factor_slack`, `cached_mc64`, `resolved_method`,
   `resolved_preprocess`, `effective_params: SupernodeParams`,
   `t_total: Option<Instant>`.

2. **`fn symbolic_prefix_concrete(matrix, snode_params, method) ->
   Result<SymbolicPrefix>`** — move body `mod.rs:1068..=1335` here. Assumes
   `method` is concrete (not AutoRace) and `preprocess` is concrete
   (`Auto` resolved by the caller for the *fill race*; the internal
   `resolved_preprocess` match at 1117 still handles `None`/`LdltCompress`/
   `External`-forced-`None`). Returns the struct; runs no tail stage.
   Captures `t_total` at entry.

3. **`fn symbolic_finish(prefix: SymbolicPrefix) -> SymbolicFactorization`** —
   move body `mod.rs:1337..=1407` here. Reads profiler from
   `prefix.effective_params.symbolic_profiler`. `#[cfg(test)]` increment of a
   `FINISH_RUNS` atomic at the top. `set_total` from `prefix.t_total`.

4. **`fn symbolic_prefix_preprocess_auto(matrix, snode_params, method) ->
   Result<SymbolicPrefix>`** — rewrite of today's
   `symbolic_factorize_preprocess_auto` to return the winning **prefix**
   instead of a finished factorization. Same predicate, same
   `PREPROCESS_FILL_INFLATION_LIMIT` compare on `factor_nnz_estimate`, same
   fresh-profiler-per-arm handling — but no `copy_profiler` here (deferred to
   the finisher). Each arm calls `symbolic_prefix_concrete` with the arm's
   `preprocess` forced.

5. **`fn resolve_prefix(matrix, snode_params, method) ->
   Result<SymbolicPrefix>`** — External validation (as at 1053-1056);
   if `preprocess == Auto && !external` → `symbolic_prefix_preprocess_auto`,
   else → `symbolic_prefix_concrete`.

6. **`fn finish_and_copy(prefix, shared: Option<&Arc<Mutex<..>>>) ->
   SymbolicFactorization`** — clone the prefix's profiler arc; `symbolic_finish`;
   if `shared` is Some and not `Arc::ptr_eq` to the prefix's arc, copy the
   prefix arc's snapshot into `shared`.

7. **Rewrite `symbolic_factorize_with_method`**: keep the `AutoRace` guard
   (→ `symbolic_factorize_race`); otherwise
   `Ok(finish_and_copy(resolve_prefix(matrix, snode_params, method)?,
   snode_params.symbolic_profiler.as_ref()))`.

8. **Rewrite `symbolic_factorize_race`**: for each `RACE_CANDIDATES`,
   fresh cand profiler, `resolve_prefix(matrix, &cand_params, cand)`; keep the
   prefix with the smallest `factor_nnz_estimate` (same `<` as today);
   `finish_and_copy(best_prefix, snode_params.symbolic_profiler.as_ref())`
   once. Error only if every candidate failed.

## Tests first (write before steps 2-8)

- `tests/issue127_pipeline_split.rs`:
  - `autorace_matches_winning_concrete_method` — over a few real/synth KKTs.
  - `preprocess_auto_matches_winning_arm` — a None-wins and an
    LdltCompress-wins matrix (reuse fixtures from `issue91_preprocess_misfire`
    / `auto_strategy` if handy).
- in-module `#[cfg(test)] mod tests` in `mod.rs`:
  - `finish_runs_once_under_preprocess_auto`
  - `finish_runs_once_under_autorace`
  (reset `FINISH_RUNS`, factorize, assert `== 1`).

## Bench

`cargo run --bin bench --release` full corpus (inertia gate). Plus a
symbolic-phase A/B on an Auto-firing KKT if a probe is cheap. Record in the
checkpoint per protocol.
