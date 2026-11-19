# Plan — issue #237 step 4: opt-in InfNorm reuse

Research: `dev/research/issue-237-infnorm-reuse-2026-11-19.md`.

1. `scaling::infnorm::rows_within_ratio(matrix, d, rho)`: one pass with
   the existing serial/parallel sweep; hand-oracle test.
2. `InfNormReuse` + `Solver::with_infnorm_reuse` + cache fields +
   `infnorm_reuse_hit_count`; clear on pattern change / invalidation.
3. `factor_impl`: hit check after the sticky-Auto pin; install/refresh
   after a successful non-hit factor whose route is InfNorm.
4. Solver tests per the research note.
5. Measure on the chain KKT (MetisND, 4 threads): scaling time on hits,
   hit rate, inertia, refined residual; default digests unchanged.
6. CHANGELOG, decisions.md (opt-in reuse vs the #49 entry).
