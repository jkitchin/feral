# Phase 2.5.2 — Rayon on the assembly tree: implementation plan

**Status as of session 2026-04-20-10: DEFERRED.** Planning this session;
execution is a future session. Research note at
`dev/research/phase-2.5.2-rayon-assembly-tree.md`.

## Why deferred

The Phase 2 planning doc's 2.5.2 entry estimates 4-8 hours; a closer
code read (`src/numeric/factorize.rs:383-740`) confirms the scope:
the 200-line per-supernode body mutates five distinct fields of the
shared `FactorWorkspace` plus the shared `contrib_blocks` vector.
Extracting the body into a reusable unit and adding a correct
task-graph parallel driver that preserves bit-exact parity on 154k
KKT matrices is not safely doable in a single auto session after
the 2.5.x/2.5.4 work already completed. Documenting the design is
a higher-quality outcome than shipping a risky partial refactor.

## Committable sub-steps (next session)

### Step A — Dependency + per-task workspace

- Add `rayon = "1.10"` to `Cargo.toml` (workspace-only if needed).
- Derive `Clone` on `FactorWorkspace` or implement a
  `FactorWorkspace::split_off() -> Self` factory that returns a
  fresh workspace with the same capacity hints. Per-task workspaces
  are needed because `row_map`, `frontal_values`, `build_delayed`,
  `build_trailing`, `build_seen`, `dense_values` are mutably used
  inside the supernode body and cannot be shared across threads.
- Exit: tests pass unchanged; no behavioural change.

### Step B — Extract per-supernode body into a helper

- Define a free function
  ```rust
  fn factor_one_supernode(
      snode_idx: usize,
      symbolic: &SymbolicFactorization,
      permuted: &CscMatrix,
      full_pattern: &CscPattern,
      scaling_pivot_order: &[f64],
      is_root: &[bool],
      params: &NumericParams,
      ws: &mut FactorWorkspace,
      contrib_blocks: &mut Vec<Option<ContribBlock>>,
  ) -> Result<(NodeFactors, Inertia, bool), FeralError>
  ```
  that runs lines 491-720 of the current
  `factorize_multifrontal_supernodal_with_workspace` for a single
  `snode_idx`. Returns `(node_factors_entry, inertia_delta,
  needs_refinement_flag)`.
- Rewrite the sequential driver to call this helper in postorder.
- Exit: sequential path bit-exact against pre-change; all tests
  pass; 154k-matrix full corpus run shows zero inertia/residual
  drift.

### Step C — Task-graph parallel driver (first cut)

- Build a reverse-children map: `parents: Vec<Option<usize>>`
  where `parents[i]` is the index of the supernode containing `i`
  as a child (None for roots).
- Build a "ready" counter: `unresolved_children: Vec<AtomicUsize>`
  initialised to `children.len()` for each supernode.
- Use `rayon::scope` to spawn tasks: a supernode is "ready" when
  `unresolved_children[i] == 0`. When a leaf finishes, it
  decrements the parent's counter and spawns the parent if the
  counter hits zero.
- Shared state:
  - `contrib_blocks: Arc<Mutex<Vec<Option<ContribBlock>>>>` OR
    `Vec<Mutex<Option<ContribBlock>>>`. The latter is
    finer-grained. A `Vec<OnceCell<ContribBlock>>` may work if we
    ensure write-once semantics.
  - `node_factors: Arc<Mutex<Vec<Option<NodeFactors>>>>` with
    `node_factors[snode_idx] = Some(...)` by each task; after
    scope ends, collect in postorder.
- Per-thread workspace: `thread_local!` or pass via rayon's
  thread-local storage. Safest is to use `rayon::ThreadPool` and
  own one workspace per thread.
- Exit: bit-exact parity with sequential path on 154k matrices;
  demonstrable speedup on large matrices (CRESC132-class).

### Step D — Threshold gate

- Only engage the parallel driver when:
  - `n_snodes >= N_PAR_MIN` (tentative: 32)
  - the etree has at least one node with ≥ 2 non-trivial children
- Below the threshold, fall through to the sequential driver.
- Exit: no regression on the small-matrix tail of the corpus.

### Step E — Validation and bench

- Corpus cross-check: walk all KKT matrices, factor both
  sequentially and in parallel, assert bit-exact inertia and
  factor NNZ equality. (Residual equality cannot be bit-exact in
  general once FP ordering changes — accept ≤ 2 ULP drift on the
  residual norm, with zero inertia drift.)
- Full bench: p90 factor/MUMPS before/after.
- Target: ≥ 1.5× speedup on dense-root large matrices, no
  regression on p90.

## Risks and decisions to make in the implementing session

- **FP determinism.** `extend_add` scatters child contribs into a
  parent frontal. Order of child scatter matters for FP sums if
  two children's contribs overlap on the same (i, j) cell. Today
  this is deterministic (postorder child iteration). Parallel
  sibling processing changes completion order. Mitigation: sort
  `snode.children` by a stable key (e.g., snode_idx) before
  scatter, OR serialise the scatter phase per-parent.
- **Inertia stability.** A parent's pivot selection is sensitive
  to the assembled frontal's entries. FP scatter order affects
  entries → affects pivot path → can affect inertia counts on
  ill-conditioned matrices. Mitigation: the sorting-by-key
  trick keeps the scatter order identical to sequential.
- **Memory.** Per-task workspaces multiply peak memory by
  `num_threads`. For `num_threads = 8` and a 128 GB frontal on a
  worst-case matrix, this explodes. Mitigation: cap per-task
  work by subtree size; dispatch small subtrees through the
  sequential path even under the parallel driver.

## Notes

The Phase 2 planning doc's mention of `ContribPool` as a LIFO
stack is aspirational — the current code uses
`Vec<Option<ContribBlock>>` with per-supernode `Vec<f64>` storage.
A buddy allocator for contribs is a separable Phase 3 item;
2.5.2 does not require it to ship a first parallel driver.

## Entry conditions for the implementing session

- Session duration ≥ 4 hours (this is Step B + C + D + E).
- Clean working directory; no other in-flight refactors.
- Fresh benchmark baseline recorded in-session.
