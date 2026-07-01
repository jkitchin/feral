# issue #102 — parallel dense-front re-entrant deadlock (0% CPU stall) — 2026-07-01

PR #92 regressed two POUNCE mittelmann problems (`cont5_2_4_l`, `dirichlet120`)
from converged to a **300 s timeout at ~0 % CPU** — a synchronization stall in
the parallel dense-front factorization, not slow arithmetic.

## Root cause — nested-rayon re-entrant workspace-mutex deadlock

The parallel multifrontal driver (`factorize_multifrontal_supernodal_parallel`)
gives each rayon worker a per-thread workspace `thread_ws[current_thread_index()]`,
guarded by a `Mutex`. `process_one_supernode` **locks** that mutex and **holds it
across** `factor_one_supernode` → `factor_frontal_blocked_in_place_with_scratch`
→ the intra-front `apply_blocked_schur_panel` `par_chunks_mut`.

That nested `par_chunks_mut` is rayon-on-rayon. While the outer worker waits for
its nested chunks (`WorkerThread::wait_until_cold`), rayon has it **steal another
scope job — another `process_one_supernode`** — and run it *on the same thread*.
That stolen task computes `thread_idx = current_thread_index()` = the **same
index**, and tries to lock `thread_ws[i]` — which its own outer frame already
holds. `std::sync::Mutex` is not re-entrant ⇒ **self-deadlock**, 0 % CPU forever.

Confirmed by `sample` on the hung factor (dirichlet120 KKT, feral `main`):

```
factor_one_supernode → factor_frontal_blocked_in_place_with_scratch
  → rayon ParallelIterator::for_each (intra-front par_chunks_mut)
    → WorkerThread::wait_until_cold            (outer worker steals…)
      → ScopeBase::execute_job_closure         (…another process_one_supernode)
        → std::sync::Mutex::lock → __psynch_mutexwait   (re-locks held thread_ws[i])
```

## What the toggles proved (dirichlet120 KKT factored in feral `main`)

| config | result |
|---|---|
| default (`main`) | **STALL** (>45 s, 0 % CPU) |
| `FERAL_INTRAFRONT=off` | solves, ~765 ms |
| `FERAL_INTRAFRONT_MIN_AREA=65536` (old 256² floor) | **still STALLS** |

So (1) the stall **is** the intra-front nested-rayon path, and (2) it is **not
Lever-B-specific** — the pre-#92 floor deadlocks too. PR #92's *ordering* change
(verify-`LdltCompress`-by-fill, which drops `LdltCompress` here) merely selects a
front that clears the intra-front area gate; 0.11.3's `LdltCompress` ordering
produced a front that stayed under it. The deadlock itself was **latent** since
intra-front parallelism was introduced.

## Fix — `try_lock` the per-thread workspace, throwaway fallback on re-entry

Each `thread_ws[i]` slot is only ever locked by rayon worker `i` (or the donated
caller, last slot), so **cross-thread contention never happens** — a `try_lock`
that `WouldBlock` *uniquely* means this thread already holds it via a nested
re-entry. The driver now `try_lock`s and, on `WouldBlock`, factors with a fresh
throwaway `FactorWorkspace` instead of blocking. Correct because the factor result
is written to the separate `contrib_blocks` / `node_factors_out` mutexes, not the
workspace — only the pooled scratch buffers differ (re-allocated on the rare
nested path). Non-nested calls take the fast path unchanged.

This is the robust fix: it makes intra-front nesting *safe under any ordering*,
rather than papering over #92's ordering choice (which is the correct fill
choice). It preserves byte-exactness (parallel_parity 8/8, blocked_ldlt 21/8,
parity 8/8, factor_workspace_parity 21/21, lib 394/0) and the intra-front speedup.

Result: dirichlet120 KKT factors in ~0.4–0.5 s (10-core), inertia
`(+54122,−241,0)`.

## Regression guard

`tests/issue102_intrafront_deadlock.rs` factors `tests/data/large/dirichlet120_kkt.mtx`
(gitignored; `dev/scripts/regen_dirichlet120_kkt.sh`) on a worker thread with a
120 s wall-clock guard — a returning deadlock fails the test instead of hanging.

## Follow-up (out of scope)

The issue's direction #1 (a dense-front-aware ordering cost signal) is now moot
for correctness — the deadlock is fixed regardless of ordering. It remains a
possible *performance* refinement if a chosen front parallelizes poorly, tracked
under the #99 dense-front-throughput umbrella.
