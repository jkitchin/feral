# Plan — Fix 2: cancellation-free 2×2 inertia classification

Session 2026-05-21-03 · Track A3 fix-forward · follows Fix 1 (`42434a5`)

## Problem

Fix 1 (fine-grained delayed pivoting) made `probe_kkt_replay pinene_3200`
100× faster (456 s → 4.7 s) but returns `WrongInertia` on the borderline
near-singular iterates 8/9 (δ_c ≈ 1e-11): a spurious `inertia.zero`.

Pre-Fix-1 break-on-first cascaded ~116k–133k columns to a dense root
front; full Bunch-Kaufman pivoting there gave Sylvester-exact inertia.
Fix 1's lean delay no longer cascades, which *exposes* — does not cause —
a pre-existing bug in the 2×2 inertia classifier.

## Root cause (see journal 2026-05-21-03 §17:40, §18:05)

`sym2_eigenvalues` computes `λ = 0.5·(tr ∓ s)` with `s` cancellation-free
but the **final subtraction** `0.5·(tr − s)` itself cancels: a genuine
non-singular 2×2 whose small eigenvalue is below `ULP(0.5·tr)`
IEEE-rounds to *exactly 0.0*. `count_2x2_inertia_val` and the `else`
(non-singular) branch of `count_2x2_inertia` then count that as `zero`.

`s vs |tr|` (journaled at §17:40) was **rejected** at §18:05: for a
diagonal block `[[d11,0],[0,d22]]` with `d11` below `ULP(d22)`, both
`tr = d11+d22` and `s = |d11−d22|` lose `d11` — the same cancellation —
so `s == |tr|` and it still mis-reports `det == 0`.

## Fix

Classify the 2×2 inertia from the **cancellation-free sign of the
determinant** (Kahan fused difference-of-products) plus the sign of the
trace. For a symmetric 2×2, `λ₁λ₂ = det` and `λ₁+λ₂ = tr` fix the
inertia from those two signs alone:

- `det < 0` → straddle → (1, 1, 0)
- `det > 0` → both share sign of `tr` → (2,0,0) if tr>0 else (0,2,0)
- `det = 0` exactly → one genuine zero → (1,0,1)/(0,1,1)/(0,0,2) by tr

`det_sym2x2(d11,d21,d22)`: `w = d21*d21; e = fma(d21,d21,-w);
det = fma(d11,d22,-w) + e`. Relative error ≤ 2u for any inputs
(Jeannerod, Louvet & Muller 2013) — `sign(det)` exact unless the block
is genuinely singular to working precision. The product `d11*d22` never
adds `d11` into `d22`, so no diagonal entry is annihilated.

## Steps (FERAL lifecycle)

1. **Tests first** (in-file `#[cfg(test)] mod sym2_inertia_tests`,
   private fns). External oracle = hand calculation (diagonal 2×2 inertia
   by inspection of diagonal signs; det/tr classification is textbook).
   - `[[1e-30,0],[0,1e30]]` → (2,0,0)  — current code gives (1,0,1)
   - `[[-1e-30,0],[0,1e30]]` → (1,1,0) — current code gives (1,0,1)
   - `[[-1e-30,0],[0,-1e30]]` → (0,2,0)
   - genuine singular `[[0,0],[0,5]]` → (1,0,1); `[[0,0],[0,-5]]` →
     (0,1,1); `[[0,0],[0,0]]` → (0,0,2)
   - regression guards: `[[2,1],[1,2]]` → (2,0,0); `[[0,1],[1,0]]` →
     (1,1,0); existing borderline-PD tests stay green.
   - `count_2x2_inertia` else-branch on a genuine non-singular block no
     longer fabricates `zero`.
   Confirm the new tests FAIL on current code first.

2. **Implement**: add `det_sym2x2` + `classify_2x2_inertia`; rewrite
   `count_2x2_inertia_val` to call `classify_2x2_inertia`; rewrite the
   three branches of `count_2x2_inertia` to classify via
   `classify_2x2_inertia` (force-accept bands fold `zero` into `neg`,
   preserving issue #42 Option A `lam>0→pos else→neg`; non-singular
   branch reports `zero` honestly). Keep the `det` param for the
   near-singular gates. Keep `sym2_eigenvalues` (used by
   `perturb_2x2_to_floor`).

3. **Verify on the real matrix**: `probe_kkt_replay pinene_3200`
   (warm + `FRESH=1`) and `probe_issue46_supernode pinene_3200_0008.mtx`.
   Target: pinene iters 8 and 9 → (64000,63995,0); robot/marine
   unchanged.

4. **Benchmark**: `cargo run --bin bench --release` — must stay PASS.

## Risk

Low. The change only touches 2×2 inertia *sign accounting*, not pivot
selection or the numerical update. `classify_2x2_inertia` is
mathematically exact; the regression-guard tests pin existing behavior.
