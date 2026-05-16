# Cascade-break defaults: where `ratio = 0.5` and `eps = 1e-10` come from

Issue: https://github.com/jkitchin/feral/issues/25 (Milestone M2 of
`dev/plans/robustness-roadmap.md`)
Date: 2026-05-16
Author: cascade-break audit, derived from issues #8, #15, #17, the
two prior cascade-break research notes
(`dev/research/issue-15-cascade-break-symbolic-arm.md` and
`dev/research/cascade-break-l-perturbation-2026-05-15.md`), and the
introducing commits `b998e36`, `7998386`, `672ab7a`.

## TL;DR

The two defaults are **empirical**, not derived from a published
criterion. The closest published precedents — Wächter & Biegler 2006
§3.1 inertia escalation, Schenk & Gärtner 2006 supernode-pivot
perturbation, Bunch & Kaufman 1977 partial-pivoting threshold —
all *frame* the problem (perturbation-tax vs delay-cascade trade-off,
pivot-replacement floors near machine precision) but none of them
yield `ratio = 0.5` or `eps = 1e-10` as a derived constant. The
numbers were chosen on `pinene_3200_0009` evidence (issue #8) and
re-validated on `marine_1600`, `qcqp1000-1nc`, `robot_1600` in
issues #15 and #17.

Since they cannot be derived, the project has already taken the
defensible posture: as of session 2026-05-15-02 both defaults were
flipped to `None` (cascade-break is *opt-in*, not auto-armed). The
`0.5` / `1e-10` pair survives only as the recommended starting
values when a caller does opt in. They should be documented as
calibrated, with the regression test that locks them to the
calibration matrix.

## Where the defaults live in the code

Constants and triggers:

- `src/numeric/factorize.rs:140` — `pub cascade_break_ratio: Option<f64>`
- `src/numeric/factorize.rs:156` — `pub cascade_break_eps: Option<f64>`
- `src/numeric/factorize.rs:427-428` — `Default::default()` returns
  `None, None` (current state — opt-in)
- `src/numeric/factorize.rs:1908-1918` — the trigger predicate:

  ```rust
  let cascade_break = match params.cascade_break_ratio {
      Some(r)
          if !is_root[snode_idx]
              && params.allow_delayed_pivots
              && expanded_ncol > 0
              && symbolic.n >= CASCADE_BREAK_MIN_N =>
      {
          (n_delayed_in as f64) / (expanded_ncol as f64) >= r
      }
      _ => false,
  };
  ```

- `src/numeric/factorize.rs:1928-1929` — when the trigger fires the
  supernode-local BK policy switches its zero-pivot action:

  ```rust
  let on_zero = match params.cascade_break_eps {
      Some(eps) => ZeroPivotAction::PerturbToEps { abs_floor: eps },
      None      => ZeroPivotAction::ForceAccept,
  };
  ```

- `src/dense/factor.rs:355-385` — `ZeroPivotAction::PerturbToEps {
  abs_floor }` definition and its corrected docstring (the bound is
  *not* Weyl-localised; see the L-perturbation note).
- `src/dense/factor.rs:391-398` — `perturb_to_floor(d, abs_floor) =
  sign(d) · max(|d|, abs_floor)`.
- `src/capi.rs:48` — the C ABI sets `with_cascade_break_eps(1e-10)`
  for ipopt-feral integration (so consumers downstream of the C ABI
  still see the historical default unless they override).

The original issue body cites `src/dense/factor.rs:2633-2645`. That
line range is the `try_reject_1x1_with_rook_rescue` →
`finish_1x1_outcome` glue (the unblocked-BK 1×1 fallthrough path,
not the cascade-break logic itself). The actual cascade-break
gating and policy override are in `src/numeric/factorize.rs` as
listed above; the `PerturbToEps` *action* is implemented inside
`src/dense/factor.rs` (definition at line 355, applications at
lines 500-518, 2742, 2902-2903, 3278-3279, 3473-3479) which the
issue's line range is pointing at indirectly.

## (a) What cascade-break does, mathematically

**Trigger predicate.** Inside the multifrontal driver, at each
non-root supernode the runtime measures `n_delayed_in` (the number
of columns delayed from child supernodes) and `expanded_ncol` (the
front width after adding those delayed columns). The supernode
opts into a different pivot policy if and only if

```
n_delayed_in / expanded_ncol  >=  cascade_break_ratio
```

and the symbolic problem is large enough to amortise the change
(`symbolic.n >= CASCADE_BREAK_MIN_N = 4096` — issue #15).

**Policy override.** When the trigger fires the supernode runs with
`may_delay = false` (so no further delays leave this node) and the
zero-pivot policy becomes one of two choices controlled by
`cascade_break_eps`:

- `cascade_break_eps = None` → `ZeroPivotAction::ForceAccept`
  (legacy unbounded path; tiny pivots are accepted as-is with the
  L column zeroed below diagonal — this loses inertia in general).
- `cascade_break_eps = Some(eps)` → `ZeroPivotAction::PerturbToEps
  { abs_floor: eps }`, which replaces every failing pivot by

  ```
  d_new = sign(d_orig) · max(|d_orig|, eps)
  ```

  and keeps the L column live. The factorization that comes out
  satisfies `L · D · L^T = A + Δ` exactly (within roundoff), but
  per the May-2026 forensics
  (`dev/research/cascade-break-l-perturbation-2026-05-15.md`)
  the implicit `Δ` is **not** localised to the perturbed diagonal:
  off-diagonal column-k entries are preserved (`Δ[i,k] = 0`), but
  the Schur-update perturbation flows through `L[i,k] · L[j,k] ·
  d_new = A[i,k]·A[j,k] / d_new`, so

  ```
  ‖Δ_schur‖   ≲   ‖A[k+1:,k]‖² · |1/d_new − 1/d_orig|
              ≲   ‖A[:,k]‖² / eps   (worst case)
  ```

  Inertia is preserved provided every nonzero eigenvalue of A
  exceeds the cumulative perturbation — empirically true on
  IPM KKTs at `eps = 1e-10`, but *not* the naive Weyl bound the
  original docstring claimed. See the L-perturbation note for
  the full derivation and the residual measurements on
  `robot_1600_0004` (cb=default unrefined residual ≈ 1e-5 vs
  cb=off ≈ 6e-7; iterative refinement closes the gap).

**Why "cascade-break" is the name.** In multifrontal factorization
with SSIDS-style delayed pivoting, rejected pivots at child
supernodes are passed up to the parent as additional columns to
factor. On certain matrices (notably the IPM KKTs in the issue #8
target `pinene_3200_0009`) METIS-ND ordering concentrates ~118k
delays into three ~14k-column expanded fronts at the root, with
dense O(N³) cost. The "cascade" is the avalanche of upward-
delayed columns; the "break" is the local switch to non-delaying
policy at the overloaded internal node, which absorbs the
perturbation in place instead of pushing it further up.

## (b) Why `ratio = 0.5`

**Empirical origin.** Commit `b998e36` (issue #8, 2026-05-13) chose
`0.5` after a per-matrix sweep on `pinene_3200_0009` with
`cascade_break_eps = 1e-10`. The sweep table (reproduced in the
commit body):

```
ratio=0.25, eps=1e-10 → factor=0.029s, inertia exact, rel_res ~1e-14
ratio=0.50, eps=1e-10 → factor=~0.03s, inertia exact, rel_res ~1e-14
ratio=0.75 / 0.90 / 0.94 / 0.95 / 0.99: all correct under bounded eps
```

Under the bounded-Δ path the trigger threshold is not very
sensitive (any value in `[0.25, 0.99]` gave correct inertia and
~3000× speedup on the target). `0.5` was picked as a defensible
midpoint: "fire whenever the front is at least half delayed
columns" reads cleanly and falls in the safe region.

**Cross-checks at chosen defaults.** Issue #15's evidence sweep
(`dev/research/issue-15-cascade-break-symbolic-arm.md`,
`src/bin/diag_cascade_ratio_distribution.rs`) measured the
empirical distribution of `n_delayed_in / expanded_ncol` across
three families with cb=off (to see what the *natural* ratio would
have been):

| family       | non-root snodes | p99   | max   | would fire @ 0.5? |
|--------------|----------------:|------:|------:|-------------------|
| qcqp1000-1nc |           5,340 | 0.000 | 0.000 | never             |
| marine_1600  |         146,930 | 0.846 | 0.999 | yes (cliff)       |
| pinene_3200  |         105,669 | 0.515 | 0.999 | yes (cliff)       |

The distribution is bimodal — most fronts are at ratio ≈ 0, and a
small cluster of cliff fronts at ratio ≈ 0.85–0.99+. There is no
natural threshold *between* the modes; any value in `[0.5, 0.85]`
catches the cliff fronts and misses the healthy mode. `0.5` is at
the conservative end of that band.

**Is `0.5` derivable from any published criterion?** No, not
directly. Three reasonable starting points and why each fails:

1. **Wächter & Biegler 2006 §3.1 (inertia correction).** Their
   perturbation update for `δ_w` (the Hessian-regularization
   inertia-fix parameter) uses geometric factors of `1/3` and
   `8` (and `100` after the first iterate), bisecting in log-space.
   The `1/2` reduction factor for `δ_w` *after* a successful
   factor is `κ⁻_w = 1/3` by default in IPOPT, not `1/2`. No
   `0.5` constant arises naturally — this is escalation of a
   regularization scalar, not a delay-fraction predicate.

2. **Schenk & Gärtner 2006 (PARDISO supernode perturbation).**
   Their pivot-replacement strategy uses an `α · machine_eps ·
   ‖A‖_∞` floor with `α` tuned per problem class (typically
   `10⁻³` to `10⁻⁸`). They do not gate the trigger by a
   delay-fraction predicate; PARDISO perturbs *every* pivot that
   falls below the floor without a structural gate. So the
   `ratio` parameter has no analogue.

3. **Bunch & Kaufman 1977 §4.** The famous BK constant
   `α = (1 + √17) / 8 ≈ 0.6404` is the *pivot-acceptance*
   threshold (when to take 2×2 vs 1×1), derived from minimising
   element growth. It is unrelated to delay accounting at the
   supernode level. The numerical proximity of `0.5` to `0.6404`
   is coincidence.

**Conclusion for (b).** `ratio = 0.5` is a calibrated value with
empirical justification on `pinene_3200_0009` and validation against
the bimodal distribution observed on the cascade-break-relevant
corpus. It is **not** derivable from the published literature on
pivoting or IPM perturbation. The robust statement is: any value
in `[0.5, 0.85]` on the validated corpus would be defensible;
`0.5` was picked as the conservative end.

## (c) Why `eps = 1e-10`

**Empirical origin.** Commit `b998e36` swept `eps ∈ {1e-8, 1e-10}`
on `pinene_3200_0009`. Both gave exact inertia and `rel_res ~1e-14`
at every ratio tested. `1e-10` was promoted to the default because
it gave a slightly larger safety margin against `eps`-driven
amplification in the (then mis-stated) Weyl bound. The May-2026
L-factor forensics later showed that the bound is not Weyl-localised
but the choice held up on `robot_1600_0004` (rel-res ≈ 1e-5
unrefined, fine after one refinement pass) — the `eps` value
controls the cancellation-error magnitude in the Schur update
rather than the eigenvalue-shift bound.

**Closest published-criterion framing.** `1e-10` sits about six
decimal digits above the IEEE-754 double-precision unit roundoff
`u = 2^{-53} ≈ 1.11 × 10^{-16}`. In Higham's *Accuracy and
Stability of Numerical Algorithms* (2nd ed., 2002, §2.6) the
standard recipe for a pivot-replacement floor is `O(u · ‖A‖_∞)`:
small enough that the perturbation is hidden by working-precision
roundoff on a normalised matrix, large enough that division by
the perturbed pivot does not catastrophically amplify the L
factor. The factor `1e-10 ≈ 10⁶ · u` is consistent with this
"safety margin against rounding-noise activation" framing, and is
of the same order as the PARDISO recipe (`α · u · ‖A‖_∞` with
`α ≈ 10⁻³` to `10⁻⁸`) — but again, the Schenk-Gärtner paper does
not pin a single constant.

**Comparison to peers.**

- LAPACK / SuperLU static pivoting: replacement floor is
  `√u · ‖A‖_∞ ≈ 10⁻⁸ · ‖A‖` after Demmel-Higham. Looser than
  `1e-10` by ~2 decades. Pointed at by the `PerturbToEps`
  docstring (`src/dense/factor.rs:375-376`, citing Trefethen &
  Bau §22) — that exact reference is not in `dev/references.bib`.
- MA57 `cntl(4)`: user-supplied static pivot floor, default
  `0.0` (off). When set, typical IPM usage is `1e-8` to `1e-12`.
  Documented in Duff 2004 (`duff2004ma57` in the bib).
- IPOPT-Pardiso pivot perturbation: `α · ‖A‖_∞ · u` with
  `α ≈ 1e-8` by default, applied to all rejected pivots
  (Schenk & Gärtner 2006).

`1e-10` is in the middle of this published range, but no single
source gives `1e-10` as the derived value.

**Conclusion for (c).** `eps = 1e-10` is consistent with — but not
derived from — Higham's `O(u · ‖A‖_∞)` recipe for pivot-replacement
floors, and falls in the published range used by LAPACK static
pivoting, MA57 `cntl(4)`, and PARDISO supernode perturbation. The
specific value was selected empirically on `pinene_3200_0009` and
holds up under the L-perturbation forensics on `robot_1600_0004`.

**Important caveat.** `cascade_break_eps` is interpreted as an
**absolute** floor — `d_new = sign(d) · max(|d|, eps)` — *not* as
a relative floor `eps · ‖A‖_∞`. On matrices with `‖A‖_∞ ≫ 1` the
absolute floor is overly aggressive (in the wrong direction); on
matrices with `‖A‖_∞ ≪ 1` it can leave near-zero pivots intact.
The C ABI (`src/capi.rs:48`) and the `PerturbToEps` docstring
(`src/dense/factor.rs:377-378`) both note that the recommended
recipe is `eps_rel · ‖A‖_∞` with `eps_rel ∈ [1e-12, 1e-8]`; for
ipopt-feral the equilibration (`d_eq` scaling at factor time,
`src/dense/factor.rs:459`) brings `‖A‖_∞ ≈ 1`, which is what
makes the absolute `1e-10` viable as a default.

## (d) What happens if the defaults change

| direction              | effect                                                                              |
|------------------------|-------------------------------------------------------------------------------------|
| `ratio` smaller (→0.25)  | fires on more supernodes; on `pinene_3200_0009` no correctness loss but the savings plateau (`0.25` and `0.5` both ~30 ms factor) |
| `ratio` larger (→0.85)   | fires only on the highest-cliff fronts; still rescues `marine_1600` and `pinene_3200_*` per #15 data; misses the moderate band `[0.5, 0.85)` (2,800 fronts on `marine_1600`, 657 on `pinene_3200`) |
| `ratio` very large (→0.95+) | misses some cliff fronts where `n_delayed_in / expanded_ncol ≈ 0.85–0.94`; performance regression on the optimal-control corpus |
| `eps` smaller (→1e-12)   | smaller per-pivot perturbation; tighter unrefined residual; risk of insufficient floor (perturbed pivot still ≈ 0 → division blow-up via `1/d_new` factor in L) |
| `eps` larger (→1e-8)     | larger per-pivot perturbation; safer against blow-up; unrefined residual degrades (linearly in `1/eps` in the Schur term) — refinement must do more work |
| `eps` removed (→ None)   | falls back to `ForceAccept`: zeros the L column, sets `D[k,k] = 0`. Loses inertia on perturbed pivots and corrupts solve at position k. Should not be used as a default. |

The robot_1600 sensitivity sweep from issue #17 shows the picture
is more complex than monotonic: `ratio ∈ {0.1, 0.5, 0.8, 0.9}` all
hit MaxIter inside IPOPT, while `ratio = 0.7` is a narrow working
island. The non-monotonicity reflects an interaction with the IPM's
outer perturbation handler that cascade-break's local choice cannot
control — which is part of the evidence that defaulting cascade-
break to *off* (the current state) is the right call.

## (e) Recommendation

**The defaults are empirical, not derivable.** Document them as
such and do not chase a published derivation that does not exist.
The decisive moves are:

1. **Keep `NumericParams::default()` returning `None, None`** —
   already done in the May-2026 flip
   (`src/numeric/factorize.rs:427-428`, recorded in
   `dev/research/cascade-break-l-perturbation-2026-05-15.md`
   §Resolution). This is non-controversial; reverse-discovery of
   "why these numbers" should not require digging through several
   sessions and a 14× IPM regression to find the answer.

2. **Keep the `0.5` / `1e-10` pair as the recommended opt-in
   values** when callers (notably ipopt-feral, see `src/capi.rs:48`)
   want the `pinene_3200`-style cascade-absorption speedup. These
   are the values calibrated on the only published matrix where
   the trigger has demonstrably saved 2600× factor time
   (`pinene_3200_0009`: 88.6s → 34ms, issue #8 commit `672ab7a`).

3. **Pin the values with a regression test** — there is already a
   bench at `src/bin/bench_issue8.rs` (commit `24cc6b9`,
   `bench(issue-8): regression bench for cascade-break defaults`)
   that gates against the calibration matrices. If anyone later
   changes the recommended values, the bench will trip. That is
   the strongest enforcement available given the values cannot
   be derived from first principles.

4. **Do not add a "do not change without revisiting" comment to
   `factor.rs`** as the issue body suggests. The defaults are
   *off* (`None, None`); the comment block at
   `src/numeric/factorize.rs:408-426` already explains why,
   references this note's sibling
   (`dev/research/cascade-break-l-perturbation-2026-05-15.md`),
   and points at the rejected fix attempt. Cross-link this new
   note from there when committing it, but no inline numeric
   constants need a stronger guard than the existing test.

## References

The following references in `dev/references.bib` are relevant; see
note column for their role in this analysis.

| key                  | role here                                                                  | local copy? |
|----------------------|----------------------------------------------------------------------------|-------------|
| `bunch1977stable`    | BK pivoting threshold `α = (1+√17)/8`; not the source of `0.5`             | no, cited from bib only |
| `ashcraft1998accurate` | Modified BK / rook pivoting; relevant to delayed-pivot accounting        | no, cited from bib only |
| `wachter2006ipopt`   | IPOPT §3.1 inertia correction; framing only — not source of `0.5`          | no, cited from bib only |
| `duff2004ma57`       | MA57 `cntl(4)` static-pivot floor (peer recipe for `eps`)                  | no, cited from bib only |
| `desimone2022sparseapprox` | IPM KKT pivoting review                                              | no, cited from bib only |

**Newly added to `dev/references.bib` for this note:**

| key                 | source                                                                              |
|---------------------|-------------------------------------------------------------------------------------|
| `schenk2006fastfact` | Schenk & Gärtner, ETNA 2006 — PARDISO supernode pivot-perturbation             |
| `higham2002accuracy` | Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM 2002    |

**Needs library fetch** (referenced in passing in `PerturbToEps`
docstring and prior notes, but not in `dev/references.bib`):

- Trefethen & Bau, *Numerical Linear Algebra* (SIAM 1997), §22 on
  LAPACK static pivoting. Cited in
  `src/dense/factor.rs:375` but the bib entry does not exist.
  Low priority — it is a textbook citation for context, not a
  load-bearing derivation.

## Pointers to prior in-tree research

- `dev/research/cascade-break-l-perturbation-2026-05-15.md` —
  forensics on the L-factor perturbation bound; corrects the
  original Weyl claim; documents the rejected "L-zeroing fix" and
  the May-2026 default flip to `None, None`.
- `dev/research/issue-15-cascade-break-symbolic-arm.md` — the
  symbolic-arm gate (`CASCADE_BREAK_MIN_N = 4096`) and the
  empirical distribution of `n_delayed_in / expanded_ncol` across
  the qcqp / marine / pinene families.
- `dev/sessions/2026-05-13-04.md` — initial #15 distribution
  measurement; ruled out loosening to `0.85` after FMA was
  identified as the actual source of the qcqp1000-1nc regression.
- `dev/sessions/2026-05-15-01.md` / `-02.md` — `robot_1600` /
  `NARX_CFy` triage, refinement wire-up, default flip to off.
- Commits: `b998e36` (bounded-Δ PerturbToEps introduced, `0.5` /
  `1e-10` sweep), `7998386` (same commit family, eps sweep
  documented), `672ab7a` (auto-arm defaults — later reverted),
  `585d739` (`fix(numeric): make cascade-break opt-in`),
  `24cc6b9` (regression bench), `19c2192` (#15 symbolic-arm
  gate), `da23d13` (#17 C ABI cb=off default).
