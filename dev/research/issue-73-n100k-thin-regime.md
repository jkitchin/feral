# Issue #73 — Characterizing the n>100k thin regime above AMF_BAND_MAX

**Status:** RESOLVED. Real factor+solve A/B (Step 3) ran; it **contradicted the
fill-guard design** and motivated the simpler **unconditional AMF extension**
(drop `AMF_BAND_MAX`). Implemented in `choose_adaptive`. Follow-up to #67.
**Date:** 2026-06-03 (session 06).

> **Outcome (read this first):** The fill-guarded reroute proposed in the
> "Recommendation" section below — option (b), route to AMF above 100k only
> when AMF's `factor_nnz_estimate ≤ MetisND's` — is **WRONG and was rejected**.
> The real factor+solve A/B (Finding 3) shows **nql180** has *smaller* MetisND
> fill yet AMF is **2.05× faster** on the actual factor+solve. Fill (nnz_L) and
> flop_proxy do **not** reliably predict speed, so a fill guard would forfeit a
> 2× win. The shipped change routes **every** would-be-MetisND `Auto` decision
> to AMF at all `n` (the avg_deg<5 → AMD #50 and arrow → AMF #64 catches still
> fire first). The original-status/recommendation text below is preserved as
> the pre-A/B reasoning; Finding 3 is the decisive evidence.

## Question

#67 shipped the thin-large reroute (would-be-MetisND → AMF) but bounded it at
`n ≤ AMF_BAND_MAX = 100_000` (`src/symbolic/mod.rs:225,237`). The bound was set
because, just above the band, `pinene_3200` (n≈128k) favored AMF but
`RDW2D51U` (n≈195k) "did not complete a single Auto+AMF+MetisND pass in ~10
min" and was killed. That measurement used the **full** Solver path (ordering
+ symbolic + numeric + solve, reps=1, all three orderings), so the timeout was
**unattributed**: we could not tell whether RDW2D51U was an AMF *fill* blowup
(a reason to keep the bound) or just an expensive *numeric* factorization
(orthogonal to the ordering choice).

## Method

`crates/feral-diagnostics/src/bin/probe_issue73_symbolic.rs` — **symbolic
only**, no numeric factor, no solve. For each matrix under AMF and MetisND it
reports cheap numeric-cost predictors:

- `sym_ms` — wall time of `symbolic_factorize_with_method` (ordering +
  analysis). Isolates the ordering cost itself.
- `nnz_L_est` — `factor_nnz_estimate` (predicted L nonzeros = fill).
- `max_front` — largest supernode `nrow` (dimension of the biggest dense
  block the numeric phase factors).
- `flop_proxy` — Σ over supernodes of `ncol·nrow²`, a dense-multifrontal work
  proxy (panel factor + Schur update); the dominant term in numeric wall-time.
- `peak_MB` — `peak_contrib_bytes` / 1e6.

Run on the locally-available n>100k KKT families (one iterate, `_0000`).

## Finding 1 — the RDW2D51U #67 timeout was NUMERIC, not an AMF fill blowup

| RDW2D51U (n=195075) | sym_ms | nnz_L_est | max_front | flop_proxy | peak_MB |
|---------------------|-------:|----------:|----------:|-----------:|--------:|
| AMF                 |  167   |  28.19 M  |    1398   |  1.147e10  |  40.1   |
| MetisND             |  944   |  35.59 M  |    1608   |  1.782e10  |  44.9   |

AMF's symbolic analysis finishes in **167 ms**. The #67 ">10 min" was the
**numeric** factorization (28 M nnz_L, a 1398-wide dense front, ~11 GFLOP
proxy) — genuinely expensive, but unrelated to whether AMF or MetisND ordered
it. Moreover AMF is the **cheaper** ordering on RDW2D51U: 1.26× fewer nonzeros
and 1.55× less numeric work than MetisND. The #67 A/B blew its budget on
MetisND's *heavier* factor, not on AMF. Above the band, the bound is leaving
an AMF win on the table here — it is not guarding against an AMF blowup.

## Finding 2 — AMF wins or ties 6/7 of the affected population (symbolic)

Only `n>100k && avg_deg ≥ 5` matrices route to MetisND today (avg_deg<5 →
AMD, the #50 powerflow-class path, unaffected by `AMF_BAND_MAX`). Sweep over
that population (flop_proxy ratio = MetisND/AMF; >1 ⇒ AMF cheaper):

| matrix      |    n | avg_deg | nnz_L M/A | flop_proxy M/A | verdict        |
|-------------|-----:|--------:|----------:|---------------:|----------------|
| dtoc2       | 104k |   17.5  |   1.92    |     4.36       | AMF wins big   |
| pinene_3200 | 128k |    9.42 |   1.21    |     2.43       | AMF wins       |
| cont5_1_l   | 181k |    6.96 |   1.43    |     1.99       | AMF wins       |
| RDW2D51U    | 195k |   18.24 |   1.26    |     1.55       | AMF wins       |
| RDW2D52U    | 195k |   18.24 |   1.26    |     1.55       | AMF wins       |
| YATP1NE     | 246k |    5.97 |   1.66    |    17.55       | AMF wins big   |
| QUADCOPTER  | 280k |    5.14 |   1.00    |     1.00       | tie (near-diag)|
| **nql180**  | 260k |    5.74 |   0.98    |   **0.86**     | **MetisND wins** |

(`optmass`, n=110k avg_deg 3.55, is AMD-routed → unaffected.)

AMF wins clearly on 5, ties 1 (QUADCOPTER, identical predictors), loses on 1.
**nql180** is the lone MetisND win: ~14% less flop_proxy, ~2% fewer nnz_L.
nql180 is a known matrix (cascade-break Free-mode case, see `decisions.md`).

## Finding 3 — real factor+solve A/B: AMF wins ALL, fill mispredicts nql180

Step 3: the actual factor+solve A/B via `probe_issue67_thin --reps 1` on the
MetisND-routed n>100k families. `fill_r = nnzL_met / nnzL_amf`;
`time_r = (fac+slv)_met / (fac+slv)_amf` (>1 ⇒ AMF faster on the real workload):

| matrix    |    n | fill_r | fac_amf | fac_met | time_r |
|-----------|-----:|-------:|--------:|--------:|-------:|
| dtoc2     | 104k |  1.92  |  264 ms |  671 ms |  2.49  |
| pinene    | 128k |  1.20  |  896 ms | 1054 ms |  1.18  |
| cont5_1_l | 181k |  1.35  |  215 ms |  620 ms |  2.75  |
| nql180    | 260k |**0.98**| 1903 ms | 3949 ms |**2.05**|
| YATP1NE   | 246k |  1.44  | 1922 ms | 4104 ms |  2.13  |

(RDW2D51U + QUADCOPTER did not finish the A/B on the loaded test machine; their
symbolic predictors from Finding 2 already favor AMF 1.55× / tie respectively,
and RDW2D51U's symbolic is the cheaper ordering — see Finding 1. They do not
change the conclusion.)

**The design-breaker is nql180.** The symbolic probe (Finding 2) predicted
MetisND *wins* nql180 (nnz_L 0.98×, flop_proxy 0.86× — MetisND has 2% smaller
fill and ~14% less predicted work). On the **real** factor+solve, AMF is
**2.05× faster** (fac 1.90 s vs 3.95 s). So nnz_L — and even the flop_proxy —
**mispredict** nql180. A fill-guarded reroute (route to AMF only when AMF nnz_L
≤ MetisND nnz_L) would have **kept nql180 on MetisND and forfeited the 2×**.

Conclusion: AMF wins factor+solve on **every measured matrix** (1.18×–2.75×),
including the lone fill "counterexample". Fill is not a speed proxy at this
scale, so the correct change is the **unconditional** extension — not a race.

## Decision — unconditional AMF, fill-guard rejected

Drop `AMF_BAND_MAX` entirely. In `choose_adaptive`, the would-be-MetisND
decision is overridden to AMF at **every** `n`:

```rust
if base == OrderingMethod::MetisND {
    return OrderingMethod::Amf;
}
```

The earlier `n > 100_000 && avg_deg < 5 → Amd` (#50 powerflow) and
arrow → AMF (#64) catches fire first and are untouched — the powerflow-class
guardrail still holds. The fill-guarded race (option (b) below) is rejected:
nql180 proves the guard's predicate is anti-correlated with real speed there.
See `dev/decisions.md` and `dev/tried-and-rejected.md`.

## Why no code change yet *(pre-A/B reasoning — superseded by Finding 3)*

1. `nnz_L_est` / `flop_proxy` are **predictions**, not measured factor+solve
   wall time. #67's methodology weighs the real number — e.g. pinene's
   measured `time_r` was 1.18 while flop_proxy reads 2.43 (direction agrees,
   magnitude overstated). **nql180 specifically needs the real measurement**,
   not the proxy, before we trust that MetisND truly wins it.
2. 8 families, one machine, one iterate each — under-sampled for a
   routing-**default** change. #50's powerflow lesson warns against broad
   reroutes on thin matrices.
3. A default change must mirror #67's validation: real factor+solve A/B on the
   affected population + a no-powerflow-regression check.

## Recommendation / next step

The symbolic evidence is strong enough to **justify a full factor+solve A/B**
on the 7 MetisND-routed n>100k families (extending `probe_issue67_thin` to
this regime, with a generous per-matrix timeout since these are minutes-scale
numeric factorizations). Decision rule for a band change:

- If the real A/B confirms AMF wins or ties on factor+solve across the
  population **except** isolated cases like nql180, consider either (a) raising
  `AMF_BAND_MAX`, or (b) gating the extension on a cheap symbolic-fill check
  (route to AMF above 100k only when AMF's `factor_nnz_estimate` is ≤ MetisND's
  — which would correctly keep nql180 on MetisND).
- Record the outcome in `dev/decisions.md` either way.

Option (b) is attractive: the symbolic probe shows the fill predictor already
separates the AMF-wins cases from nql180 at ~zero cost relative to the numeric
factor, so a fill-guarded reroute would capture the wins without the nql180
regression. But that is a design for the next phase, pending the real A/B.

## Artifacts

- `crates/feral-diagnostics/src/bin/probe_issue73_symbolic.rs` (this note's data)
- Journal: `dev/journal/2026-06-03-06.org` (:issue-73:)
- Guardrail unchanged: `n>100k && avg_deg<5 → Amd` (#50) untouched throughout.
