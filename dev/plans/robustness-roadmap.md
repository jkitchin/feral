# Feral robustness roadmap

Goal: make feral correct, residual-clean, and well-conditioned across a
broad range of symmetric-indefinite problems, not just the curated KKT
corpus we already do well on.

Driver: the stress suite at `external_benchmarks/stress/` is the
acceptance gate. Every milestone below ends with `report.py` passing
on a larger fraction of the manifest.

## Baseline (this branch, 2026-05-16)

Stress harness built. Full-manifest baseline captured in
`external_benchmarks/stress/baseline.txt` and `baseline.json`.

  - **total**: 28 matrices (18 SuiteSparse GHS_indef + 10 synth)
  - **ok**: 26
  - **flagged**: 2
    - `rankdef_200_20`: synth, `(112, 88, 0)` vs expected `zero=20`
      → **F-01** (issue #21)
    - `bloweybl`: real, `NumericallyRankDeficient` rejection on
      saddle-block with 2/3 zero diagonals; sibling `bloweybq`
      factors fine → **F-03** (issue #32)
  - **dropped**: `copter1` — SuiteSparse mirror returns 404; will be
    replaced during M3.

All 7 saddle matrices pass (`turon_m` n=189924 in 199ms). All PDE,
near-singular, ill-conditioned, dense, and cascade matrices pass.
Worst residual under the flag threshold: `stokes128` at 9.6e-14.

To reproduce:
```
cargo build --release --bin bench_one_matrix
python3 external_benchmarks/stress/synth.py
python3 external_benchmarks/stress/fetch.py
python3 external_benchmarks/stress/run.py
python3 external_benchmarks/stress/report.py
```

## Milestone plan

### M1 — Coverage (1 session) — issue #22

Wire `fetch.py` into a one-shot setup and run the full manifest baseline.

- [ ] Run `fetch.py` for the 19 SuiteSparse matrices
- [ ] Run `run.py` over the full sample, record results
- [ ] Tag each failure / high-residual in this doc as a finding
      `F-NN` with file:line evidence (probe with `bench_one_matrix`
      manually for any that hang or crash)
- [ ] If any matrix exceeds 10 min, mark it `tier=large` in the
      manifest and exclude from default smoke
- [ ] Add `report.py` invocation to CI as a non-blocking job
      (after baseline is captured — won't block PRs until we know the
      stable expected-pass list)

### M2 — Fix scoped weak spots (2-4 sessions)

Each item already has a research note or open finding; this is the
work of actually resolving them.

- [ ] **F-01 rankdef_200_20** — issue #21
- [ ] **MSS1 inertia monotonicity** — closed in #5; re-add MSS1 to
      stress manifest as `category=opt` to prevent regression
- [ ] **ACOPP30 residual plateau** — issue #23
- [ ] **MC64 silent fallback** — issue #24
- [ ] **Cascade-break perturbation semantics** — issue #25
      (related: #15 closed, #17 open)

### M3 — Broaden the SuiteSparse corpus (1-2 sessions) — issue #26

Triple the stress manifest by pulling more of `GHS_indef` plus adjacent
collections.

- [ ] Add the remaining GHS_indef matrices in the n ≤ 100k tier
      (~40 more rows)
- [ ] Add the `Schenk_IBMNA` and `Schenk_AFE` groups (sparse indef PDE)
- [ ] Add `Boeing/bcsstk*` (mechanics, indefinite stiffness)
- [ ] Add Stokes / driven-cavity from the `Mittelmann` set if not
      already in `data/matrices/kkt-mittelmann/`
- [ ] For each, mark `category` correctly so report.py rolls up cleanly

### M4 — Synthetic generators for under-served pathologies (1 session) — issue #27

The synthetic side of `manifest.tsv` only has 4 categories. Extend.

- [ ] **Saddle blocks with known nullity** (`[H A^T; A 0]` with rank
      deficiency in A): inertia oracle is `(?, ?, dim ker(A))`
- [ ] **Wide-frontal cases** (parametric n × n with supernodes > 1000)
      to stress the sparse kernel
- [ ] **MC64-resistant matrices** (matchings exist but scaling stays
      bad) — sourced from the cases in
      `dev/research/mc64-failure-modes.md` if present, else synthesize
- [ ] **Structured indefinite saddle from Stokes**: Q1-P0 / TH-P1
      discretizations on a small grid, parametric by mesh density

### M5 — Wire stress into CI gate (1 session) — issue #28 (blocked by #22)

Once M1 baseline is stable for two sessions in a row.

- [ ] Add `external_benchmarks/stress/` smoke run (synth-only +
      n ≤ 1000 SuiteSparse subset) to the GitHub Actions workflow
- [ ] Fail the job if `report.py` exits non-zero
- [ ] Cache the matrices/ directory keyed on manifest hash
- [ ] Document the gate semantics in `external_benchmarks/stress/README.md`

### M6 — Deep dives (continuous)

Open-ended investigation work that can run in parallel with the
milestones above. Each item gets its own research note under
`dev/research/`.

- [ ] **FBRAIN3LS 2x2 pivot stability** — issue #29
- [ ] **Deep-elimination null cascades on real matrices**
      (synthetic `deep_null_cascade_*` already passes; real-world
      cases likely don't — needs corpus + issue once a failing
      example surfaces from M1 baseline)
- [ ] **Iterative-refinement convergence policy** — issue #30
- [ ] **Inertia certification on near-singular** — issue #31

## Sequencing rationale

- M1 unblocks everything — without baseline numbers we can't measure
  regressions.
- M2 buys back known issues that already have research notes; cheapest
  reliability wins per hour.
- M3 vs M4 are independent and could be interleaved. M3 has higher
  signal-per-effort (real-world matrices catch unknown bugs); M4
  closes oracle-verifiable holes.
- M5 must come after M1-M4 settle: gating on a flaky baseline is worse
  than no gate.
- M6 runs in parallel; each item is a research note + maybe a tweak.

## How to track progress

- Each milestone ends with a commit on a feature branch + a session
  checkpoint
- Findings (`F-NN`) accumulate at the top of this doc with
  status: `open / wip / fixed`
- Per-milestone, before merging to main, baseline `report.py` output
  is committed to `external_benchmarks/stress/baseline.txt` so we can
  see what each milestone moved

## Findings log

| ID    | matrix          | symptom                                              | issue | status |
| ----- | --------------- | ---------------------------------------------------- | ----- | ------ |
| F-01  | rankdef_200_20  | reports `zero=0`, expected `zero=20`                 | #21   | open   |
| F-03  | bloweybl        | `NumericallyRankDeficient` on saddle w/ 2/3 zero diag | #32  | open   |
