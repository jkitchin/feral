# Phase 1b Exit via Multi-Source Consensus — Ultraplan

## Why this plan exists

The Phase 1b exit criterion in `FERAL-PROJECT-SPEC.md` §1712 is:

> 100% correct inertia + solution on the collected KKT benchmark set.

After this session's fixes (postorder, refinement, threshold mismatch),
feral is at:

- Dense:  inertia 99.2%, residual 99.7%, worst residual 8.97e-1 → ~5e-17
  (POLAK6 fixed; ACOPP30 family now dominant at 3.15e-2)
- Sparse: inertia 99.3%, residual 99.8%, worst residual 3.14e-4 (ERRINBAR)

The remaining ~0.7% gap has three identifiable categories:

1. **Inertia disagreements at machine precision** (~880 matrices). Feral
   solves correctly (residual at machine precision) but labels boundary
   pivots differently than the oracle. QPNBLEND, MSS1, CORE1, CRESC*,
   KIRBY2 all fall here. **Not a feral bug.**

2. **Real residual failures concentrated in problem families** (~400):
   ACOPP30 (68 matrices, residual ~3e-2), FBRAIN3LS, CERI651DLS, HS46,
   DEVGLA2, PFIT2, PALMER1ENE, MISTAKE, HATFLDFL. Most have correct
   inertia but bad residuals — these are ill-conditioned KKTs where
   ForceAccept produces a wrong A⁻¹ that refinement can't recover.

3. **Sparse-only failures** (88 matrices). The multifrontal pipeline
   has bugs the dense path doesn't. ERRINBAR_0824 is the worst.

The strict 100% criterion is unreachable without one of:

- Implementing delayed pivoting (Phase 2 feature) for category 2.
- Re-defining "correct" so category 1 is no longer counted as failure.
- Both.

**The deeper problem:** rmumps, the oracle currently used to label the
153k sidecars, is itself a Rust implementation under development by the
same author. It is not the canonical Fortran MUMPS — only a port.
Treating rmumps as ground truth is questionable when feral is being
developed by a process that also produced rmumps. The disagreements may
be feral bugs OR rmumps bugs OR both.

This plan replaces the single-oracle exit criterion with a
**multi-source consensus** across four independent solvers, builds the
two missing canonical oracles natively in Fortran, and defines Phase 1b
exit in terms of agreement with the consensus rather than agreement
with one Rust port.

## The four oracles

| # | Name | Language | Source | Purpose |
|---|------|---|---|---|
| 1 | **feral** | Rust | this crate | system under test |
| 2 | **rmumps** | Rust | `../ripopt/rmumps` | existing sidecar oracle (port of MUMPS) |
| 3 | **MUMPS** | Fortran 5.8.2 | `ref/mumps` | canonical MUMPS reference |
| 4 | **SSIDS** | Fortran (SPRAL) | `ref/spral` | independent canonical multifrontal solver |

These are intentionally diverse:

- Two Rust implementations (feral, rmumps) — useful for catching Rust
  porting bugs.
- One canonical Fortran MUMPS — the original implementation rmumps is
  ported from, runs on the same algorithm.
- One canonical Fortran SSIDS — an *independent* multifrontal solver
  with different pivot ordering, scaling, and pivot acceptance.
  Disagreements with all three Rust+Fortran-MUMPS solvers identify
  matrices where MUMPS-style algorithms have shared blind spots.

The intended invariants once this is in place:

- If all four solvers agree, that answer is **the** answer for the matrix.
- If feral matches the consensus but rmumps doesn't, **rmumps** has a
  bug that the user should fix in `../ripopt/rmumps`.
- If feral doesn't match the consensus, **feral** has a bug that this
  project must fix.
- If rmumps and Fortran MUMPS disagree but feral matches one of them,
  rmumps's port has diverged from canonical MUMPS — informative but not
  feral's problem.
- If MUMPS and SSIDS disagree, the matrix is genuinely ambiguous and
  excluded from the strict exit criterion.

## What "consensus" means operationally

For each matrix `M` in the corpus, compute four (inertia, residual)
pairs `(I_k, r_k)` for k ∈ {feral, rmumps, mumps, ssids}.

### Inertia consensus

Inertia is a discrete vote. For `(positive, negative, zero)` triples:

- **Strong consensus:** at least 3 of 4 inertia triples are equal.
  The agreed value is the consensus inertia.
- **Weak consensus:** exactly 2 of 4 are equal AND the other two each
  differ from the majority by `(±1, ∓1, 0)` or similar
  "near-singular" perturbations. This usually indicates a borderline
  pivot that solver labels differently.
- **No consensus:** all four disagree, or two pairs disagree
  inconsistently. The matrix is genuinely ambiguous.

### Residual consensus

Residuals are continuous. Define `passing(r) := r ≤ n · ε · 1e6` (the
existing bench threshold). A matrix is **consensus-solvable** if at
least 3 of 4 solvers produce a passing residual on the same RHS.

### Per-matrix verdicts

After running all four solvers on matrix M:

| Inertia consensus | Residual consensus | Verdict |
|---|---|---|
| Strong (3+ agree) | Solvable (3+ pass) | **Definitive** — feral must match inertia and pass residual |
| Strong | Not solvable | **Numerically intractable** — excluded from criterion |
| Weak (2 v 2 with similar perturbation) | Solvable | **Borderline** — feral must pass residual; inertia tolerated if within ε of consensus |
| Weak | Not solvable | **Excluded** |
| None | Any | **Excluded** — undefined |

### Phase 1b exit criterion (proposed replacement)

> **Phase 1b exits when feral satisfies the per-matrix verdict for
> every Definitive matrix in the 153k corpus, AND fails on no more
> than X Borderline matrices where X is recorded in dev/decisions.md.**

The size of the Definitive subset is itself a measurement we have to
take — we don't know it yet. The 880 inertia-only failures from
Category 1 will likely move into Borderline or Definitive depending on
what MUMPS and SSIDS say. The 400 residual failures from Category 2
will likely fall into Numerically intractable (excluded) for the
worst-conditioned matrices and Borderline for the rest.

## Architecture

### Where the native oracles live

A new top-level directory `external_benchmarks/` (parallel to `src/`,
`tests/`, `dev/`):

```
external_benchmarks/
├── README.md                  # build instructions, output schema
├── Makefile                   # builds both binaries; pure Make
├── mumps_oracle/
│   ├── Makefile
│   ├── mumps_bench.F          # Fortran 77 driver, MPI-free (SEQ build)
│   └── mumps_bench.json.fmt   # canonical output schema
├── ssids_oracle/
│   ├── meson.build
│   ├── ssids_bench.f90        # Fortran 90 driver
│   └── ssids_bench.json.fmt
├── consensus/
│   ├── compute_consensus.py   # reads all four sources, writes verdict.json
│   └── verdict_schema.json
└── run_all.sh                 # one-shot runner: build + execute all
```

This directory:

- **Is not built by `cargo`** (CLAUDE.md: "Pure Rust, stable toolchain;
  Zero non-Rust dependencies in the core solver"). The constraint is on
  the *core solver*, not the test infrastructure.
- **Is not in CI**. Native oracles run manually on a developer machine
  with MUMPS+SSIDS installed.
- **Outputs are not git-tracked at scale** but a small subset (100
  matrices) can be committed for CI smoke tests.

### Per-matrix output schema

Each oracle writes one JSON file per matrix, alongside the existing
`<id>.json` ipopt sidecar:

```
data/matrices/kkt/<problem>/<id>.mtx           # input matrix (existing)
data/matrices/kkt/<problem>/<id>.json          # ipopt sidecar (existing, contains rmumps inertia)
data/matrices/kkt/<problem>/<id>.mumps.json    # NEW — Fortran MUMPS output
data/matrices/kkt/<problem>/<id>.ssids.json    # NEW — Fortran SSIDS output
data/matrices/kkt/<problem>/<id>.feral.json    # OPTIONAL — feral cached output for comparison
```

Schema (each `.{solver}.json`):

```json
{
  "solver": "mumps-5.8.2",
  "version": "5.8.2",
  "matrix": "MGH10S_0000",
  "n": 51,
  "nnz": 83,
  "factor_us": 1234,
  "solve_us": 567,
  "inertia": {"positive": 35, "negative": 16, "zero": 0},
  "rhs_source": "sidecar",
  "residual_2norm_relative": 1.23e-15,
  "needs_refinement": false,
  "factorization_status": "ok",
  "solver_info": {
    "mumps_infog_1": 0,
    "mumps_infog_28": 0
  }
}
```

The `solver_info` field is solver-specific and used for debugging.
The other fields are mandatory and parseable uniformly.

### Consensus computation

A separate post-processing step (Python or Rust) reads
`<id>.{feral,rmumps,mumps,ssids}.json` for every matrix and writes:

```
data/matrices/kkt/<problem>/<id>.verdict.json
```

```json
{
  "matrix": "MGH10S_0000",
  "consensus_inertia": {"positive": 35, "negative": 16, "zero": 0},
  "inertia_agreement": "strong",
  "consensus_solvable": true,
  "verdict": "definitive",
  "feral_match": true,
  "feral_residual_pass": true,
  "dissenters": []
}
```

Then `bench` reads `verdict.json` files alongside the matrices and
reports against the new criterion.

## Discovery — what's actually available

(From the exploration session before user interrupt.)

- **Toolchain.** gfortran 15.2 (Homebrew), mpif90 (open-mpi), meson,
  cmake, pkg-config — all present. Good.
- **MUMPS.** Two paths:
  - Build from `ref/mumps` (5.8.2) using
    `Make.inc/Makefile.inc.generic.SEQ` (sequential, no MPI). The
    canonical MUMPS source. Estimated 1–3 hours to build first time.
  - Use `/opt/homebrew/lib/libdmumps.dylib` (installed by Homebrew
    `ipopt`, version unknown — likely older 5.x). Faster to use but
    not necessarily 5.8.2. **Decision:** build from ref/mumps for
    canonical reproducibility.
- **SSIDS.** Source in `ref/spral`, meson build. Dependencies:
  - BLAS, LAPACK (Apple Accelerate or OpenBLAS — present)
  - METIS — **not yet installed**. Brew has it via
    `brew install metis` (we can install when ready).
  - hwloc — present at `/opt/homebrew/include/hwloc.h`
  - CUDA — optional, off
- **rmumps.** Already at `../ripopt/rmumps`. Used by `collect_kkt` to
  generate existing sidecars. No build work needed.
- **Existing sidecar contents.** Each `<id>.json` has fields
  `delta_c, delta_w, inertia, iteration, m, n, problem_name, rhs,
  status`. The `inertia` field is rmumps's output. The `rhs` is the
  RHS used to evaluate residuals. To use rmumps as oracle #2 we just
  read the sidecars (no new work).

## Phases of work

### Phase 0 — Decision record (15 min, blocking everything else)

Append to `dev/decisions.md`:

> **2026-04-12** — Phase 1b exit criterion will be redefined in terms
> of multi-source consensus (feral, rmumps, MUMPS, SSIDS) rather than
> agreement with rmumps alone. The new criterion is documented in
> `dev/plans/phase-1b-consensus-exit.md`. This requires building the
> Fortran MUMPS and SSIDS oracles and computing per-matrix verdicts.
> The strict 100%-vs-rmumps criterion in FERAL-PROJECT-SPEC.md §1712
> is superseded for the purpose of declaring Phase 1b complete.

This decision is irreversible per CLAUDE.md (decisions.md is
append-only). User must approve before any other phase begins.

### Phase 1 — Triage cleanup before changing the oracle (30 min)

Before introducing the new oracles, finish the active triage to be
sure the remaining gap is not feral bugs:

- ✅ POLAK6_0021 fixed (threshold-mismatch fix landed in this session,
  residual 8.97e-1 → 4.6e-17).
- ⏳ Triage one ACOPP30 matrix (task #2). 68 matrices with identical
  failure pattern — small chance of a single fix that closes the
  whole family. Worth ~1-2 hours before declaring "needs Phase 2
  delayed pivoting". If the triage finds a fixable bug, fix it. If
  not, document it as Category 2 and move on.
- ⏳ Triage ERRINBAR_0824 (task #3). Worst sparse-only failure. The
  88 sparse-only failures are likely a single sparse-pipeline bug
  (similar in nature to the postorder issue). Worth ~1 hour.

After this phase the *known* feral bug surface is empty and any
remaining gap is genuinely a question of "what does correct mean".

### Phase 2 — Build the rmumps oracle adapter (1 hour)

This is the cheapest oracle because rmumps already exists and the
sidecars already contain its output. The work is just:

1. Write `external_benchmarks/rmumps_oracle/extract_rmumps.py` (or
   `.rs`) that reads each `<id>.json` and writes `<id>.rmumps.json`
   in the canonical schema. This rewrites the existing data into the
   new format.
2. The `factor_us`, `solve_us`, `residual_2norm_relative` fields
   come from re-running rmumps via `collect_kkt --emit-detailed`. If
   collect_kkt doesn't have such a flag, write a small Rust binary
   that loads each sidecar, runs rmumps, and writes the new format.

This phase produces 153k `.rmumps.json` files. Time to run: ~1 hour.

### Phase 3 — Build native MUMPS oracle (2-4 hours)

1. Copy `ref/mumps/Make.inc/Makefile.inc.generic.SEQ` to
   `ref/mumps/Make.inc`. Edit if needed (set BLAS path, etc.).
2. `cd ref/mumps && make d` to build the double-precision sequential
   library. Outputs in `ref/mumps/lib/libdmumps.a`,
   `lib/libmumps_common.a`, `libseq/libmpiseq.a`,
   `PORD/lib/libpord.a`.
3. Adapt `ref/mumps/examples/dsimpletest.F` into
   `external_benchmarks/mumps_oracle/mumps_bench.F`:
   - Add MTX file reader (replace stdin matrix input).
   - Add JSON output writer (Fortran 77 string ops are clunky;
     consider using a small C++ helper or fprintf format strings).
   - Loop over matrices in a directory.
   - Set `JOB = -1` (init), `JOB = 1` (analyse), `JOB = 2` (factor),
     `JOB = 3` (solve), `JOB = -2` (free).
   - Read `INFOG(12)` for negative pivot count → derive inertia.
   - For inertia: MUMPS reports negatives in `INFOG(12)` and zeros
     in `INFOG(28)`; positives = n − negatives − zeros.
   - Compute residual `||Ax−b||/||b||` after solve.
4. `external_benchmarks/Makefile` builds `mumps_bench` linked against
   the static libs in `ref/mumps/lib/`.
5. Run on the 153k corpus. Estimated runtime: ~2 hours
   (matrices are small; MUMPS overhead per matrix is dominated by
   init/free).

**Validation step before running on 153k**: pick 10 matrices we
already understand (MGH10S_0000, POLAK6_0021, ACOPP30_0000,
ERRINBAR_0824, plus 6 simple SPD/diagonal cases). Manually verify
mumps_bench output matches what the existing rmumps sidecars say
(or differs in exactly the way we expect). If they don't match,
fix the mumps_bench driver before running on 153k.

### Phase 4 — Build native SSIDS oracle (3-6 hours)

1. `brew install metis` (and verify openblas/scalapack/hwloc).
2. Build SPRAL via meson:
   ```
   cd ref/spral
   mkdir build
   meson setup build -Dgpu=false -Dexamples=true \
        --prefix=$(pwd)/install
   meson compile -C build
   meson install -C build
   ```
3. Adapt `ref/spral/examples/Fortran/ssids.f90` into
   `external_benchmarks/ssids_oracle/ssids_bench.f90`:
   - Add MTX reader, JSON writer, directory walker.
   - Set `posdef = .false.` (KKT is indefinite).
   - Use `ssids_analyse → ssids_factor → ssids_solve` pipeline.
   - Inertia from `inform%num_neg` and `inform%matrix_rank`.
4. `external_benchmarks/ssids_oracle/meson.build` links against the
   installed `libspral.a` from step 2.
5. Run on 153k corpus. Estimated: ~3 hours.

**Validation step**: same 10-matrix sanity check as Phase 3.

### Phase 5 — Consensus computation (1-2 hours)

1. Write `external_benchmarks/consensus/compute_consensus.py`:
   - For each matrix, load `<id>.{feral,rmumps,mumps,ssids}.json`.
   - Apply the consensus rules from this document.
   - Write `<id>.verdict.json`.
   - Print summary statistics:
     - How many matrices in each verdict bucket
     - Pairwise agreement matrix (feral-rmumps, feral-mumps, etc.)
     - List of matrices where MUMPS and SSIDS disagree (interesting)
2. Aggregate counts across all 153k.
3. Write the per-matrix verdict files.

### Phase 6 — Bench integration (1 hour)

1. Add to `src/bin/bench.rs`:
   - Read `<id>.verdict.json` if present.
   - For each matrix, classify the feral run by verdict.
   - New summary section: "Phase 1b consensus criterion":
     - Definitive matrices: `X / Y passing`
     - Borderline matrices: `X / Y passing`
     - Excluded matrices: count only
2. Bench prints both old (`vs rmumps sidecar`) and new
   (`vs consensus`) numbers side-by-side until the spec is updated.

### Phase 7 — Phase 1b exit decision (30 min)

After Phase 6, the bench output tells us:

- How many Definitive matrices feral fails on (these are the only
  required-to-fix bugs).
- How many Borderline matrices feral fails on (the soft target).
- How many Excluded matrices we have (the documented carve-out).

Three possible outcomes:

A. **Feral passes all Definitive matrices.** Phase 1b exits cleanly
   per the new criterion. Write the exit validation document.

B. **Feral fails N Definitive matrices for small N (say N ≤ 20).**
   Triage each one individually. Likely yields one or two more
   structural fixes like the postorder bug. Re-run, exit.

C. **Feral fails many Definitive matrices.** Indicates a category of
   bugs not yet identified. The triage tooling from this session is
   the right starting point — the failure analysis report will show
   them grouped by family.

### Phase 8 — Spec update (15 min)

Update `FERAL-PROJECT-SPEC.md` §1712 with the new exit criterion.
Append to `dev/decisions.md` summarizing the change. Write the Phase
1b exit validation document (`dev/sessions/<date>-1b-exit.md` or
similar) recording:

- Final consensus numbers
- The Definitive / Borderline / Excluded breakdown
- The list of Borderline failures (if any) with rationale for
  acceptance
- Any Phase 2 work that became obviously necessary

## Cumulative time estimate

| Phase | Duration |
|---|---|
| 0 — Decision record | 15 min (user approval gate) |
| 1 — Triage cleanup (ACOPP30, ERRINBAR) | 2-3 hours |
| 2 — rmumps oracle adapter | 1 hour |
| 3 — Native MUMPS oracle | 2-4 hours |
| 4 — Native SSIDS oracle | 3-6 hours |
| 5 — Consensus computation | 1-2 hours |
| 6 — Bench integration | 1 hour |
| 7 — Exit decision (best case) | 30 min |
| 7 — Exit decision (more bugs found) | 4-12 hours of triage |
| 8 — Spec update | 15 min |
| **Total best case** | **10-17 hours** |
| **Total realistic** | **15-25 hours** |

This is one focused work week if pursued continuously, more likely
2-3 weeks of part-time work with debug sessions.

## Risks

1. **Building MUMPS or SSIDS fails on this machine.** macOS + gfortran
   15 + Apple Silicon is a non-trivial combo. Mitigation: validate the
   build incrementally on the 8-element example matrix in
   `ref/mumps/examples/input_simpletest_real` before doing anything
   complex.

2. **MUMPS and SSIDS disagree more than expected.** If the two
   canonical Fortran solvers disagree on >5% of matrices, the
   "Definitive" set may shrink to the point where the consensus
   criterion is meaningless. Mitigation: if this happens, the
   Borderline category absorbs them with explicit per-matrix
   rationale.

3. **Per-matrix runtime makes 153k runs intractable.** MUMPS
   `JOB=-1; JOB=1; JOB=2; JOB=3; JOB=-2` per matrix is overhead-heavy.
   Mitigation: reuse one MUMPS instance across matrices, calling only
   `JOB=1; JOB=2; JOB=3` per matrix and `JOB=-2` once at the end.
   Same trick for SSIDS.

4. **The 88 sparse-only feral failures are real bugs and Phase 1**
   triage doesn't find them. Mitigation: the triage step has explicit
   permission to run as long as it needs to.

5. **rmumps oracle adapter discovers rmumps bugs.** Likely outcome
   when comparing rmumps vs canonical MUMPS on borderline matrices.
   These are out of scope for this project — file in
   `../ripopt/rmumps`'s issue tracker, document in the verdict file,
   move on.

6. **The new criterion still requires Phase 2 features.** Possible
   that even the consensus says "no, you need delayed pivoting to
   handle these matrices correctly". In that case Phase 1b exit is
   blocked on Phase 2, and we have to either pull Phase 2 forward or
   re-define exit yet again.

## Recommended execution order

**Today's session (whatever's left of it):**
- Phase 0: write the decision record draft, get user approval.
- Phase 1: finish ACOPP30 triage (fast — single matrix, possibly a
  fix). Finish ERRINBAR triage (fast — single matrix). These remove
  the "is this a feral bug?" overhang before changing the oracle.
- Commit the threshold-mismatch fix that landed earlier this session
  but hasn't been committed yet.

**Next 1-2 sessions:**
- Phase 2: rmumps oracle adapter. Cheapest because no new code paths.
- Phase 3 validation step: 10-matrix MUMPS sanity check. Don't run
  the full 153k yet; prove the build works first.

**Once builds are validated:**
- Phase 3 full: 153k MUMPS run.
- Phase 4 full: 153k SSIDS run.

**Final stretch:**
- Phase 5: consensus.
- Phase 6: bench integration.
- Phase 7: triage anything Definitive that fails.
- Phase 8: spec update + exit document.

## What I'm asking you to decide right now

1. **Do you accept the consensus exit criterion in principle?**
   This is a deviation from the spec and should be your call, not
   mine. Per CLAUDE.md, decisions.md is append-only and changes to
   constraints require an explicit decision record.

2. **Where should the four oracles' outputs live?** I've proposed
   sidecar JSONs alongside each `.mtx` file. That makes the data
   self-contained per matrix but adds 4× the file count to
   `data/matrices/kkt/`. Alternative: one big SQLite or parquet file
   with all four solvers' outputs as rows. Easier to query, less
   self-contained.

3. **Phase 1 (triage) before or after building the oracles?** I
   recommend before, because it removes "is this a feral bug or an
   oracle bug" ambiguity. But you may want the oracles built first
   so the triage itself uses the consensus.

4. **How patient are we?** The "best case" is ~10-17 hours of work;
   realistic is more. Some of this is unattended (the 153k runs).
   But most is hands-on. If you want to compress, we can skip SSIDS
   (just 3-source consensus: feral, rmumps, MUMPS) and recover SSIDS
   later as a separate quality bar.

5. **Threshold-mismatch fix from earlier this session — commit it
   now** (it's the POLAK6 fix that already landed in the working tree
   but isn't committed yet) **or wait until after Phase 1?** I'd
   prefer commit now since it's a clean atomic change with passing
   tests, regardless of what happens with the consensus plan.
