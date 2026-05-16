# Stress-corpus M3 expansion (issue #26)

Triple the stress-suite SuiteSparse corpus from 18 rows (all `GHS_indef`)
to ≥80, spanning Schenk_IBMNA, additional GHS_indef, and Boeing
mechanics. Final landed count: **104 SuiteSparse rows** plus 18 synth
and 3 cuter_kkt = 125 manifest rows total.

## Groups and where they live

All SuiteSparse downloads come from
`https://suitesparse-collection-website.herokuapp.com/MM/<group>/<name>.tar.gz`
via `external_benchmarks/stress/fetch.py` (unchanged). Local copies
land under `external_benchmarks/stress/matrices/<group>/<name>.mtx`.

| Group           | Catalog URL anchor                            | Rows added | Total in manifest |
| --------------- | --------------------------------------------- | ---------- | ----------------- |
| `GHS_indef`     | <https://sparse.tamu.edu/GHS_indef>           | 33         | 51                |
| `Schenk_IBMNA`  | <https://sparse.tamu.edu/Schenk_IBMNA>        | 49         | 49                |
| `Boeing`        | <https://sparse.tamu.edu/Boeing>              | 4          | 4                 |
| `Schenk_AFE`    | <https://sparse.tamu.edu/Schenk_AFE>          | 0          | 0 (see below)     |
| `synth`         | local `synth.py`                              | 0          | 18                |
| `cuter_kkt`     | local `data/matrices/kkt/<family>/...`        | 0          | 3                 |

Per-matrix entries are listed in `external_benchmarks/stress/manifest.tsv`.
The `notes` column on each new row identifies the matrix class
(saddle/PDE/QP/mechanics/etc.) for human readers; the structured
`category` column drives the report.py roll-up.

## Category assignment

`report.py`'s per-category roll-up is driven by the `category` field.
The existing taxonomy
(`saddle`, `pde`, `opt`, `mech`, `dense`, `indef`, plus synthetic
categories `rankdef`, `near_sing`, `illcond`, `cascade`, `borderline`,
`saddle_rankdef`, `stokes`, `wide_frontal`, `mc64_resistant`) was
preserved. New rows were assigned by name-prefix heuristic, verified
against the SuiteSparse problem-kind metadata:

- `saddle` — `aug3dcqp`, `mario002`, `tuma2`, `stokes64s`,
  `a0nsdsil`, `a2nnsnsl`, `cont-201`, `cont-300`, `k1_san`,
  `olesnik0`. (Note: `aug2d`, `aug2dc`, `aug3d` were initially included
  but dropped — see "Pathologies" below.)
- `pde` — `brainpc2`, `darcy003`, `sit100`, `helm2d03`, `dawson5`.
- `opt` — `qpband`, `dtoc`, `bloweya`, `blockqp1`, `ncvxqp5`,
  `ncvxqp7`, `ncvxqp9`, `boyd1`. Non-convex QPs and dual-form bound
  constraints.
- `mech` — Boeing `bcsstk35`, `bcsstk37`, `bcsstk39`, `nasa1824`.
  Indefinite stiffness blocks only (see SPD filter below).
- `dense` — `exdata_1`-style high-fill matrices: `sparsine`,
  `copter2`, `linverse`, `spmsrtls`, `laser`.
- `indef` — catch-all for the Schenk_IBMNA `c-XX` family (circuit
  / nonlinear-arithmetic Jacobians) and GHS_indef mirror copies
  (`c-58`..`c-72`). 59 rows total. Schenk_IBMNA is by far the largest
  group; rolling them under `indef` rather than fragmenting prevents
  the report from becoming dominated by a single semantic bucket.

## SPD filter (what was excluded)

The issue requires "indefinite stiffness blocks only — skip the SPD
ones" for Boeing. Group-level posdef metadata was scraped from each
matrix's catalog page (`Positive Definite: yes/no` field). Excluded
because SPD:

| Group         | Matrix(es)                                             |
| ------------- | ------------------------------------------------------ |
| Boeing        | `bcsstk34`, `bcsstk36`, `bcsstk38`, `ct20stif`, `msc04515`, `msc10848`, `msc23052` |
| Schenk_AFE    | `af_0_k101` .. `af_5_k101`, `af_shell3/4/7/8`          |
| GHS_indef     | `bloweybq` (already in manifest as opt baseline; kept) |

`bloweybq` is flagged "Positive Definite: yes" but was in the original
manifest as a PD-baseline `opt` row; kept for back-compat.

## Schenk_AFE skipped entirely

Every Schenk_AFE matrix is ≥ 500k rows. The issue's GHS_indef tier
specifies `n ≤ 100k`; applying that same size filter to Schenk_AFE
left zero candidates. Of the 16 matrices in the group, 10 are SPD
(`af_*_k101` family + half of `af_shell*`); the remaining indefinite
shells (`af_shell1/2/5/6/9/10`) are all 504k–1.5M rows and would each
take several minutes to factor, blowing the 1-hour budget on a single
add. Documented as a future-work item (issue followup) in the
`tried-and-rejected` log.

## Schenk_IBMNA size selection

All Schenk_IBMNA matrices satisfying `n ≤ 100k` were included — 49 of
52 (the three excluded are `c-73`, `c-73b`, `c-big`, all with n > 100k).
This produced 49 indef rows in one swoop. The full sweep was kept
rather than subsampling because (a) they factor fast (sub-second to
~10s each), (b) they form a coherent series (Schenk's IBM Nonlinear
Arithmetic benchmark suite) and subsampling would hide structural
patterns that the future-work analyzer can pick up, and (c) the
category roll-up already places them all under `indef`.

## Pathologies discovered while fetching/factoring

### 1. Pattern-only Boeing matrices (`%%MatrixMarket coordinate pattern`)

`nasa2910` and `nasa4704` ship as **pattern** matrices (no numerical
values, only sparsity structure). Feral's matrix-market reader
(`src/io/mtx.rs`) requires `coordinate real symmetric` headers and
errors out with:

```
read_mtx IoError("...nasa4704.mtx: unsupported header
 '%%MatrixMarket matrix coordinate pattern symmetric'
 (expected: %%MatrixMarket matrix coordinate real symmetric)")
```

Dropped from manifest. Note: `nasa1824` is pattern *or* real depending
on the SuiteSparse release; the tarball we fetched contains
`coordinate real symmetric`, so it is retained.

### 2. Integer-coded GHS_indef saddle matrices

`aug2d`, `aug2dc`, `aug3d` ship as `%%MatrixMarket coordinate integer
symmetric`. Same reader rejection mode. Dropped from manifest.

Both classes (pattern, integer) are documented in
`dev/tried-and-rejected.md` so a future session knows not to re-add
them without first extending `mtx.rs` to handle them.

### 3. nnz-column convention

The existing manifest's `nnz` column stores the value from each
matrix's `.mtx` header (which is the lower-triangle nnz for a
symmetric MM matrix, sometimes called "structural nnz"). The
SuiteSparse catalog page reports a different number (typically the
full symmetric nnz, `2 * lower - diag`). The new rows were written
with catalog values first and then rewritten to match the existing
convention via `read_nnz()` on each downloaded `.mtx`. Net effect:
the manifest is internally consistent; the `nnz` column reflects the
literal MM header value.

## Runtime budget verification

Wall-clock for `python3 external_benchmarks/stress/run.py
--time-limit 3600` on the full 125-row manifest (123 with matrices on
disk, since 2 of the 3 cuter_kkt rows have no local source in this
worktree):

```
real    4m35.174s
user    7m36.408s
sys     0m21.398s
```

Comfortably below the 1-hour acceptance ceiling — 13× headroom.
Per-matrix factor times are dominated by the largest GHS_indef rows
(`d_pretok` 182k×885k nnz at 144 ms, `turon_m` 189k×912k at 166 ms);
Schenk_IBMNA matrices factor in sub-millisecond to ~10 ms each.

If the suite grew enough to exceed 1 h, the natural partition is to
keep the manifest as the "extended" suite and add a `manifest-core.tsv`
with ~20 hand-curated rows for fast CI gating. That partition would
be architectural and would land in `decisions.md`; not needed today.

## report.py stability

Two back-to-back invocations of `report.py` produce byte-identical
output:

```
$ python3 external_benchmarks/stress/report.py > /tmp/report_a.txt
$ python3 external_benchmarks/stress/report.py > /tmp/report_b.txt
$ diff /tmp/report_a.txt /tmp/report_b.txt && echo STABLE
STABLE
```

(`report.py` has no nondeterminism — it only re-reads cached sidecars
and sorts by `(category, n, name)`.)

## Final per-category counts

```
category        total     ok   flag   miss
borderline          3      0      0      3    (cuter_kkt; data/ absent in this worktree)
cascade             2      2      0      0
dense               6      6      0      0
illcond             2      2      0      0
indef              59     59      0      0
mc64_resistant      1      1      0      0
mech                4      4      0      0
near_sing           3      3      0      0
opt                15     15      0      0
pde                 5      5      0      0
rankdef             6      6      0      0
saddle             15     15      0      0
saddle_rankdef      2      2      0      0
stokes              1      1      0      0
wide_frontal        1      1      0      0
total             125    122      0      3
```

Acceptance gate (`report.py` exit code) is **0** — no matrix flagged
on the M3 corpus. The 3 missing rows are pre-existing borderline
cuter_kkt FBRAIN3LS samples whose source files live under the main
checkout's `data/matrices/kkt/`, not the agent worktree; outside the
scope of this expansion.

## Stokes / driven-cavity from `data/matrices/kkt-mittelmann/`

The issue mentions moving Stokes or driven-cavity entries from the
in-repo `data/matrices/kkt-mittelmann/` directory under
`kkt_mittelmann`. Inspection of that directory (in the main checkout
at `/Users/jkitchin/projects/feral/data/matrices/kkt-mittelmann/`)
shows 47 family directories — `arki0003`, `bearing_400`,
`camshape_6400`, `clnlbeam`, `cont5_*`, `dtoc*`, `henon120`,
`marine_1600`, etc. — but **no Stokes or driven-cavity entries** in
the list. The closest matches are `cont5_*` (continuous PDE control),
which is a different class. The two Stokes matrices that are in the
stress suite (`stokes64`, `stokes128`, plus new `stokes64s`) all come
from SuiteSparse GHS_indef and were already correctly grouped under
`saddle` in the existing manifest; no relocation was needed.

The synth `stokes_q1p0_8` row (category `stokes`) is in a different
category bucket on purpose — it's an oracle-checked synthetic with
known null space (2 spurious pressure modes), not a benchmark
saddle-point.
