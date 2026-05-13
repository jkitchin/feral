# feral as an Ipopt linear solver — C shim research note

Date: 2026-05-13
Status: research phase (no code yet)
Target: Ipopt 3.14.x (`ref/Ipopt`)

## Goal

Make feral pluggable into canonical (C++) Ipopt 3.14 as a
replacement for MUMPS at the `linear_solver` option. The user-
visible end state is: `IpoptApplication.Options()->SetStringValue
("linear_solver", "feral")` selects feral, and Ipopt drives feral
through its existing `SparseSymLinearSolverInterface` lifecycle
exactly the way it drives MUMPS today.

This is **separate from** ripopt's existing `feral_direct.rs` /
`feral_hybrid.rs` / `feral_iterative.rs` wiring at
`/Users/jkitchin/projects/ripopt/src/linear_solver/`. That path
calls feral from a Rust IPM via the Rust API. This note is
about the C++ Ipopt → C ABI → feral path.

## What Ipopt expects from a linear solver

### The abstract base class

`ref/Ipopt/src/Algorithm/LinearSolvers/IpSparseSymLinearSolverInterface.hpp:98-256`
defines the pure-virtual surface a linear solver must implement.
The relevant methods (all `Index = int`, `Number = double`):

- `InitializeImpl(const OptionsList&, const std::string& prefix)
  -> bool` (line 125). Called once per optimization run. Read
  pivot tolerance and other options here.
- `MatrixFormat() const -> EMatrixFormat` (line 231). Returns
  one of `Triplet_Format`, `CSR_Format_0_Offset`,
  `CSR_Format_1_Offset`, `CSR_Full_Format_0_Offset`,
  `CSR_Full_Format_1_Offset` (enum at lines 102-114).
- `InitializeStructure(Index dim, Index nonzeros, const Index*
  ia, const Index* ja) -> ESymSolverStatus` (line 139). Called
  once per structure; do symbolic analysis if cheap.
- `GetValuesArrayPtr() -> Number*` (line 155). Return a pointer
  to a `nonzeros`-sized buffer **owned by the solver**; the
  caller writes A's nonzero values into this buffer before each
  factor. This is the zero-copy hand-off.
- `MultiSolve(bool new_matrix, const Index* ia, const Index*
  ja, Index nrhs, Number* rhs_vals, bool check_NegEVals, Index
  numberOfNegEVals) -> ESymSolverStatus` (line 190). If
  `new_matrix`: factor (using values previously deposited in
  the buffer). Then solve in place on `rhs_vals` (size
  `nrhs*n`, column-major).
- `NumberOfNegEVals() const -> Index` (line 207). Inertia query
  after the most recent factor.
- `IncreaseQuality() -> bool` (line 220). Escalate pivot
  tolerance; return false if already at max.
- `ProvidesInertia() const -> bool` (line 226). For feral: true.

Default-implemented (skip):
`ProvidesDegeneracyDetection` / `DetermineDependentRows` (lines
240, 248). feral does not need these.

### The MUMPS reference implementation

`ref/Ipopt/src/Algorithm/LinearSolvers/IpMumpsSolverInterface.cpp`
is the closest analog. Key call pattern Ipopt drives during a
typical solve:

1. `InitializeImpl` (cpp ~138 + 372): read `mumps_pivtol`,
   `mumps_pivtolmax`, etc. Initialize the MUMPS struct
   (`JOB=-1`).
2. `MatrixFormat` returns `Triplet_Format` (constexpr).
3. `InitializeStructure(n, nnz, ia, ja)` (cpp:372-406): store
   `n`, allocate the values buffer of size `nnz`, park `irn=ia`
   and `jcn=ja` (cast away const — MUMPS struct doesn't have
   const pointers).
4. `GetValuesArrayPtr` returns `mumps_->a`.
5. Caller fills the values buffer.
6. `MultiSolve(new_matrix=true, ia, ja, nrhs, rhs, ...)`
   (cpp:270-329): if no symbolic done yet, `SymbolicFactorization`
   (`JOB=1`, cpp:408); then `Factorization(check_NegEVals,
   numberOfNegEVals)` (`JOB=2`, cpp:471); on success,
   `Solve(nrhs, rhs_vals)` (`JOB=3`, cpp:566).
7. Caller may call `MultiSolve(new_matrix=false, ...)` again
   for more RHS against the same factor — skips factor, just
   back-solves.
8. On `WRONG_INERTIA` or `SINGULAR` Ipopt calls
   `IncreaseQuality` (cpp:615-633) and retries — or perturbs
   the matrix via `PDPerturbationHandler` and refills the
   values buffer (caller sets `new_matrix=true`).

### Matrix format

`EMatrixFormat` enum
(`IpSparseSymLinearSolverInterface.hpp:102-114`):

- `Triplet_Format` — lower triangle, 1-based, **raw** (the
  values buffer the solver receives can contain duplicate
  `(i,j)` entries; the solver must sum them).
- `CSR_Format_0_Offset` — upper triangle, 0-based, CSR.
  **Deduplicated by Ipopt's `TripletToCSRConverter`** before
  the values buffer is filled.
- `CSR_Format_1_Offset` — upper triangle, 1-based, CSR.
  Same dedup.
- `CSR_Full_Format_{0,1}_Offset` — both triangles stored.

MUMPS picks `Triplet_Format` because MUMPS itself sums
duplicates in its assembly step. The MA57 / PARDISO C++
interfaces (`IpMa57TSolverInterface.cpp`,
`IpPardisoMKLSolverInterface.cpp`) pick `CSR_*_Offset`
because they want a clean deduplicated CSR.

**Feral picks `CSR_Format_0_Offset`.** Critical
observation: for a symmetric matrix, CSR-of-upper-triangle
is byte-identical to CSC-of-lower-triangle (just relabel
"row index" → "col index"). Feral's existing `CscMatrix`
is 0-based, lower-triangle CSC with sorted row indices and
no duplicates (`from_triplets` deduplicates on construction,
`src/sparse/csc.rs:35-100`). The layout match is exact:

- `n` ↔ `dim`
- `col_ptr[]` ↔ Ipopt's CSR `ia[]` (length `n+1`, 0-based,
  offsets into row-index array)
- `row_idx[]` ↔ Ipopt's CSR `ja[]` (length `nnz`, 0-based,
  row indices within each "column" — which is what Ipopt
  calls a "row" of the upper triangle).
- `values[]` is the value buffer feral exposes.

`TSymLinearSolver::InitializeStructure` (cpp:330-371) hands
the converter the raw triplet structure once, gets back
the CSR pattern, and calls
`solver_interface_->InitializeStructure(dim_, nonzeros_compressed_,
ia, ja)`. On each `GiveMatrixToSolver` call
(cpp:453-533) it allocates a temporary
`Number[nonzeros_triplet_]`, fills it from the SymMatrix,
applies scaling if any, then calls
`triplet_to_csr_converter_->ConvertValues(...)` (cpp:528)
which scatters/sums the triplet values into the CSR values
buffer feral previously exposed via `GetValuesArrayPtr`.

**Implications for the C ABI**:

- No dedup map inside feral. The values buffer feral
  exposes via `feral_get_values_ptr` is literally
  `csc.values.as_mut_ptr()`.
- The structure feral receives from
  `feral_initialize_structure` is already deduplicated and
  sorted. Feral's `CscMatrix` construction reduces to a
  zero-allocation pattern wrap (we are handed `col_ptr`
  and `row_idx`; we allocate the `values` vec ourselves).
- Trade-off: Ipopt allocates one `Number[nonzeros_triplet_]`
  scratch buffer per `GiveMatrixToSolver`. This matches
  what every CSR-mode HSL solver pays today, so it is
  uncontroversial.

### Inertia / status return values

`ESymSolverStatus` enum at `IpSymLinearSolver.hpp:19-33`:

- `SYMSOLVER_SUCCESS = 0` — factor and (if requested) solve
  succeeded; `NumberOfNegEVals()` is valid.
- `SYMSOLVER_SINGULAR` — caller (`PDPerturbationHandler`)
  bumps `delta_c` (constraint regularization) and refills
  values + retries.
- `SYMSOLVER_WRONG_INERTIA` — caller bumps `delta_x`
  (Hessian perturbation) and refills + retries. Returned by
  MUMPS at cpp:555-561 when `check_NegEVals &&
  NumberOfNegEVals() != numberOfNegEVals`.
- `SYMSOLVER_CALL_AGAIN` — caller refills values buffer and
  re-invokes `MultiSolve`. MUMPS uses this when
  `pivtol_changed_ && !new_matrix` to force a fresh value
  load.
- `SYMSOLVER_FATAL_ERROR` — abort optimization.

Ipopt does not separately consume "number of zero
eigenvalues"; singularity is signaled by the status code.

### No C ABI plugin point

Searched `IpAlgBuilder.cpp` and `IpLibraryLoader.cpp`.
Ipopt 3.14 has **no C ABI for adding a new linear solver**:

- `linear_solver=custom` (cpp:517-520, 575-583) injects a
  custom `AugSystemSolver` higher up the stack, after Sigma
  reduction. Not the right hook for replacing MUMPS at the
  symmetric-indefinite-solve layer.
- HSL solvers are dlopen-loaded via `IpLibraryLoader`
  (`Common/IpLibraryLoader.{hpp,cpp}`), but only the
  underlying HSL Fortran symbols are loaded that way; the
  C++ class `Ma27TSolverInterface` is still compiled into
  libipopt and just resolves Fortran symbols at runtime
  (`IpMa27TSolverInterface.cpp:180-189`). This is precedent
  for loading a numerical kernel via dlopen — **not**
  precedent for dlopen-loading a
  `SparseSymLinearSolverInterface` subclass.

**Conclusion**: integration requires a thin C++ shim class
`FeralSolverInterface : public SparseSymLinearSolverInterface`
that forwards each pure-virtual method to a stable C ABI
exported by feral. Two sub-options for how the shim gets
into Ipopt:

1. **Patch Ipopt** (`IpAlgBuilder.cpp`). Add a
   `linear_solver=feral` branch around cpp:509-515 (alongside
   `mumps`). Construct `new FeralSolverInterface()`. The
   shim and feral libs link statically or dynamically as
   normal libraries. *Cleanest end-state; requires a custom
   Ipopt build.*

2. **Out-of-tree custom AugSystemSolver**. Use the
   `linear_solver=custom` path. The shim then has to do
   more work — `AugSystemSolver` operates after Sigma
   reduction, so we'd reimplement either `StdAugSystemSolver`
   or extract its Sigma assembly. *More invasive; works
   against an unmodified Ipopt.*

Recommendation: option 1. We already need a feral fork or
patch to land the `linear_solver=feral` option string in
`IpAlgBuilder`'s registered option list anyway; the patch
is ~30 lines.

## What feral currently exposes

`src/lib.rs:14-39` re-exports:

- `Solver` (high-level: factor / solve / increase_quality /
  inertia / num_negative_eigenvalues — already shaped like
  Ipopt's interface)
- `CscMatrix` (lower-triangle CSC, sorted row indices, sums
  duplicates via `from_triplets` at `src/sparse/csc.rs:35-100`)
- `Inertia`, `FeralError`, `FactorStatus`, `QualityLevel`,
  `NumericParams`, `SchurBlock`

`numeric::solver::Solver` (src/numeric/solver.rs:102-460) is
already structured around the same lifecycle Ipopt expects:

- `factor(matrix, check_inertia)` — refactor or symbolic+factor
- `solve_many(rhs, nrhs)` / `solve_many_refined` — RHS solves
- `increase_quality() -> bool` — two-stage escalation
  (scaling, then pivot threshold)
- `num_negative_eigenvalues() / inertia() / provides_inertia()`
- `FactorStatus::{Success, Singular, WrongInertia, FatalError}`
  — maps cleanly onto `ESymSolverStatus`

This is convenient: the C ABI is essentially "expose `Solver`
through `extern "C"`".

**No FFI surface exists yet.** Feral has no `cdylib` /
`staticlib` crate-type, no `extern "C"` functions, no header
file generation. All of that has to be added.

## Proposed C ABI

A new **workspace-member crate** `feral-capi/` alongside the
core `feral` crate. The core crate's `Cargo.toml`
`[lib].crate-type` stays at `["rlib"]`; no cdylib in core.

```
# Top-level Cargo.toml becomes a workspace root
[workspace]
members = [".", "feral-capi"]

# feral-capi/Cargo.toml
[package]
name = "feral-capi"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "staticlib", "rlib"]

[dependencies]
feral = { path = ".." }
```

Decision recorded in `dev/decisions.md` 2026-05-13:
"`feral-capi` as a separate workspace member, not a core
feature". The "pure Rust core" constraint becomes a
crate-level invariant — all `extern "C"` and FFI-boundary
`unsafe` lives in `feral-capi/src/lib.rs`, audited in one
place. Default `cargo build` in workspace root produces an
rlib for `feral` and a cdylib for `feral-capi`; the cdylib
is consumed only by the C++ shim.

C ABI surface (sketch — to be detailed in the plan):

```c
// Opaque handle.
typedef struct FeralSolver FeralSolver;

// Status codes mirroring ESymSolverStatus.
typedef enum {
    FERAL_SUCCESS = 0,
    FERAL_SINGULAR = 1,
    FERAL_WRONG_INERTIA = 2,
    FERAL_CALL_AGAIN = 3,
    FERAL_FATAL_ERROR = 4,
} FeralStatus;

// Lifecycle.
FeralSolver* feral_create(void);
void         feral_destroy(FeralSolver*);

// Options (string-keyed, mirrors Ipopt's OptionsList convention).
int feral_set_option_num(FeralSolver*, const char* key, double v);
int feral_set_option_int(FeralSolver*, const char* key, int v);
int feral_set_option_str(FeralSolver*, const char* key,
                         const char* v);

// Structure phase. CSR (= CSC-of-lower-tri on a symmetric matrix),
// 0-based, upper triangle (Ipopt's convention) / lower triangle
// CSC (feral's convention) — same layout.
//
// ia: length n+1, column pointers into ja.
// ja: length nnz, row indices within each column.
// Already deduplicated and sorted by Ipopt's TripletToCSRConverter.
//
// Stores n + col_ptr + row_idx; allocates CSC values vec;
// hands its ptr back via feral_get_values_ptr.
FeralStatus feral_initialize_structure(
    FeralSolver*, int n, int nnz,
    const int* ia, const int* ja);
double*     feral_get_values_ptr(FeralSolver*);

// Numerical phase.
FeralStatus feral_factor(
    FeralSolver*,
    int check_neg_evals, int expected_neg_evals);

// Solve in place. rhs is column-major, n*nrhs.
FeralStatus feral_solve(
    FeralSolver*, int nrhs, double* rhs);

// Inertia query.
int feral_num_neg_evals(const FeralSolver*);

// Quality escalation. Returns 1 on success, 0 if exhausted.
int feral_increase_quality(FeralSolver*);
```

**Design notes**:

1. **CSR_Format_0_Offset is byte-identical to feral's
   `CscMatrix` layout**. Ipopt does the triplet→CSR
   conversion + dedup in `TSymLinearSolver`
   (cpp:330-371 for structure, cpp:453-533 for values).
   `feral_initialize_structure` just wraps the `ia`/`ja`
   arrays into a `CscMatrix` shell, allocates the
   `values` vec of size `nnz`, and returns its mutable
   pointer via `feral_get_values_ptr`. No dedup map
   needed inside feral.

2. **`SYMSOLVER_CALL_AGAIN`** maps to feral's two-stage
   `increase_quality` triggering a re-factor on next call.
   The shim sets a `pending_quality_change` flag inside
   `feral_increase_quality`; the next `feral_factor` clears
   it and forces a fresh factor.

3. **No `unsafe` in feral core**. All `unsafe` lives in
   `src/capi.rs` at the FFI boundary. Each `unsafe` block
   gets a safety comment per CLAUDE.md hard rules.

4. **Error handling**: feral core returns `Result<_,
   FeralError>`. The C ABI catches the error and returns
   `FERAL_FATAL_ERROR`. A `feral_last_error_message(char*
   buf, int len)` getter exposes the message for logging.
   No panics across the FFI boundary — the shim wraps
   every entry in `std::panic::catch_unwind` and converts
   panics to `FERAL_FATAL_ERROR`.

## Proposed C++ shim

**Resolved 2026-05-13 (decisions.md):** the C++ shim lives
in-tree at `feral/feral-ipopt-shim/` during bring-up.
Split to its own repo once the C ABI stabilizes (semver
1.0) and/or multiple shim variants need to coexist.

A new sub-directory `feral-ipopt-shim/` at the feral repo
root containing:

- `FeralSolverInterface.hpp` — subclass of Ipopt's
  `SparseSymLinearSolverInterface`.
- `FeralSolverInterface.cpp` — implements each pure-virtual
  by calling into the C ABI; manages the opaque
  `FeralSolver*` lifetime via RAII.
- `feral_c_api.h` — the C ABI header (committed alongside,
  used by both the shim and any other consumer).
- `CMakeLists.txt` — builds `libferal_ipopt_shim.a` or
  `.so`, depending on Ipopt's link mode. Links against
  `libferal.{a,so}` (the cdylib/staticlib from cargo).
- `patches/ipopt-3.14-add-feral-solver.patch` — the ~30-line
  patch to `IpAlgBuilder.cpp` adding the
  `linear_solver=feral` branch.
- `tests/` — at minimum a smoke test factoring a 5×5 KKT
  triplet through the shim and comparing inertia to feral's
  native Rust API.

**Repo layout decided (in-tree):** the shim is *not* a
Cargo workspace member — it's a sibling directory with
its own CMake build, consumed by builders who want the
Ipopt integration. The Rust workspace contains only
`feral` (core) and `feral-capi` (FFI).

## Lifecycle mapping (Ipopt method → shim → feral C ABI)

| Ipopt method | Shim action | C ABI call(s) |
|---|---|---|
| `InitializeImpl` | Read OptionsList; forward each | `feral_set_option_*` (many) |
| `MatrixFormat` | constexpr | (none) returns `CSR_Format_0_Offset` |
| `InitializeStructure` | Pass through | `feral_initialize_structure` |
| `GetValuesArrayPtr` | Pass through | `feral_get_values_ptr` |
| `MultiSolve(new_matrix=true)` | factor + solve | `feral_factor` then `feral_solve` |
| `MultiSolve(new_matrix=false)` | solve only | `feral_solve` |
| `NumberOfNegEVals` | Pass through | `feral_num_neg_evals` |
| `IncreaseQuality` | Pass through | `feral_increase_quality` |
| `ProvidesInertia` | constexpr | (none) returns `true` |
| destructor | Pass through | `feral_destroy` |

## Open questions (resolve before plan / code)

1. ~~Repo layout~~ **RESOLVED 2026-05-13** (decisions.md
   "`feral-ipopt-shim` lives in-tree during bring-up"):
   in-tree at `feral/feral-ipopt-shim/`. Split when the
   C ABI hits semver 1.0 and/or multiple shim variants
   need to coexist.

2. ~~Triplet duplicate handling~~ **RESOLVED 2026-05-13**.
   `TSymLinearSolver::InitializeStructure` (cpp:330-371)
   and `GiveMatrixToSolver` (cpp:453-533) confirm:
   - `Triplet_Format` solvers receive **raw triplets**
     with duplicates (the solver sums them — MUMPS does
     this in its assembly step).
   - `CSR_Format_*` solvers receive the structure and
     values **deduplicated** by `TripletToCSRConverter::
     InitializeConverter` (once) and `ConvertValues`
     (per factor, scatters/sums triplet values into the
     CSR buffer at cpp:528).

   Decision: pick `CSR_Format_0_Offset`. CSR-of-upper-tri
   on a symmetric matrix is the same layout as feral's
   `CscMatrix` (CSC-of-lower-tri), 0-based, deduplicated,
   sorted within group. No dedup map needed inside feral;
   the C ABI is correspondingly simpler.

3. **Option string mapping**: Ipopt's MUMPS interface
   registers options under prefix `mumps_*` (`mumps_pivtol`,
   `mumps_pivtolmax`, `mumps_pivot_order`, ...). What's
   the feral analog set?
   - `feral_pivtol` ↔ `BunchKaufmanParams.pivot_threshold`
   - `feral_pivtolmax` ↔ `NumericParams.pivtol_max`
   - `feral_scaling` ↔ `ScalingStrategy` ("none"|"infnorm")
   - `feral_parallel` ↔ `Solver::with_parallel(bool)`
   - `feral_print_level` — pass through to feral's
     internal logger.
   *Resolution*: enumerate in the plan; pick conservative
   defaults that match MUMPS's defaults so a drop-in
   replacement doesn't change Ipopt's iteration count on
   well-conditioned problems.

4. **Build integration**: how does the feral cdylib get
   onto the Ipopt build's link line? Two paths:
   - Cargo workspace + cmake `find_package` style. Cleanest
     for end-users.
   - Manual: user runs `cargo build --release`, sets
     `LIBRARY_PATH` and `LD_LIBRARY_PATH`, then configures
     Ipopt with `--with-feral`. Reasonable for bring-up;
     ugly for shipping.
   *Resolution*: bring-up uses manual path; plan a proper
   `pkg-config` file once the shim works.

5. ~~Constraint discipline~~ **RESOLVED 2026-05-13**
   (decisions.md "`feral-capi` as a separate workspace
   member, not a core feature"): adding a C ABI does not
   violate the "pure Rust core" constraint because the C
   ABI lives in a sibling crate (`feral-capi`), not in
   `feral` itself. Core stays rlib-only with no FFI.
   The constraint refers to the *core's runtime / build
   dependencies* — feral exporting a C-callable surface
   is the opposite direction (downstream consumers, not
   upstream dependencies).

6. **Inertia semantics on consensus-excluded matrices**.
   feral's inertia gate is "agree with at least one of
   MUMPS/SSIDS on definitive matrices" (CLAUDE.md). Ipopt
   does not query consensus; it compares against
   `numberOfNegEVals` computed from the KKT structure.
   For KKT matrices from a well-posed NLP this is the
   same value MUMPS would report, so feral should agree
   here too. No new constraint, but worth recording: the
   shim does not adjust inertia returns — what feral's
   `Inertia` says is what Ipopt sees.

7. **Cancellation / signal handling**. MUMPS catches Ctrl-C
   in a fragile way (Fortran I/O quirks). feral has no
   in-flight cancellation API. The shim cannot safely
   interrupt mid-factor. *Resolution*: acceptable for the
   first version; document as known limitation.

## What this note does not cover

- Detailed plan / phasing — that goes in
  `dev/plans/feral-ipopt-shim.md` next.
- Tests against `IpoptApplication`-level NLPs (CUTEst, the
  Ipopt sample problems). Until the shim builds, these are
  blocked.
- Performance comparison vs. MUMPS at the Ipopt level. The
  small-front bench-ratio gap (issues #11/#12/#13) is a
  feral-internal metric; the meaningful Ipopt-level metric
  is end-to-end wall-clock on a real NLP corpus, after
  the shim works.

## References

- `ref/Ipopt/src/Algorithm/LinearSolvers/IpSparseSymLinearSolverInterface.hpp`
  (abstract base class, lines 98-256)
- `ref/Ipopt/src/Algorithm/LinearSolvers/IpMumpsSolverInterface.{hpp,cpp}`
  (MUMPS implementation; closest analog)
- `ref/Ipopt/src/Algorithm/LinearSolvers/IpTSymLinearSolver.{hpp,cpp}`
  (wrapper that calls into a `SparseSymLinearSolverInterface`)
- `ref/Ipopt/src/Algorithm/LinearSolvers/IpSymLinearSolver.hpp:19-33`
  (`ESymSolverStatus` enum)
- `ref/Ipopt/src/Algorithm/IpAlgBuilder.cpp:427-552`
  (`SymLinearSolverFactory`, dispatch on `linear_solver` option;
  `mumps` branch at 509-515)
- `ref/Ipopt/src/Common/IpLibraryLoader.{hpp,cpp}` (dlopen pattern,
  precedent for loading a numerical kernel — not a
  `SparseSymLinearSolverInterface`)
- `ref/Ipopt/src/Algorithm/LinearSolvers/IpMa27TSolverInterface.cpp:180-189`
  (HSL precedent — Fortran symbols loaded at runtime, C++ class
  compiled in)
- `src/lib.rs:14-39` (feral's current public API)
- `src/numeric/solver.rs:102-460` (`Solver` — already
  Ipopt-shaped)
- `src/sparse/csc.rs:35-100`
  (`CscMatrix::from_triplets`, duplicate handling)
